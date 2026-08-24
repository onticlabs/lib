"""Distributed-training helpers that aren't tied to any specific model wrapper.

The single export here, `avg_log_dict_across_ranks`, is used by `train.py`'s
shared train/val log path so the same metric-reduction works for both
`NVSWrapper` and `DynWrapper.validation_step` (they each return a `log_dict`).

Design goals:

* **No wrapper-side changes**: each wrapper's `training_step` / `validation_step`
  returns its log_dict as before. The all-reduce happens in the shared caller.
* **No-op when single-rank**: passthrough for interactive workstation /
  single-GPU runs, so dev-test parity holds.
* **Don't mutate caller's dict** — return a new dict; leave incoming tensors
  alone (caller may still want the raw per-rank tensor).
* **Hang-safe under rank-divergent key sets.** Some keys appear only on rank 0
  (e.g. `gs_plots/opacity_scale_stats`, and the scalars added by
  `log_gaussian_stats_plot`, which early-returns when `wandb.run is None`).
  We `all_gather_object` the key categories first and only reduce the
  intersection so a key missing on one rank can never deadlock NCCL. The
  rank-local value is kept untouched for any non-common key — useful for
  rank-0 viz artifacts like `wandb.Image`.
* **One batched all-reduce for scalars** instead of one collective per key,
  since NVS's debug logging adds dozens of scalar entries per step.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _classify(log_dict: dict) -> tuple[set[str], dict[str, tuple[int, ...]]]:
    """Split a log_dict into reducible scalar keys and reducible-tensor keys-with-shape.

    Everything else (strings, None, lists, wandb.Image, int tensors, bool) is left
    out — it will pass through unchanged.
    """
    scalar_keys: set[str] = set()
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for k, v in log_dict.items():
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                tensor_shapes[k] = tuple(v.shape)
            # else: integer / bool tensor → passthrough
        elif isinstance(v, bool):
            pass  # bool is a Python int but we don't want to average it
        elif isinstance(v, (int, float)):
            scalar_keys.add(k)
        # else: str, None, list, wandb.Image, ... → passthrough
    return scalar_keys, tensor_shapes


def avg_log_dict_across_ranks(log_dict: dict) -> dict:
    """Return a copy of `log_dict` with numeric entries averaged across DDP ranks.

    Reducible entries (Python int/float, floating-point tensor) are AVG'd across
    ranks. Entries missing on at least one rank, or floating tensors whose shape
    differs across ranks, are kept as their rank-local value and not reduced —
    so rank-0-only viz artifacts (e.g. `wandb.Image`) pass through cleanly and
    one rank's extra debug keys can't deadlock NCCL.

    On a non-distributed run, this is a shallow-copy passthrough.
    """
    if not _is_distributed():
        return dict(log_dict)

    default_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Start from a shallow copy: any key/value not touched below survives as-is.
    out: dict[str, Any] = dict(log_dict)

    local_scalars, local_tensors = _classify(log_dict)

    # Sync key sets + tensor shapes so all ranks agree on what to reduce. Use
    # all_gather_object once instead of one collective per key — cheaper, and
    # the only way to detect a key/shape that exists on a strict subset of ranks.
    world_size = dist.get_world_size()
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, (local_scalars, local_tensors))

    common_scalars = sorted(set.intersection(*[g[0] for g in gathered]))
    # A tensor key is only reducible if every rank has it AND every rank's shape matches.
    all_tensor_keys = set.intersection(*[set(g[1].keys()) for g in gathered])
    common_tensors = sorted(k for k in all_tensor_keys if len({g[1][k] for g in gathered}) == 1)

    # Batch the scalars into one collective.
    if common_scalars:
        scalar_tensor = torch.tensor(
            [float(log_dict[k]) for k in common_scalars],
            device=default_device,
            dtype=torch.float32,
        )
        dist.all_reduce(scalar_tensor, op=dist.ReduceOp.AVG)
        for k, v in zip(common_scalars, scalar_tensor.tolist()):
            out[k] = v

    # Tensors: one collective each (shapes are agreed but may vary across keys).
    for k in common_tensors:
        v = log_dict[k]
        t = v.detach().to(default_device, copy=False).float().clone()
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        out[k] = t.to(v.device)

    return out
