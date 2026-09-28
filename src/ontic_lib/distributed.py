"""Multi-process training helpers for ``torchrun`` launches.

`setup` reads torchrun's environment and initialises the process group; `sync_gradients`
averages gradients once per optimizer step for loops that cannot be wrapped in
``DistributedDataParallel`` (several backward passes per step, as in chunked rollouts);
`broadcast_module` and `broadcast_flag` pin rank 0's state; `avg_log_dict_across_ranks`
averages a training ``log_dict`` hang-safely. Every helper is a no-op in a single process, so
a one-GPU run is the ``world_size == 1`` case of the same training code.

Design goals of `avg_log_dict_across_ranks`:

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
* **SUM then divide** rather than ``ReduceOp.AVG``, which the gloo backend lacks.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


_is_distributed = is_distributed


def is_main() -> bool:
    """True on rank 0 and in a single process: the rank that logs, validates and saves."""
    return not is_distributed() or dist.get_rank() == 0


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def dist_device() -> torch.device:
    """Where collectives put their scratch tensors: the current GPU under NCCL, the CPU otherwise."""
    if is_distributed() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def setup(
    backend: str | None = None,
    timeout: datetime.timedelta = datetime.timedelta(hours=2),
) -> tuple[int, int]:
    """``(rank, world_size)``, initialising the process group from torchrun's environment.

    A plain ``python train.py`` (no ``WORLD_SIZE``, or ``WORLD_SIZE=1``) gets ``(0, 1)`` and
    no process group. Otherwise ``RANK``, ``LOCAL_RANK``, ``MASTER_ADDR`` and ``MASTER_PORT``
    are read by ``init_process_group``; under NCCL (the default with CUDA) the current device is
    set to ``LOCAL_RANK`` first. The long default timeout covers rank 0 running a validation
    while the other ranks wait at the next collective.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return 0, 1
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend=backend, timeout=timeout)
    return dist.get_rank(), dist.get_world_size()


def teardown() -> None:
    """Barrier, then destroy the process group; a no-op without one."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def broadcast_module(module: nn.Module, src: int = 0) -> None:
    """Parameters and buffers of ``src`` to every rank (ranks built from the same seed and files
    should agree already; this pins it, and it is what makes a resumed run identical on all ranks)."""
    if not is_distributed():
        return
    for t in list(module.parameters()) + list(module.buffers()):
        dist.broadcast(t.data, src=src)


def sync_gradients(module_or_params: nn.Module | Iterable[nn.Parameter], *, average: bool = True) -> None:
    """Sum, and by default average, the gradients of the trainable parameters across ranks.

    One all-reduce per (dtype, device) group over a flattened copy, then written back into
    ``p.grad``. A parameter without a gradient on this rank contributes zeros and receives the
    result, so the ranks end the call with identical gradients. Call it after the last backward
    pass of a step and before clipping and the optimizer step.
    """
    if not is_distributed():
        return
    params = module_or_params.parameters() if isinstance(module_or_params, nn.Module) else module_or_params
    groups: dict[tuple[torch.dtype, torch.device], list[nn.Parameter]] = {}
    for p in params:
        if p.requires_grad:
            groups.setdefault((p.dtype, p.device), []).append(p)
    world = dist.get_world_size()
    for group in groups.values():
        grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in group]
        flat = torch.cat([g.reshape(-1) for g in grads])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM)
        if average:
            flat.div_(world)
        for p, g in zip(group, torch.split(flat, [g.numel() for g in grads])):
            if p.grad is None:
                p.grad = g.view_as(p).clone()
            else:
                p.grad.copy_(g.view_as(p))


def gather_dicts(local: dict, dst: int = 0) -> list[dict]:
    """Every rank's small dict (picklable values) collected on ``dst``, in rank order; other
    ranks get ``[]``. A single process gets ``[local]``. For per-rank facts such as GPU
    utilisation that the logging rank merges into one record."""
    if not is_distributed():
        return [local]
    world = dist.get_world_size()
    out: list = [None] * world if dist.get_rank() == dst else None
    dist.gather_object(local, out, dst=dst)
    return list(out) if out is not None else []


def broadcast_flag(flag: bool, src: int = 0) -> bool:
    """``src``'s boolean on every rank: for decisions taken on one rank that every rank must
    follow, such as leaving the training loop on a wall-clock budget."""
    if not is_distributed():
        return bool(flag)
    t = torch.tensor([int(bool(flag))], device=dist_device())
    dist.broadcast(t, src=src)
    return bool(t.item())


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
    if not is_distributed():
        return dict(log_dict)

    default_device = dist_device()

    # Start from a shallow copy: any key/value not touched below survives as-is.
    out: dict[str, Any] = dict(log_dict)

    local_scalars, local_tensors = _classify(log_dict)

    # Sync key sets + tensor shapes so all ranks agree on what to reduce. Use
    # all_gather_object once instead of one collective per key — cheaper, and
    # the only way to detect a key/shape that exists on a strict subset of ranks.
    world = dist.get_world_size()
    gathered: list[Any] = [None] * world
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
        dist.all_reduce(scalar_tensor, op=dist.ReduceOp.SUM)  # SUM then divide: gloo has no AVG
        scalar_tensor.div_(world)
        for k, v in zip(common_scalars, scalar_tensor.tolist()):
            out[k] = v

    # Tensors: one collective each (shapes are agreed but may vary across keys).
    for k in common_tensors:
        v = log_dict[k]
        t = v.detach().to(default_device, copy=False).float().clone()
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        out[k] = t.div_(world).to(v.device)

    return out
