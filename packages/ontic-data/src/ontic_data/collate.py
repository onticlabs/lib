"""Batch collation for :class:`~ontic_data.example.BatchedTempExample` samples."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def worker_init_fn(worker_id: int) -> None:
    """Seed ``random`` and ``numpy`` from the torch worker seed."""
    seed = int(torch.utils.data.get_worker_info().seed) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)


def _truncate(d: dict, T: int, Tm: int) -> dict:
    return {
        k: (v[:Tm] if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == T else v)
        for k, v in d.items()
    }


def harmonize_target_horizon(batch: list[dict]) -> list[dict]:
    """Truncate every sample's target (and actions) to the batch-min horizon.

    Horizon-aware loading decodes ``n_step_state + n_pred(step)`` frames per sample; when
    the step counter crosses a schedule increment between two samples of one batch they
    carry different horizons and ``default_collate`` cannot stack them. Truncating to the
    minimum is safe because the training loop truncates to ``n_pred(step)`` anyway.
    No-op when all horizons agree.
    """
    tgts = [s.get("target") for s in batch]
    if any(not isinstance(t, dict) or "image" not in t for t in tgts):
        return batch
    Ts = [t["image"].shape[0] for t in tgts]
    Tm = min(Ts)
    if all(T == Tm for T in Ts):
        return batch
    for s, T in zip(batch, Ts):
        if T == Tm:
            continue
        s["target"] = _truncate(s["target"], T, Tm)
        if isinstance(s.get("actions"), dict):
            s["actions"] = _truncate(s["actions"], T, Tm)
    return batch


def collate_examples(batch: list[dict]) -> dict:
    """``default_collate`` for everything but ``actions``, which are zero-padded per key."""
    batch = harmonize_target_horizon(batch)

    actions_list = []
    rest_batch = []
    for sample in batch:
        sample = dict(sample)
        actions_list.append(sample.pop("actions", None))
        rest_batch.append(sample)

    collated = default_collate(rest_batch)

    batch_actions = collate_actions(actions_list)
    batch_actions = {k: v for k, v in batch_actions.items() if v is not None}
    if batch_actions:
        collated["actions"] = batch_actions
    return collated


def collate_actions(actions_list: list[dict | None]) -> dict[str, torch.Tensor]:
    """Pad per-sample ``{key: (T, ...) }`` action dicts to ``{key: (B, T_max, ..., D)}``.

    Every loader encodes per-frame presence in the last channel (xyz + presence for
    point actions), so zero-padding marks absent rows / timesteps as presence 0.
    """
    keys: list[str] = []
    for act in actions_list:
        if act is not None:
            keys.extend(act.keys())
    keys = list(set(keys))

    per_key: dict[str, list] = {}
    for k in keys:
        per_key[k] = [act[k] if act is not None and k in act else None for act in actions_list]

    batched: dict[str, torch.Tensor] = {}
    for k, acts in per_key.items():
        timesteps = [a.shape[0] if a is not None else -1 for a in acts]
        if len(timesteps) == 0:
            batched[k] = None
            continue
        max_ix = int(np.argmax(timesteps))
        B = len(acts)
        action_shape = acts[max_ix].shape[:-1]
        D = acts[max_ix].shape[-1]
        if D < 4:
            raise ValueError(
                f"action '{k}' has {D} channels; loaders must emit xyz + a presence bit"
            )
        padded = torch.zeros(B, *action_shape, D)
        for i, act in enumerate(acts):
            if act is None:
                continue
            padded[i, : act.shape[0]] = act
        batched[k] = padded
    return batched
