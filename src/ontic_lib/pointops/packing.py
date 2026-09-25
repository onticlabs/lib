"""Packed ("offset") point layout: N points of B groups concatenated.

``batch: (N,) int64`` is the group id of each point, sorted ascending.
``offset: (B,) int64`` holds cumulative counts with ``offset[-1] == N`` and no
leading zero. Padded layouts are ``(*G, K, ...)`` with a ``(*G, K)`` bool mask;
the flattened leading dims ``*G`` (row-major) are the group ids.
"""

from __future__ import annotations

import torch
from torch import Tensor


def offset_to_counts(offset: Tensor) -> Tensor:
    """``(B,)`` cumulative counts -> ``(B,)`` per-group counts."""
    return torch.diff(offset, prepend=offset.new_zeros(1))


def offset_to_batch(offset: Tensor) -> Tensor:
    """``(B,)`` cumulative counts -> ``(N,)`` sorted group ids."""
    counts = offset_to_counts(offset)
    return torch.arange(counts.numel(), device=offset.device, dtype=torch.int64).repeat_interleave(
        counts
    )


def batch_to_offset(batch: Tensor, num_groups: int | None = None) -> Tensor:
    """``(N,)`` sorted group ids -> ``(B,)`` cumulative counts.

    ``num_groups`` allows trailing (or interior) empty groups; without it
    ``B = batch.max() + 1``.
    """
    counts = torch.bincount(batch, minlength=num_groups or 0)
    if num_groups is not None and counts.numel() != num_groups:
        raise ValueError(f"batch has ids >= num_groups={num_groups}")
    return counts.cumsum(0)


def pack_padded(mask: Tensor, *tensors: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
    """Select the masked slots of ``(*G, K, ...)`` tensors.

    Returns ``(batch, packed)``: ``batch`` is the flattened group id of every
    kept point (sorted, since groups are visited row-major and slots in
    order) and ``packed[i]`` is ``tensors[i]`` restricted to the mask, ``(N, ...)``.
    """
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")
    group_shape = mask.shape[:-1]
    slots = mask.shape[-1]
    flat_mask = mask.reshape(-1, slots)
    num_groups = flat_mask.shape[0]
    group_ids = torch.arange(num_groups, device=mask.device, dtype=torch.int64)
    batch = group_ids.unsqueeze(1).expand(num_groups, slots)[flat_mask]
    packed = []
    for tensor in tensors:
        if tensor.shape[: mask.ndim] != mask.shape:
            raise ValueError(
                f"tensor of shape {tuple(tensor.shape)} does not start with mask shape "
                f"{tuple(mask.shape)}"
            )
        trailing = tensor.shape[len(group_shape) + 1 :]
        packed.append(tensor.reshape(num_groups, slots, *trailing)[flat_mask])
    return batch, tuple(packed)


def unpack_to_padded(
    packed: Tensor,
    batch: Tensor,
    *,
    num_groups: int | None = None,
    max_count: int | None = None,
    pad_value: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """``(N, ...)`` packed rows -> ``(G, K, ...)`` padded tensor and ``(G, K)`` mask.

    Within a group, rows keep their relative order. ``batch`` need not be
    sorted. Rows beyond ``max_count`` in a group are dropped.
    """
    if batch.ndim != 1 or batch.shape[0] != packed.shape[0]:
        raise ValueError(f"batch {tuple(batch.shape)} must be (N,) with N={packed.shape[0]}")
    n = packed.shape[0]
    if num_groups is None:
        num_groups = int(batch.max().item()) + 1 if n > 0 else 0
    counts = torch.bincount(batch, minlength=num_groups)
    if counts.numel() != num_groups:
        raise ValueError(f"batch has ids >= num_groups={num_groups}")
    if max_count is None:
        max_count = int(counts.max().item()) if num_groups > 0 else 0

    sort_index = torch.argsort(batch, stable=True)
    batch_sorted = batch[sort_index]
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
    within = torch.arange(n, device=batch.device) - starts[batch_sorted]
    valid = within < max_count

    rows = batch_sorted[valid]
    slots = within[valid]
    padded = packed.new_full((num_groups, max_count, *packed.shape[1:]), pad_value)
    padded[rows, slots] = packed[sort_index[valid]]
    mask = torch.zeros(num_groups, max_count, dtype=torch.bool, device=batch.device)
    mask[rows, slots] = True
    return padded, mask


def scatter_to_padded(packed: Tensor, mask: Tensor, pad_value: float = 0.0) -> Tensor:
    """Exact inverse of :func:`pack_padded`: write ``(N, ...)`` rows into the masked slots."""
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be bool, got {mask.dtype}")
    if int(mask.sum().item()) != packed.shape[0]:
        raise ValueError(f"mask selects {int(mask.sum())} slots but packed has {packed.shape[0]}")
    padded = packed.new_full((*mask.shape, *packed.shape[1:]), pad_value)
    padded[mask] = packed
    return padded
