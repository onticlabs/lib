"""Voxel coordinates and cluster (segment) reductions in pure torch.

Points are packed rows ``(N, ...)`` with a group id ``batch: (N,) int64``; a
cluster id maps each row to one of ``M`` output rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

Reduce = Literal["sum", "mean", "max", "min", "any"]

_COORD_BITS = 16


def voxel_coords(
    coord: Tensor,
    grid_size: float,
    *,
    origin: Tensor | None = None,
    max_index: int = 2**15 - 1,
) -> Tensor:
    """``(N, 3)`` float points -> ``(N, 3)`` int32 voxel indices.

    ``trunc((coord - origin) / grid_size)`` clamped to ``[0, max_index]``.
    ``origin`` defaults to the per-axis minimum over finite entries;
    non-finite entries are pinned to the origin (index 0). ``coord`` itself
    is never modified.
    """
    if coord.ndim != 2 or coord.shape[-1] != 3:
        raise ValueError(f"expected (N, 3) coordinates, got {tuple(coord.shape)}")
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")
    if coord.shape[0] == 0:
        return torch.zeros(0, 3, dtype=torch.int32, device=coord.device)
    finite = torch.isfinite(coord)
    all_finite = bool(finite.all())
    if origin is not None:
        lo = origin.to(coord)
    elif all_finite:
        lo = coord.min(0).values
    else:
        safe = torch.where(finite, coord, torch.full_like(coord, float("inf")))
        lo = safe.min(0).values
        lo = torch.where(torch.isfinite(lo), lo, torch.zeros_like(lo))
    if not all_finite:
        coord = torch.where(finite, coord, lo.expand_as(coord))
    grid = torch.div(coord - lo, grid_size, rounding_mode="trunc").int()
    return grid.clamp_(0, max_index)


def cluster_reduce(values: Tensor, cluster: Tensor, num_clusters: int, reduce: Reduce) -> Tensor:
    """Reduce ``(N, ...)`` rows into ``(M, ...)`` by cluster id.

    ``"any"`` treats values as bool. Empty clusters yield 0 (False).
    """
    if cluster.ndim != 1 or cluster.shape[0] != values.shape[0]:
        raise ValueError(f"cluster {tuple(cluster.shape)} must be (N,) with N={values.shape[0]}")
    cluster = cluster.long()
    out_shape = (num_clusters, *values.shape[1:])
    if reduce == "any":
        out = torch.zeros(out_shape, dtype=torch.int64, device=values.device)
        index = cluster.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
        out.scatter_reduce_(0, index, values.long(), reduce="amax", include_self=True)
        return out.bool()
    if reduce in ("sum", "mean"):
        if reduce == "mean" and not values.is_floating_point():
            raise ValueError("mean reduction requires floating-point values")
        out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
        out.index_add_(0, cluster, values)
        if reduce == "mean":
            counts = torch.bincount(cluster, minlength=num_clusters).clamp_min(1)
            out = out / counts.to(out.dtype).view(-1, *([1] * (values.ndim - 1)))
        return out
    if reduce in ("max", "min"):
        out = torch.zeros(out_shape, dtype=values.dtype, device=values.device)
        index = cluster.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
        return out.scatter_reduce(
            0, index, values, reduce="amax" if reduce == "max" else "amin", include_self=False
        )
    raise ValueError(f"unknown reduce {reduce!r}")


@dataclass(frozen=True)
class Clusters:
    """Partition of ``N`` rows into ``M`` sorted clusters.

    ``cluster (N,)`` id per row; ``counts (M,)``; ``head (M,)`` lowest row index
    of each cluster; ``sorted_index (N,)`` rows stably sorted by cluster;
    ``ptr (M+1,)`` segment boundaries into ``sorted_index``.
    """

    cluster: Tensor
    counts: Tensor
    head: Tensor
    sorted_index: Tensor
    ptr: Tensor

    @property
    def num_clusters(self) -> int:
        return int(self.counts.numel())

    @classmethod
    def from_ids(cls, cluster: Tensor, counts: Tensor) -> Clusters:
        sorted_index = torch.argsort(cluster, stable=True)
        ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        head = sorted_index[ptr[:-1]]
        return cls(cluster=cluster, counts=counts, head=head, sorted_index=sorted_index, ptr=ptr)


def _pack_grid_key(grid_coord: Tensor, batch: Tensor) -> Tensor:
    """``(batch, x, y, z)`` -> one sortable int64 key (16 bits per axis, batch on top)."""
    grid = grid_coord.long()
    if grid.shape[0] > 0 and (grid.min() < 0 or grid.max() >= 1 << _COORD_BITS):
        raise ValueError(f"grid coordinates must lie in [0, {1 << _COORD_BITS})")
    if batch.shape[0] > 0 and batch.max() >= 1 << 15:
        raise ValueError("at most 2**15 groups are supported")
    key = batch.long() << (3 * _COORD_BITS)
    key = key | grid[:, 0] << (2 * _COORD_BITS) | grid[:, 1] << _COORD_BITS | grid[:, 2]
    return key


def grid_clusters(grid_coord: Tensor, batch: Tensor, *, stride: int = 1) -> tuple[Tensor, Clusters]:
    """Cluster rows sharing ``(batch, grid_coord // stride)``.

    Returns the ``(M, 3)`` pooled grid coordinates (same dtype as the input)
    and the clusters, sorted by ``(batch, x, y, z)``.
    """
    if grid_coord.ndim != 2 or grid_coord.shape[-1] != 3:
        raise ValueError(f"expected (N, 3) grid coordinates, got {tuple(grid_coord.shape)}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    coarse = torch.div(grid_coord.long(), stride, rounding_mode="trunc")
    key = _pack_grid_key(coarse, batch)
    unique, cluster, counts = torch.unique(
        key, sorted=True, return_inverse=True, return_counts=True
    )
    mask = (1 << _COORD_BITS) - 1
    pooled = torch.stack(
        [unique >> (2 * _COORD_BITS) & mask, unique >> _COORD_BITS & mask, unique & mask], dim=-1
    )
    return pooled.to(grid_coord.dtype), Clusters.from_ids(cluster, counts)


def code_clusters(code: Tensor) -> Clusters:
    """Cluster rows sharing a serialization code (sorted by code)."""
    if code.ndim != 1:
        raise ValueError(f"expected (N,) codes, got {tuple(code.shape)}")
    _, cluster, counts = torch.unique(code, sorted=True, return_inverse=True, return_counts=True)
    return Clusters.from_ids(cluster, counts)
