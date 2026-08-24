"""Point-budget sampling: voxel pooling, FPS, and space-filling-curve striding."""

from __future__ import annotations

import torch
from torch import Tensor

from .serialization import SpaceFillingOrder, encode_grid


def voxel_pool(
    points: Tensor, features: Tensor | None, voxel_size: float
) -> tuple[Tensor, Tensor | None]:
    """Mean-pool points and optional features in cubic voxels."""
    if voxel_size <= 0 or points.shape[0] == 0:
        return points, features
    grid = torch.floor(points / voxel_size).to(torch.int64)
    _, parent = torch.unique(grid, dim=0, return_inverse=True)
    count = int(parent.max().item()) + 1
    counts = torch.bincount(parent, minlength=count).to(points.dtype).clamp_min(1)

    pooled_points = points.new_zeros(count, points.shape[-1])
    pooled_points.index_add_(0, parent, points)
    pooled_points /= counts[:, None]

    if features is None:
        return pooled_points, None
    pooled_features = features.new_zeros(count, *features.shape[1:])
    pooled_features.index_add_(0, parent, features)
    reshape = (count,) + (1,) * (features.ndim - 1)
    pooled_features /= counts.to(features.dtype).reshape(reshape)
    return pooled_points, pooled_features


def furthest_point_indices(points: Tensor, count: int) -> Tensor:
    """Deterministic pure-torch furthest-point sampling for one cloud."""
    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}")
    count = min(count, points.shape[0])
    indices = torch.empty(count, dtype=torch.long, device=points.device)
    if count == 0:
        return indices
    indices[0] = 0
    distances = torch.full(
        (points.shape[0],),
        float("inf"),
        dtype=points.dtype,
        device=points.device,
    )
    last = points[0]
    for index in range(1, count):
        distances = torch.minimum(distances, torch.sum((points - last).square(), dim=-1))
        indices[index] = torch.argmax(distances)
        last = points[indices[index]]
    return indices


def furthest_point_sample(
    points: Tensor, features: Tensor | None, maximum_points: int
) -> tuple[Tensor, Tensor | None]:
    if maximum_points <= 0 or points.shape[0] <= maximum_points:
        return points, features
    indices = furthest_point_indices(points, maximum_points)
    return points[indices], features[indices] if features is not None else None


def space_filling_stride_indices(
    points: Tensor,
    stride: int,
    grid_size: float,
    *,
    order: SpaceFillingOrder = "hilbert",
    depth: int = 16,
) -> Tensor:
    if stride <= 1:
        return torch.arange(points.shape[0], device=points.device)
    if points.shape[0] == 0:
        return torch.empty(0, dtype=torch.long, device=points.device)
    grid_size = max(grid_size, 0.004)
    grid = torch.floor(points / grid_size).to(torch.int64)
    grid -= grid.amin(dim=0)
    grid.clamp_(max=2**depth - 1)
    code = encode_grid(grid, depth=depth, order=order)
    return torch.argsort(code)[::stride]


def space_filling_stride(
    points: Tensor,
    features: Tensor | None,
    stride: int,
    grid_size: float,
    *,
    order: SpaceFillingOrder = "hilbert",
) -> tuple[Tensor, Tensor | None]:
    indices = space_filling_stride_indices(points, stride, grid_size, order=order)
    return points[indices], features[indices] if features is not None else None
