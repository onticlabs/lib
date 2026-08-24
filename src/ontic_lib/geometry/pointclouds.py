"""Framework-neutral point-cloud construction, filtering, and sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..depth.lifting import DepthType, depth_to_world_points
from .space_filling import SpaceFillingOrder, encode_grid


@dataclass
class PointCloud:
    points: Tensor
    colors: Tensor | None = None


def pointcloud_from_depth_views(
    depth: Tensor,
    camera_to_world: Tensor,
    intrinsics_normalized: Tensor,
    rgb: Tensor | None = None,
    confidence: Tensor | None = None,
    *,
    stride: int = 1,
    confidence_threshold: float = 0.0,
    depth_type: DepthType = "z",
    minimum_depth: float | None = None,
) -> PointCloud:
    """Unproject ``(V, H, W)`` depth maps into one world-space cloud."""
    views, height, width = depth.shape
    points = depth_to_world_points(
        depth.unsqueeze(-1),
        camera_to_world,
        intrinsics_normalized,
        (height, width),
        depth_type=depth_type,
    )

    colors = None
    if rgb is not None:
        colors = F.interpolate(
            rgb,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        colors = colors.permute(0, 2, 3, 1).clamp(0.0, 1.0)

    if confidence is None:
        valid = torch.ones((views, height, width), dtype=torch.bool, device=depth.device)
    else:
        valid = confidence > confidence_threshold
    valid &= torch.isfinite(depth)
    if minimum_depth is not None:
        valid &= depth > minimum_depth

    if stride > 1:
        points = points[:, ::stride, ::stride]
        valid = valid[:, ::stride, ::stride]
        if colors is not None:
            colors = colors[:, ::stride, ::stride]

    points = points.reshape(-1, 3)
    valid = valid.reshape(-1)
    if colors is not None:
        colors = colors.reshape(-1, 3)[valid]
    return PointCloud(points=points[valid], colors=colors)


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


def aabb_mask(points: Tensor, minimum: Tensor, maximum: Tensor) -> Tensor:
    """Return ``(..., 1)`` mask for points inside an inclusive AABB."""
    return (points >= minimum).all(dim=-1, keepdim=True) & (points <= maximum).all(
        dim=-1, keepdim=True
    )


def crop_to_aabb(
    points: Tensor, features: Tensor | None, minimum, maximum
) -> tuple[Tensor, Tensor | None]:
    if minimum is None or maximum is None or points.shape[0] == 0:
        return points, features
    minimum = torch.as_tensor(minimum, dtype=points.dtype, device=points.device)
    maximum = torch.as_tensor(maximum, dtype=points.dtype, device=points.device)
    mask = aabb_mask(points, minimum, maximum).squeeze(-1)
    return points[mask], features[mask] if features is not None else None


def nearest_point_to_ray(
    points: Tensor, ray_origin, ray_direction, *, maximum_distance: float | None = None
) -> Tensor | None:
    if points.shape[0] == 0:
        return None
    origin = torch.as_tensor(ray_origin, dtype=points.dtype, device=points.device)
    direction = torch.as_tensor(ray_direction, dtype=points.dtype, device=points.device)
    direction = direction / direction.norm().clamp_min(1e-9)
    relative = points - origin
    distance_along_ray = relative @ direction
    perpendicular = (relative - distance_along_ray.unsqueeze(-1) * direction).norm(dim=-1)
    perpendicular = torch.where(
        distance_along_ray > 0,
        perpendicular,
        torch.full_like(perpendicular, float("inf")),
    )
    index = int(torch.argmin(perpendicular))
    if not torch.isfinite(perpendicular[index]):
        return None
    if maximum_distance is not None and perpendicular[index] > maximum_distance:
        return None
    return points[index]
