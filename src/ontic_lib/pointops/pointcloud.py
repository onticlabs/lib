"""Point-cloud container, construction from depth views, and spatial filtering."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..depth.lifting import DepthType, depth_to_world_points


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
