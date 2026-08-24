"""Pinhole projection and unprojection between image and camera/world space."""

from __future__ import annotations

import torch
from torch import Tensor

from ..transforms.rigid import apply_matrix, invert_rigid_transform, transform_points


def project_camera_points(
    points_camera: Tensor,
    intrinsics: Tensor,
    *,
    epsilon: float = torch.finfo(torch.float32).eps,
    infinity: float = 1e8,
) -> Tensor:
    """Project camera-space xyz using intrinsics in matching output units."""
    normalized = points_camera / (points_camera[..., -1:] + epsilon)
    normalized = normalized.nan_to_num(posinf=infinity, neginf=-infinity)
    projected = apply_matrix(intrinsics, normalized)
    return projected[..., :-1]


def project_world_points(
    points_world: Tensor,
    camera_to_world: Tensor,
    intrinsics: Tensor,
    *,
    epsilon: float = torch.finfo(torch.float32).eps,
) -> tuple[Tensor, Tensor]:
    world_to_camera = invert_rigid_transform(camera_to_world)
    points_camera = transform_points(world_to_camera, points_world)
    in_front = points_camera[..., 2] >= 0
    return project_camera_points(points_camera, intrinsics, epsilon=epsilon), in_front


def unproject_camera_points(coordinates: Tensor, z_depth: Tensor, intrinsics: Tensor) -> Tensor:
    """Unproject image coordinates at camera-z depth into camera-space xyz."""
    homogeneous = torch.cat((coordinates, torch.ones_like(coordinates[..., :1])), dim=-1)
    directions = apply_matrix(torch.linalg.inv(intrinsics), homogeneous)
    return directions * z_depth[..., None]


def sample_image_grid(
    image_size: tuple[int, int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Return normalized xy pixel centers and integer ij indices."""
    height, width = image_size
    rows = torch.arange(height, device=device)
    columns = torch.arange(width, device=device)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    indices = torch.stack((row_grid, column_grid), dim=-1)
    coordinates = torch.stack(
        (
            (column_grid.to(dtype) + 0.5) / width,
            (row_grid.to(dtype) + 0.5) / height,
        ),
        dim=-1,
    )
    return coordinates, indices
