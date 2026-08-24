"""Pinhole camera intrinsics, projection, unprojection, and world rays."""

from __future__ import annotations

import torch
from torch import Tensor

from .transforms import (
    apply_matrix,
    invert_rigid_transform,
    transform_points,
    transform_vectors,
)


def normalize_intrinsics(intrinsics_pixels: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Convert pixel-space intrinsics to normalized ``[0, 1]`` image coordinates."""
    height, width = image_size
    intrinsics = intrinsics_pixels.clone()
    intrinsics[..., 0, 0] /= width
    intrinsics[..., 0, 2] /= width
    intrinsics[..., 1, 1] /= height
    intrinsics[..., 1, 2] /= height
    return intrinsics


def denormalize_intrinsics(intrinsics_normalized: Tensor, image_size: tuple[int, int]) -> Tensor:
    """Convert normalized ``[0, 1]`` intrinsics to pixel coordinates."""
    height, width = image_size
    intrinsics = intrinsics_normalized.clone()
    intrinsics[..., 0, 0] *= width
    intrinsics[..., 0, 2] *= width
    intrinsics[..., 1, 1] *= height
    intrinsics[..., 1, 2] *= height
    return intrinsics


def resize_intrinsics(
    intrinsics_pixels: Tensor,
    original_size: tuple[int, int],
    new_size: tuple[int, int],
) -> Tensor:
    """Update pixel-space intrinsics after resizing an image."""
    old_height, old_width = original_size
    new_height, new_width = new_size
    intrinsics = intrinsics_pixels.clone()
    intrinsics[..., 0, 0] *= new_width / old_width
    intrinsics[..., 0, 2] *= new_width / old_width
    intrinsics[..., 1, 1] *= new_height / old_height
    intrinsics[..., 1, 2] *= new_height / old_height
    return intrinsics


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


def world_rays(
    coordinates: Tensor,
    camera_to_world: Tensor,
    intrinsics: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return world-space ray origins and unit directions for image coordinates."""
    directions_camera = unproject_camera_points(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions_camera = directions_camera / directions_camera.norm(dim=-1, keepdim=True)
    directions_world = transform_vectors(camera_to_world, directions_camera)
    origins = camera_to_world[..., :3, 3].broadcast_to(directions_world.shape)
    return origins, directions_world


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


def world_pixel_size(intrinsics_normalized: Tensor, image_size: tuple[int, int]) -> Tensor:
    height, width = image_size
    pixel_size = apply_matrix(
        torch.linalg.inv(intrinsics_normalized[..., :2, :2]),
        intrinsics_normalized.new_tensor((1 / width, 1 / height)),
    )
    return pixel_size.sum(dim=-1)
