"""World-space camera rays and pixel footprints."""

from __future__ import annotations

import torch
from torch import Tensor

from ..transforms.rigid import apply_matrix, transform_vectors
from .projection import unproject_camera_points


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


def world_pixel_size(intrinsics_normalized: Tensor, image_size: tuple[int, int]) -> Tensor:
    height, width = image_size
    pixel_size = apply_matrix(
        torch.linalg.inv(intrinsics_normalized[..., :2, :2]),
        intrinsics_normalized.new_tensor((1 / width, 1 / height)),
    )
    return pixel_size.sum(dim=-1)
