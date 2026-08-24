"""Lift depth maps through pinhole cameras into world-space points."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ..geometry.cameras import sample_image_grid, world_rays

DepthType = Literal["z", "ray"]


def _broadcast_image_grid(grid: Tensor, depths: Tensor, image_size: tuple[int, int]) -> Tensor:
    height, width = image_size
    if depths.shape[-3:-1] == (height, width):
        leading = depths.ndim - 3
        return grid.reshape(*([1] * leading), height, width, 2)
    flattened_size = height * width
    matching_dimensions = [
        index for index, size in enumerate(depths.shape) if size == flattened_size
    ]
    if not matching_dimensions:
        raise ValueError(
            f"depth shape {tuple(depths.shape)} contains neither ({height}, {width}) "
            f"nor flattened image dimension {flattened_size}"
        )
    flattened_dimension = matching_dimensions[0]
    trailing = depths.ndim - flattened_dimension - 2
    return grid.reshape(*([1] * flattened_dimension), flattened_size, *([1] * trailing), 2)


def depth_to_world_points(
    depths: Tensor,
    camera_to_world: Tensor,
    intrinsics_normalized: Tensor,
    image_size: tuple[int, int],
    *,
    offsets_xy: Tensor | None = None,
    depth_type: DepthType = "z",
    return_coordinates: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Lift depth samples to world points.

    ``depths`` follows the existing ``(..., H, W, 1)`` convention and may also
    contain a flattened ``H*W`` dimension. ``depth_type='z'`` means camera-z
    depth; ``'ray'`` means Euclidean distance from the camera center.
    """
    if depth_type not in ("z", "ray"):
        raise ValueError(f"depth_type must be 'z' or 'ray', got {depth_type!r}")
    height, width = image_size
    coordinates, _ = sample_image_grid(image_size, device=depths.device, dtype=depths.dtype)
    coordinates = _broadcast_image_grid(coordinates, depths, image_size)
    if offsets_xy is not None:
        pixel_size = depths.new_tensor((1 / width, 1 / height))
        coordinates = coordinates + (offsets_xy - 0.5) * pixel_size
    else:
        coordinates = coordinates + (depths - 0.5) * 0.0
    camera_to_world_broadcast = camera_to_world
    intrinsics_broadcast = intrinsics_normalized
    while camera_to_world_broadcast.ndim - 1 < depths.ndim:
        camera_to_world_broadcast = camera_to_world_broadcast.unsqueeze(-3)
        intrinsics_broadcast = intrinsics_broadcast.unsqueeze(-3)
    origins, directions = world_rays(coordinates, camera_to_world_broadcast, intrinsics_broadcast)
    if depth_type == "z":
        camera_z_world = camera_to_world_broadcast[..., :3, 2]
        cosine = torch.sum(directions * camera_z_world, dim=-1)
        directions = directions / cosine.unsqueeze(-1).clamp_min(1e-6)
    points = origins + directions * depths
    if return_coordinates:
        return points, coordinates
    return points
