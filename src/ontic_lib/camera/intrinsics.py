"""Pinhole intrinsics in pixel-space and normalized representations."""

from __future__ import annotations

from torch import Tensor


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
