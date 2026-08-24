"""Pinhole camera intrinsics, projection/unprojection, and world rays.

Standalone (not tied to any renderer). See the package-level conventions in
:mod:`ontic_lib`.
"""

from .intrinsics import denormalize_intrinsics, normalize_intrinsics, resize_intrinsics
from .projection import (
    project_camera_points,
    project_world_points,
    sample_image_grid,
    unproject_camera_points,
)
from .rays import world_pixel_size, world_rays

__all__ = [
    "denormalize_intrinsics",
    "normalize_intrinsics",
    "project_camera_points",
    "project_world_points",
    "resize_intrinsics",
    "sample_image_grid",
    "unproject_camera_points",
    "world_pixel_size",
    "world_rays",
]
