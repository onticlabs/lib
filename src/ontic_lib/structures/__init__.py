"""Data containers: point clouds, packed point batches, 3D Gaussians.

See the package-level conventions in :mod:`ontic_lib`.
"""

from .gaussians import Gaussians
from .pointcloud import (
    PointCloud,
    aabb_mask,
    crop_to_aabb,
    nearest_point_to_ray,
    padded_aabb,
    pointcloud_from_depth_views,
)

from .pointbatch import PointBatch

__all__ = [
    "Gaussians",
    "PointBatch",
    "PointCloud",
    "aabb_mask",
    "crop_to_aabb",
    "nearest_point_to_ray",
    "padded_aabb",
    "pointcloud_from_depth_views",
]
