"""Batched tensor ops: point sampling, serialization, alignment, point clouds.

See the package-level conventions in :mod:`ontic_lib`.
"""

from .alignment import (
    align_camera_poses_se3,
    align_camera_poses_sim3,
    align_cameras_sim3,
    align_points_sim3,
    anchor_transform,
    clamp_scale,
)
from .pointcloud import (
    PointCloud,
    aabb_mask,
    crop_to_aabb,
    nearest_point_to_ray,
    pointcloud_from_depth_views,
)
from .sampling import (
    furthest_point_indices,
    furthest_point_sample,
    space_filling_stride,
    space_filling_stride_indices,
    voxel_pool,
)
from .serialization import (
    SpaceFillingOrder,
    encode_grid,
    hilbert_decode,
    hilbert_encode,
    morton_decode,
    morton_encode,
)

__all__ = [
    "PointCloud",
    "SpaceFillingOrder",
    "aabb_mask",
    "align_camera_poses_se3",
    "align_camera_poses_sim3",
    "align_cameras_sim3",
    "align_points_sim3",
    "anchor_transform",
    "clamp_scale",
    "crop_to_aabb",
    "encode_grid",
    "furthest_point_indices",
    "furthest_point_sample",
    "hilbert_decode",
    "hilbert_encode",
    "morton_decode",
    "morton_encode",
    "nearest_point_to_ray",
    "pointcloud_from_depth_views",
    "space_filling_stride",
    "space_filling_stride_indices",
    "voxel_pool",
]
