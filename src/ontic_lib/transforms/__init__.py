"""SO(3) rotation representations and rigid/similarity transforms.

See the package-level conventions in :mod:`ontic_lib`.
"""

from .rigid import (
    apply_matrix,
    apply_transform,
    homogenize_points,
    homogenize_vectors,
    invert_rigid_transform,
    transform_camera_to_world_se3,
    transform_camera_to_world_sim3,
    transform_points,
    transform_vectors,
)
from .rotations import (
    QuaternionOrder,
    accumulate_rotation_vectors,
    increment_rotation,
    matrix_to_procrustes,
    matrix_to_quaternion,
    matrix_to_quaternion_xyzw,
    matrix_to_rotation_6d,
    matrix_to_rotation_representation,
    normalize_rotation_representation,
    procrustes_to_matrix,
    quaternion_to_matrix,
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_matrix,
    quaternion_xyzw_to_wxyz,
    rotation_6d_to_matrix,
    rotation_matrix_times_representation,
    rotation_representation_to_matrix,
)

__all__ = [
    "QuaternionOrder",
    "accumulate_rotation_vectors",
    "apply_matrix",
    "apply_transform",
    "homogenize_points",
    "homogenize_vectors",
    "increment_rotation",
    "invert_rigid_transform",
    "matrix_to_procrustes",
    "matrix_to_quaternion",
    "matrix_to_quaternion_xyzw",
    "matrix_to_rotation_6d",
    "matrix_to_rotation_representation",
    "normalize_rotation_representation",
    "procrustes_to_matrix",
    "quaternion_to_matrix",
    "quaternion_wxyz_to_xyzw",
    "quaternion_xyzw_to_matrix",
    "quaternion_xyzw_to_wxyz",
    "rotation_6d_to_matrix",
    "rotation_matrix_times_representation",
    "rotation_representation_to_matrix",
    "transform_camera_to_world_se3",
    "transform_camera_to_world_sim3",
    "transform_points",
    "transform_vectors",
]
