"""3D Gaussian-splatting helpers: covariances, SH rotation, rasterization.

SH rotation requires the ``e3nn`` extra; rendering requires the ``gsplat``
extra (both lazy — importing this package needs neither).
"""

from .gaussians import covariance_from_scale_rotation, covariance_from_scale_rotation_representation
from .rendering import RenderOutput, render_gaussians
from .sh import (
    direction_to_angles,
    rotate_sh,
    rotate_spherical_harmonics,
    rotation_matrix_to_angles,
)

__all__ = [
    "RenderOutput",
    "covariance_from_scale_rotation",
    "covariance_from_scale_rotation_representation",
    "direction_to_angles",
    "render_gaussians",
    "rotate_sh",
    "rotate_spherical_harmonics",
    "rotation_matrix_to_angles",
]
