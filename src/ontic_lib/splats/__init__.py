"""3D Gaussian-splatting helpers: covariances and spherical-harmonic rotation.

SH rotation requires the ``e3nn`` extra (``pip install ontic-lib[e3nn]``).
"""

from .gaussians import covariance_from_scale_rotation, covariance_from_scale_rotation_representation
from .sh import (
    direction_to_angles,
    rotate_sh,
    rotate_spherical_harmonics,
    rotation_matrix_to_angles,
)

__all__ = [
    "covariance_from_scale_rotation",
    "covariance_from_scale_rotation_representation",
    "direction_to_angles",
    "rotate_sh",
    "rotate_spherical_harmonics",
    "rotation_matrix_to_angles",
]
