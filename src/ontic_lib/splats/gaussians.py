"""Geometry operations for anisotropic 3D Gaussians."""

from __future__ import annotations

from torch import Tensor

from ..transforms.rotations import QuaternionOrder, rotation_representation_to_matrix


def covariance_from_scale_rotation(scales: Tensor, rotations: Tensor) -> Tensor:
    """Return ``R diag(scales**2) R.T`` for ``(..., 3)`` scales and matrices."""
    scale_covariance = scales.square().diag_embed()
    return rotations @ scale_covariance @ rotations.transpose(-1, -2)


def covariance_from_scale_rotation_representation(
    scales: Tensor,
    rotation_representation: Tensor,
    *,
    quaternion_order: QuaternionOrder = "wxyz",
    normalize: bool = True,
    eps: float = 1e-08,
) -> Tensor:
    rotations = rotation_representation_to_matrix(
        rotation_representation,
        quaternion_order=quaternion_order,
        normalize=normalize,
        eps=eps,
    )
    return covariance_from_scale_rotation(scales, rotations)
