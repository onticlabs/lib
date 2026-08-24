"""Rotation of real spherical-harmonic coefficients using e3nn."""

from __future__ import annotations

from math import isqrt

import torch
from torch import Tensor


def _safe_acos(value: Tensor, epsilon: float = 1e-06) -> Tensor:
    """Linear continuation near +/-1 with finite gradients and exact endpoints."""
    sign = torch.sign(value)
    boundary = value.new_tensor(1 - epsilon)
    slope = torch.acos(boundary) / epsilon
    return torch.where(
        value.abs() <= boundary,
        torch.acos(value),
        torch.acos(sign * boundary) - slope * sign * (value.abs() - 1 + epsilon),
    )


def direction_to_angles(direction: Tensor) -> tuple[Tensor, Tensor]:
    direction = torch.nn.functional.normalize(direction, p=2, dim=-1)
    beta = _safe_acos(direction[..., 1])
    alpha = torch.atan2(direction[..., 0], direction[..., 2])
    return alpha, beta


def rotation_matrix_to_angles(rotation: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    from e3nn.o3 import angles_to_matrix

    direction = rotation @ rotation.new_tensor([0.0, 1.0, 0.0])
    alpha, beta = direction_to_angles(direction)
    residual = angles_to_matrix(alpha, beta, torch.zeros_like(alpha)).transpose(-1, -2) @ rotation
    gamma = torch.atan2(residual[..., 0, 2], residual[..., 0, 0])
    return alpha, beta, gamma


def rotate_spherical_harmonics(coefficients: Tensor, rotations: Tensor) -> Tensor:
    """Rotate complete real-SH bands stored along the last dimension."""
    from e3nn.o3 import wigner_D

    coefficient_count = coefficients.shape[-1]
    band_count = isqrt(coefficient_count)
    if band_count**2 != coefficient_count:
        raise ValueError(
            "SH coefficient count must contain complete bands: "
            f"expected (degree + 1)^2, got {coefficient_count}"
        )
    alpha, beta, gamma = rotation_matrix_to_angles(rotations)
    output = []
    for degree in range(band_count):
        sh_rotation = wigner_D(degree, alpha, beta, gamma).to(
            device=coefficients.device, dtype=coefficients.dtype
        )
        band = coefficients[..., degree**2 : (degree + 1) ** 2]
        output.append(torch.einsum("...ij,...j->...i", sh_rotation, band))
    return torch.cat(output, dim=-1)


rotate_sh = rotate_spherical_harmonics
