"""Differentiable SO(3) representation conversions."""

from __future__ import annotations

from typing import Literal

import roma
import torch
from torch import Tensor

QuaternionOrder = Literal["wxyz", "xyzw"]


def _normalize_quaternion(quaternion: Tensor, eps: float) -> Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(eps)


def quaternion_xyzw_to_wxyz(quaternion: Tensor) -> Tensor:
    return torch.cat((quaternion[..., -1:], quaternion[..., :3]), dim=-1)


def quaternion_wxyz_to_xyzw(quaternion: Tensor) -> Tensor:
    return torch.cat((quaternion[..., 1:], quaternion[..., :1]), dim=-1)


def quaternion_xyzw_to_matrix(
    quaternion: Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-08,
) -> Tensor:
    if normalize:
        quaternion = _normalize_quaternion(quaternion, eps)
    return roma.unitquat_to_rotmat(quaternion)


def quaternion_to_matrix(
    quaternion: Tensor,
    *,
    normalize: bool = True,
    eps: float = 1e-08,
) -> Tensor:
    """Convert canonical ``wxyz`` quaternions to rotation matrices."""
    return quaternion_xyzw_to_matrix(
        quaternion_wxyz_to_xyzw(quaternion), normalize=normalize, eps=eps
    )


def matrix_to_quaternion_xyzw(rotation: Tensor) -> Tensor:
    return roma.rotmat_to_unitquat(rotation)


def matrix_to_quaternion(rotation: Tensor) -> Tensor:
    """Convert rotation matrices to canonical ``wxyz`` quaternions."""
    return quaternion_xyzw_to_wxyz(matrix_to_quaternion_xyzw(rotation))


def rotation_6d_to_matrix(rotation_6d: Tensor, *, eps: float = 1e-07) -> Tensor:
    if rotation_6d.shape[-1] == 6:
        rotation_6d = rotation_6d.reshape(*rotation_6d.shape[:-1], 3, 2)
    if rotation_6d.shape[-2:] != (3, 2):
        raise ValueError(f"expected (..., 6) or (..., 3, 2), got {rotation_6d.shape}")
    return roma.special_gramschmidt(rotation_6d, epsilon=eps)


def matrix_to_rotation_6d(rotation: Tensor) -> Tensor:
    return rotation[..., :, :2].reshape(*rotation.shape[:-2], 6)


def procrustes_to_matrix(procrustes: Tensor) -> Tensor:
    if procrustes.shape[-1] == 9:
        procrustes = procrustes.reshape(*procrustes.shape[:-1], 3, 3)
    if procrustes.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 9) or (..., 3, 3), got {procrustes.shape}")
    return roma.special_procrustes(procrustes)


def matrix_to_procrustes(rotation: Tensor) -> Tensor:
    return rotation.reshape(*rotation.shape[:-2], 9)


def rotation_representation_to_matrix(
    representation: Tensor,
    *,
    quaternion_order: QuaternionOrder = "wxyz",
    normalize: bool = True,
    eps: float = 1e-08,
) -> Tensor:
    dimension = representation.shape[-1]
    if dimension == 4:
        if quaternion_order == "wxyz":
            return quaternion_to_matrix(representation, normalize=normalize, eps=eps)
        return quaternion_xyzw_to_matrix(representation, normalize=normalize, eps=eps)
    if dimension == 6:
        return rotation_6d_to_matrix(representation, eps=eps)
    if dimension == 9:
        return procrustes_to_matrix(representation)
    raise ValueError(f"expected a 4D, 6D, or 9D rotation representation, got {dimension}")


def matrix_to_rotation_representation(
    rotation: Tensor,
    dimension: int,
    *,
    quaternion_order: QuaternionOrder = "wxyz",
) -> Tensor:
    if dimension == 4:
        if quaternion_order == "wxyz":
            return matrix_to_quaternion(rotation)
        return matrix_to_quaternion_xyzw(rotation)
    if dimension == 6:
        return matrix_to_rotation_6d(rotation)
    if dimension == 9:
        return matrix_to_procrustes(rotation)
    raise ValueError(f"expected representation dimension 4, 6, or 9, got {dimension}")


def normalize_rotation_representation(
    representation: Tensor,
    *,
    quaternion_order: QuaternionOrder = "wxyz",
    eps: float = 1e-08,
) -> Tensor:
    if representation.shape[-1] == 4:
        return _normalize_quaternion(representation, eps)
    rotation = rotation_representation_to_matrix(
        representation, quaternion_order=quaternion_order, eps=eps
    )
    return matrix_to_rotation_representation(
        rotation, representation.shape[-1], quaternion_order=quaternion_order
    )


def increment_rotation(
    representation: Tensor,
    delta: Tensor,
    *,
    quaternion_order: QuaternionOrder = "wxyz",
    eps: float = 1e-07,
) -> Tensor:
    if delta.shape[-1] == 3:
        delta_rotation = roma.rotvec_to_rotmat(delta, epsilon=eps)
    else:
        delta_rotation = rotation_representation_to_matrix(
            delta, quaternion_order=quaternion_order, eps=eps
        )
    current = rotation_representation_to_matrix(
        representation, quaternion_order=quaternion_order, eps=eps
    )
    return matrix_to_rotation_representation(
        delta_rotation @ current, representation.shape[-1], quaternion_order=quaternion_order
    )


def accumulate_rotation_vectors(deltas: Tensor, *, eps: float = 1e-07) -> Tensor:
    """Left-compose time-ordered rotation-vector increments as matrices."""
    rotations = roma.rotvec_to_rotmat(deltas, epsilon=eps)
    cumulative = [rotations[:, 0]]
    for index in range(1, rotations.shape[1]):
        cumulative.append(rotations[:, index] @ cumulative[-1])
    return torch.stack(cumulative, dim=1)


def rotation_matrix_times_representation(
    rotation: Tensor,
    representation: Tensor,
    *,
    quaternion_order: QuaternionOrder = "wxyz",
) -> Tensor:
    composed = rotation @ rotation_representation_to_matrix(
        representation, quaternion_order=quaternion_order
    )
    return matrix_to_rotation_representation(
        composed, representation.shape[-1], quaternion_order=quaternion_order
    )
