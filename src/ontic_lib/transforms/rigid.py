"""Rigid and similarity-transform operations."""

from __future__ import annotations

import torch
from torch import Tensor


def homogenize_points(points: Tensor) -> Tensor:
    return torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)


def homogenize_vectors(vectors: Tensor) -> Tensor:
    return torch.cat((vectors, torch.zeros_like(vectors[..., :1])), dim=-1)


def apply_matrix(matrix: Tensor, vectors: Tensor) -> Tensor:
    """Apply matrices to column vectors with broadcastable leading dimensions."""
    while matrix.ndim < vectors.ndim + 1:
        matrix = matrix.unsqueeze(-3)
    return torch.matmul(matrix, vectors.unsqueeze(-1)).squeeze(-1)


def apply_transform(transform: Tensor, homogeneous_coordinates: Tensor) -> Tensor:
    return apply_matrix(transform, homogeneous_coordinates)


def transform_points(transform: Tensor, points: Tensor) -> Tensor:
    return apply_transform(transform, homogenize_points(points))[..., :-1]


def transform_vectors(transform: Tensor, vectors: Tensor) -> Tensor:
    return apply_transform(transform, homogenize_vectors(vectors))[..., :-1]


def invert_rigid_transform(transform: Tensor) -> Tensor:
    """Invert ``(..., 4, 4)`` rigid transforms using ``R.T`` and ``-R.T @ t``."""
    rotation = transform[..., :3, :3]
    translation = transform[..., :3, 3:]
    rotation_inverse = rotation.transpose(-1, -2)
    inverse = torch.zeros_like(transform)
    inverse[..., :3, :3] = rotation_inverse
    inverse[..., :3, 3:] = -rotation_inverse @ translation
    inverse[..., 3, 3] = 1
    return inverse


def transform_camera_to_world_se3(camera_to_world: Tensor, transform: Tensor) -> Tensor:
    """Apply a world-frame SE(3) transform to camera-to-world poses."""
    return transform.to(camera_to_world) @ camera_to_world


def transform_camera_to_world_sim3(
    camera_to_world: Tensor,
    rotation: Tensor,
    translation: Tensor,
    scale: Tensor,
) -> Tensor:
    """Apply ``x' = scale * rotation @ x + translation`` to camera poses."""
    rotation = rotation.to(camera_to_world)
    translation = translation.to(camera_to_world)
    scale = scale.to(camera_to_world)
    output = camera_to_world.clone()
    if rotation.ndim == camera_to_world.ndim - 1:
        rotation_for_cameras = rotation.unsqueeze(-3)
    else:
        rotation_for_cameras = rotation
    output[..., :3, :3] = rotation_for_cameras @ camera_to_world[..., :3, :3]
    centers = apply_matrix(rotation, camera_to_world[..., :3, 3])
    while scale.ndim < centers.ndim:
        scale = scale.unsqueeze(-1)
    while translation.ndim < centers.ndim:
        translation = translation.unsqueeze(-2)
    output[..., :3, 3] = scale * centers + translation
    return output
