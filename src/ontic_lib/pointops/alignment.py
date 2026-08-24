"""Rigid and similarity alignment of cameras and point sets."""

from __future__ import annotations

import torch
from torch import Tensor

from ..transforms.rigid import invert_rigid_transform, transform_camera_to_world_sim3


def clamp_scale(scale: Tensor, minimum: float = 0.01, maximum: float = 100.0) -> Tensor:
    """Replace non-finite similarity scales and clamp extreme values."""
    scale = torch.nan_to_num(scale, nan=1.0, posinf=maximum, neginf=minimum)
    return scale.clamp(minimum, maximum)


def align_camera_poses_sim3(
    source_camera_to_world: Tensor,
    target_camera_to_world: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit batched Sim(3) transforms from source cameras to target cameras.

    Inputs have shape ``(B, V, 4, 4)``. Camera orientations determine rotation;
    camera centers determine scale and translation. Scale defaults to one when
    fewer than two distinct centers make it underdetermined.
    """
    with torch.no_grad():
        source = source_camera_to_world.to(torch.float64)
        target = target_camera_to_world.to(torch.float64)
        source_rotation = source[..., :3, :3]
        source_centers = source[..., :3, 3]
        target_rotation = target[..., :3, :3]
        target_centers = target[..., :3, 3]

        relative_sum = torch.einsum("bnij,bnkj->bik", target_rotation, source_rotation)

        u, _, vt = torch.linalg.svd(relative_sum)
        correction = (
            torch.eye(3, dtype=relative_sum.dtype, device=relative_sum.device)
            .expand_as(relative_sum)
            .clone()
        )
        correction[:, 2, 2] = torch.det(u @ vt).sign()
        rotation = u @ correction @ vt

        rotated_centers = torch.einsum("bij,bnj->bni", rotation, source_centers)
        source_mean = rotated_centers.mean(dim=1)
        target_mean = target_centers.mean(dim=1)
        source_zero_mean = rotated_centers - source_mean[:, None]
        target_zero_mean = target_centers - target_mean[:, None]
        denominator = source_zero_mean.square().sum(dim=(1, 2))
        scale = torch.where(
            denominator > 1e-12,
            (source_zero_mean * target_zero_mean).sum(dim=(1, 2)) / denominator.clamp_min(1e-12),
            torch.ones_like(denominator),
        )
        translation = target_mean - scale[:, None] * source_mean

    dtype = source_camera_to_world.dtype
    return rotation.to(dtype), translation.to(dtype), scale.to(dtype)


def align_camera_poses_se3(
    source_camera_to_world: Tensor,
    target_camera_to_world: Tensor,
) -> tuple[Tensor, Tensor]:
    """Fit batched rigid transforms from source cameras to target cameras."""
    with torch.no_grad():
        source = source_camera_to_world.to(torch.float64)
        target = target_camera_to_world.to(torch.float64)
        source_rotation = source[..., :3, :3]
        source_centers = source[..., :3, 3]
        target_rotation = target[..., :3, :3]
        target_centers = target[..., :3, 3]

        relative_sum = torch.einsum("bnij,bnkj->bik", target_rotation, source_rotation)

        u, _, vt = torch.linalg.svd(relative_sum)
        correction = (
            torch.eye(3, dtype=relative_sum.dtype, device=relative_sum.device)
            .expand_as(relative_sum)
            .clone()
        )
        correction[:, 2, 2] = torch.det(u @ vt).sign()
        rotation = u @ correction @ vt

        rotated_centers = torch.einsum("bij,bnj->bni", rotation, source_centers)
        translation = target_centers.mean(dim=1) - rotated_centers.mean(dim=1)

    dtype = source_camera_to_world.dtype
    return rotation.to(dtype), translation.to(dtype)


def align_points_sim3(
    points: Tensor,
    rotation: Tensor,
    translation: Tensor,
    scale: Tensor,
) -> Tensor:
    rotated = torch.matmul(points, rotation.transpose(-1, -2))
    while scale.ndim < rotated.ndim:
        scale = scale.unsqueeze(-1)
    while translation.ndim < rotated.ndim:
        translation = translation.unsqueeze(-2)
    return scale * rotated + translation


def align_cameras_sim3(
    camera_to_world: Tensor,
    rotation: Tensor,
    translation: Tensor,
    scale: Tensor,
) -> Tensor:
    return transform_camera_to_world_sim3(camera_to_world, rotation, translation, scale)


def anchor_transform(source_camera: Tensor, target_camera: Tensor) -> Tensor:
    """Transform that makes one source camera coincide with one target camera."""
    return target_camera @ invert_rigid_transform(source_camera)
