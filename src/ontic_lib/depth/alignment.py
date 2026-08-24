"""Scale and affine alignment for predicted depth."""

from __future__ import annotations

import torch
from torch import Tensor

from ..geometry.alignment import align_camera_poses_sim3, clamp_scale


def fit_depth_scale(
    predicted: Tensor,
    target: Tensor,
    *,
    mask: Tensor | None = None,
    eps: float = 1e-6,
) -> Tensor:
    """Robust median scale so ``scale * predicted`` matches ``target``."""
    ratio = target / predicted.clamp_min(eps)
    valid = torch.isfinite(ratio) & (predicted > eps) & (target > eps)
    if mask is not None:
        valid = valid & mask
    if not bool(valid.any()):
        return predicted.new_tensor(1.0)
    return ratio[valid].median()


def fit_depth_scale_and_shift(
    predicted: Tensor,
    target: Tensor,
    *,
    mask: Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Least-squares ``scale, shift`` for batched affine-invariant depth.

    Inputs have shape ``(B, ...)`` and the returned tensors have shape ``(B,)``.
    Degenerate samples fall back to identity ``(1, 0)``.
    """
    if predicted.shape != target.shape:
        raise ValueError(f"shape mismatch: {predicted.shape} vs {target.shape}")
    valid = torch.isfinite(predicted) & torch.isfinite(target)
    if mask is not None:
        valid = valid & mask
    weights = valid.to(predicted.dtype)
    predicted_flat = predicted.reshape(predicted.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    weights_flat = weights.reshape(weights.shape[0], -1)

    a00 = torch.sum(weights_flat * predicted_flat.square(), dim=1)
    a01 = torch.sum(weights_flat * predicted_flat, dim=1)
    a11 = torch.sum(weights_flat, dim=1)
    b0 = torch.sum(weights_flat * predicted_flat * target_flat, dim=1)
    b1 = torch.sum(weights_flat * target_flat, dim=1)

    determinant = a00 * a11 - a01.square()
    good = determinant.abs() > eps
    safe_determinant = torch.where(good, determinant, torch.ones_like(determinant))
    scale = (a11 * b0 - a01 * b1) / safe_determinant
    shift = (-a01 * b0 + a00 * b1) / safe_determinant
    return (
        torch.where(good, scale, torch.ones_like(scale)),
        torch.where(good, shift, torch.zeros_like(shift)),
    )


def scale_depth_from_camera_poses(
    depth: Tensor,
    source_camera_to_world: Tensor,
    target_camera_to_world: Tensor,
    *,
    minimum_scale: float = 0.01,
    maximum_scale: float = 100.0,
) -> Tensor:
    _, _, scale = align_camera_poses_sim3(source_camera_to_world, target_camera_to_world)
    scale = clamp_scale(scale, minimum_scale, maximum_scale)
    return depth * scale[:, None, None, None]
