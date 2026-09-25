"""Camera / point-set alignment (SE(3), Sim(3)) and metric alignment of predicted depth.

The metric modes place predicted depth + cameras into a target (GT) world frame."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ..depth.lifting import depth_to_world_points
from ..transforms.rigid import invert_rigid_transform, transform_camera_to_world_sim3

AlignmentMode = Literal["none", "prescale_gt", "sim3_points", "metric_mono"]
ALIGNMENT_MODES: tuple[str, ...] = ("none", "prescale_gt", "sim3_points", "metric_mono")


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


def percentile_conf_threshold(confidence: Tensor | None, drop_pct: float) -> float:
    """Absolute threshold that drops the lowest ``drop_pct`` percent of ``confidence``.

    Backbone-agnostic ("drop the lowest 20%" regardless of the raw confidence scale).
    Returns ``-inf`` (keep everything) when ``drop_pct <= 0`` or there is no confidence.
    Use with ``confidence > threshold``.
    """
    if confidence is None or drop_pct <= 0:
        return float("-inf")
    quantile = min(drop_pct, 100.0) / 100.0
    return float(torch.quantile(confidence.reshape(-1).float(), quantile))


def conf_drop_mask(confidence: Tensor | None, drop_pct: float) -> Tensor | None:
    """Per-sample keep mask above the ``drop_pct``-th confidence percentile.

    ``confidence`` has shape ``(B, ...)``; the quantile is taken over each sample's
    remaining dims. Returns a bool tensor of the same shape, or ``None`` when there is
    nothing to drop (no confidence, or ``drop_pct <= 0``).
    """
    if confidence is None or drop_pct <= 0:
        return None
    batch = confidence.shape[0]
    quantile = min(drop_pct, 100.0) / 100.0
    threshold = torch.quantile(confidence.reshape(batch, -1).float(), quantile, dim=1)
    return confidence > threshold.view(batch, *([1] * (confidence.ndim - 1)))


def _metric_cloud_centroid(
    depth: Tensor | None,
    scale: Tensor,
    camera_to_world: Tensor,
    intrinsics_normalized: Tensor | None,
) -> Tensor:
    """Per-sample mean ``(B, 3)`` of the ``depth * scale`` cloud (camera-centre mean if
    depth or intrinsics are absent). Detached: a frame choice, not a learnable op."""
    centers = camera_to_world[..., :3, 3]
    if depth is None or intrinsics_normalized is None:
        return centers.mean(1)
    with torch.no_grad():
        scaled = (depth.detach() * scale[:, None, None, None]).unsqueeze(-1)  # (B, V, H, W, 1)
        image_size = (depth.shape[-2], depth.shape[-1])
        origins = []
        for b in range(depth.shape[0]):
            points = depth_to_world_points(
                scaled[b], camera_to_world[b], intrinsics_normalized[b], image_size, depth_type="z"
            )
            valid = torch.isfinite(points).all(-1) & (depth[b] > 1e-6)
            origins.append(points[valid].mean(0) if bool(valid.any()) else centers[b].mean(0))
        return torch.stack(origins)


def compute_alignment(
    predicted_camera_to_world: Tensor | None,
    predicted_intrinsics: Tensor | None,
    target_camera_to_world: Tensor | None,
    target_intrinsics: Tensor | None,
    mode: AlignmentMode = "prescale_gt",
    *,
    metric_scale: Tensor | float | None = None,
    depth: Tensor | None = None,
    clamp: tuple[float, float] = (0.01, 100.0),
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute (but do not apply) a metric alignment: ``(scale, camera_to_world, intrinsics)``.

    Returns the per-sample depth scale ``(B,)`` and the cameras to lift the scaled depth
    with. Poses are camera-to-world ``(B, V, 4, 4)``; intrinsics normalized ``(B, V, 3, 3)``.

    * ``"none"``: scale 1, predicted cameras (target cameras if there are no predictions).
    * ``"prescale_gt"``: Sim(3) fit predicted -> target; its scale (clamped to ``clamp``),
      lift with the **target** cameras.
    * ``"sim3_points"``: the same scale, lift with the Sim(3)-mapped predicted cameras.
    * ``"metric_mono"``: ``scale = metric_scale`` (required); predicted camera centres are
      rescaled by it, then placed by a rigid SE(3) fit to the target cameras, or — without
      target cameras — centred on the mean of the scaled cloud (needs ``depth``).
    """
    if mode == "none":
        cameras = (
            predicted_camera_to_world
            if predicted_camera_to_world is not None
            else target_camera_to_world
        )
        intrinsics = predicted_intrinsics if predicted_intrinsics is not None else target_intrinsics
        return cameras.new_ones(cameras.shape[0]), cameras, intrinsics

    if mode == "metric_mono":
        if metric_scale is None:
            raise ValueError("compute_alignment(metric_mono): metric_scale is required")
        reference = (
            target_camera_to_world
            if target_camera_to_world is not None
            else predicted_camera_to_world
        )
        scale = torch.as_tensor(metric_scale, dtype=reference.dtype, device=reference.device)
        if scale.ndim == 0:
            scale = scale.expand(reference.shape[0])
        if predicted_camera_to_world is None:
            return scale, target_camera_to_world, target_intrinsics
        scaled = predicted_camera_to_world.clone()
        scaled[..., :3, 3] = scaled[..., :3, 3] * scale[:, None, None]
        if target_camera_to_world is not None:
            rotation, translation = align_camera_poses_se3(scaled, target_camera_to_world)
        else:
            rotation = torch.eye(3, dtype=scaled.dtype, device=scaled.device).expand(
                scaled.shape[0], 3, 3
            )
            translation = -_metric_cloud_centroid(depth, scale, scaled, predicted_intrinsics)
        transform = torch.eye(4, dtype=scaled.dtype, device=scaled.device).repeat(
            scaled.shape[0], 1, 1
        )
        transform[:, :3, :3] = rotation
        transform[:, :3, 3] = translation
        return scale, transform[:, None] @ scaled, predicted_intrinsics

    if mode not in ("prescale_gt", "sim3_points"):
        raise ValueError(f"compute_alignment: unknown mode {mode!r}")

    rotation, translation, scale = align_camera_poses_sim3(
        predicted_camera_to_world, target_camera_to_world
    )
    scale = clamp_scale(scale, *clamp)
    if mode == "prescale_gt":
        return scale, target_camera_to_world, target_intrinsics
    aligned = align_cameras_sim3(predicted_camera_to_world, rotation, translation, scale)
    return scale, aligned, predicted_intrinsics


def apply_metric_scale(depth: Tensor, scale: Tensor) -> Tensor:
    """Multiply ``(B, V, H, W)`` depth by a per-sample scale ``(B,)``."""
    return depth * scale[:, None, None, None]


def align(
    depth: Tensor,
    predicted_camera_to_world: Tensor | None,
    predicted_intrinsics: Tensor | None,
    target_camera_to_world: Tensor | None,
    target_intrinsics: Tensor | None,
    mode: AlignmentMode = "prescale_gt",
    *,
    metric_scale: Tensor | float | None = None,
    clamp: tuple[float, float] = (0.01, 100.0),
) -> tuple[Tensor, Tensor, Tensor]:
    """:func:`compute_alignment` + :func:`apply_metric_scale`: ``(depth, cameras, intrinsics)``."""
    scale, cameras, intrinsics = compute_alignment(
        predicted_camera_to_world,
        predicted_intrinsics,
        target_camera_to_world,
        target_intrinsics,
        mode,
        metric_scale=metric_scale,
        depth=depth,
        clamp=clamp,
    )
    return apply_metric_scale(depth, scale), cameras, intrinsics
