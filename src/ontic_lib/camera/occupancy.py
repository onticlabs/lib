"""Occupancy reprojection: splat a point cloud back into cameras as coverage masks.

Each point carries a world-space splat radius ``r_world = depth_source / focal_source``
(a one-pixel footprint at its source depth); in a target camera its pixel radius is
``splat_scale * r_world * focal_target / depth_in_target``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..transforms.rigid import invert_rigid_transform, transform_points
from .projection import project_camera_points


def points_with_radius(
    depth: Tensor,
    camera_to_world: Tensor,
    intrinsics_normalized: Tensor,
    confidence: Tensor | None = None,
    *,
    stride: int = 1,
    confidence_threshold: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Unproject ``(V, H, W)`` z-depth into world points plus a per-point splat radius.

    Applies the same stride / ``confidence > threshold`` selection as
    :func:`ontic_lib.structures.pointcloud_from_depth_views` with ``minimum_depth=0``,
    so the returned ``(points (N, 3), radius (N,))`` line up with a cloud built from
    the same inputs.
    """
    from ..depth.lifting import depth_to_world_points  # camera <-> depth import cycle

    views, height, width = depth.shape
    points = depth_to_world_points(
        depth.unsqueeze(-1),
        camera_to_world,
        intrinsics_normalized,
        (height, width),
        depth_type="z",
    )
    focal_x = intrinsics_normalized[:, 0, 0] * width
    focal_y = intrinsics_normalized[:, 1, 1] * height
    inverse_focal = 0.5 * (1.0 / focal_x + 1.0 / focal_y)
    radius = depth * inverse_focal.view(views, 1, 1)

    if confidence is None:
        valid = torch.ones((views, height, width), dtype=torch.bool, device=depth.device)
    else:
        valid = confidence > confidence_threshold
    valid &= torch.isfinite(depth) & (depth > 0)

    if stride > 1:
        points = points[:, ::stride, ::stride]
        radius = radius[:, ::stride, ::stride]
        valid = valid[:, ::stride, ::stride]

    valid = valid.reshape(-1)
    return points.reshape(-1, 3)[valid], radius.reshape(-1)[valid]


def _disk_kernel(radius: int, device: torch.device | str | None) -> Tensor:
    axis = torch.arange(-radius, radius + 1, device=device)
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    return ((x * x + y * y) <= radius * radius).to(torch.float32)[None, None]


def _dilate_disk(mask: Tensor, radius: int) -> Tensor:
    """Binary dilation of ``(H, W)`` bool ``mask`` with a disk structuring element."""
    if radius <= 0:
        return mask
    kernel = _disk_kernel(radius, mask.device)
    hit = F.conv2d(mask.to(torch.float32)[None, None], kernel, padding=radius)
    return hit[0, 0] > 0


def project_occupancy(
    points: Tensor,
    radius_world: Tensor,
    camera_to_world: Tensor,
    intrinsics_normalized: Tensor,
    image_size: tuple[int, int],
    *,
    splat_scale: float = 1.0,
    max_radius: int = 6,
) -> Tensor:
    """Rasterise ``(V, H, W)`` bool coverage masks of ``points (N, 3)`` in each camera.

    ``radius_world (N,)`` is scaled by ``splat_scale`` and converted to pixels through the
    target focal length and depth, then clamped to ``max_radius`` pixels.
    """
    height, width = image_size
    world_to_camera = invert_rigid_transform(camera_to_world)
    masks = []
    for view in range(camera_to_world.shape[0]):
        intrinsics = intrinsics_normalized[view]
        points_camera = transform_points(world_to_camera[view], points)
        z = points_camera[..., 2]
        xy = project_camera_points(points_camera, intrinsics)

        focal = 0.5 * (float(intrinsics[0, 0]) * width + float(intrinsics[1, 1]) * height)
        radius_px = splat_scale * radius_world * focal / z.clamp(min=1e-6)
        radius_px = radius_px.clamp(0, max_radius).round().to(torch.int64)

        valid = (z > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < 1) & (xy[:, 1] >= 0) & (xy[:, 1] < 1)
        px = (xy[:, 0] * width).floor().to(torch.int64).clamp(0, width - 1)
        py = (xy[:, 1] * height).floor().to(torch.int64).clamp(0, height - 1)

        mask = torch.zeros((height, width), dtype=torch.bool, device=points.device)
        for rad in torch.unique(radius_px[valid]).tolist() if bool(valid.any()) else []:
            selected = valid & (radius_px == rad)
            base = torch.zeros((height, width), dtype=torch.bool, device=points.device)
            base[py[selected], px[selected]] = True
            mask |= _dilate_disk(base, int(rad))
        masks.append(mask)
    return torch.stack(masks)


def overlay_masks_on_images(
    images: Tensor,
    masks: Tensor,
    color: tuple[float, float, float] = (0.0, 1.0, 0.0),
    alpha: float = 0.5,
) -> Tensor:
    """Blend ``color`` over the masked pixels of ``(V, 3, H, W)`` images in ``[0, 1]``."""
    out = images.clone().float()
    mask = torch.as_tensor(masks, dtype=torch.bool, device=out.device).unsqueeze(1)
    tint = torch.tensor(color, dtype=out.dtype, device=out.device).view(1, 3, 1, 1)
    blended = out * (1 - alpha) + tint * alpha
    return torch.where(mask, blended, out)
