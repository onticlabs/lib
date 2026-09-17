"""Pure render helpers: colour modes, frustum math, point-cloud assembly (no viser, no GPU).

Also places the predicted cameras in the display frame."""

from __future__ import annotations

import matplotlib
import numpy as np
import torch
from torch import Tensor

from ontic_lib.pointops import (
    align_camera_poses_sim3,
    align_cameras_sim3,
    compute_alignment,
    furthest_point_sample,
    space_filling_stride,
    voxel_pool,
)
from ontic_lib.structures import crop_to_aabb
from ontic_lib.transforms.rotations import matrix_to_quaternion

from .runner import AlignMode, BackboneResult, metric_unproject

COLOR_MODES = ["RGB", "confidence", "per-view"]

_PALETTE = torch.tensor(
    [
        [0.90, 0.10, 0.10],
        [0.10, 0.60, 0.90],
        [0.20, 0.80, 0.20],
        [0.95, 0.70, 0.10],
        [0.70, 0.30, 0.90],
        [0.10, 0.85, 0.80],
        [0.95, 0.45, 0.75],
        [0.55, 0.55, 0.20],
        [0.40, 0.40, 0.95],
        [0.85, 0.35, 0.20],
        [0.30, 0.75, 0.50],
        [0.65, 0.20, 0.45],
    ]
)


def mat_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    """``(3, 3)`` rotation matrix -> unit ``(w, x, y, z)`` quaternion (numpy float64)."""
    q = matrix_to_quaternion(torch.as_tensor(np.asarray(rotation), dtype=torch.float64))
    return q.numpy()


def frustum_params(intrinsics_norm: Tensor) -> tuple[float, float]:
    """``(vertical fov [rad], aspect W/H)`` from normalised ``(3, 3)`` intrinsics."""
    fx_n = float(intrinsics_norm[0, 0])
    fy_n = float(intrinsics_norm[1, 1])
    return 2.0 * float(np.arctan(0.5 / fy_n)), fy_n / fx_n


def conf_to_rgb(conf: Tensor) -> Tensor:
    """``(V, Hd, Wd)`` confidence -> ``(V, 3, Hd, Wd)`` turbo colours, 5-95th pct normalised."""
    flat = conf.reshape(-1)
    lo, hi = torch.quantile(flat, 0.05), torch.quantile(flat, 0.95)
    norm = ((conf - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
    rgba = matplotlib.colormaps["turbo"](norm.cpu().numpy())  # (V, Hd, Wd, 4)
    return torch.from_numpy(rgba[..., :3]).float().permute(0, 3, 1, 2)


def view_palette_image(views: int, hw: tuple[int, int]) -> Tensor:
    """``(V, 3, H, W)`` image with one solid palette colour per view."""
    h, w = hw
    cols = _PALETTE[torch.arange(views) % _PALETTE.shape[0]]
    return cols.view(views, 3, 1, 1).expand(views, 3, h, w).contiguous()


def pred_cameras_in_display_frame(
    result: BackboneResult,
    align_mode: AlignMode | str = AlignMode.NONE,
) -> tuple[Tensor, Tensor] | None:
    """Predicted poses ``(V, 4, 4)`` + intrinsics ``(V, 3, 3)`` in the frame the cloud is
    displayed in, or ``None`` without predicted poses (or ``metric_mono`` before its
    scale is fitted)."""
    mode = AlignMode(align_mode)
    pe = result.pred_extrinsics
    if pe is None:
        return None
    intr = result.pred_intrinsics if result.pred_intrinsics is not None else result.gt_intrinsics
    ge = result.gt_extrinsics
    if mode is AlignMode.METRIC_MONO:
        if result.metric_scale is None:
            return None
        gi = None if result.gt_intrinsics is None else result.gt_intrinsics.unsqueeze(0)
        _, lift, _ = compute_alignment(
            pe.unsqueeze(0),
            intr.unsqueeze(0),
            None if ge is None else ge.unsqueeze(0),
            gi,
            "metric_mono",
            metric_scale=result.metric_scale,
            depth=result.depth.unsqueeze(0),
        )
        return lift[0], intr
    if mode in (AlignMode.SIM3_POINTS, AlignMode.PRESCALE_GT):
        r, t, s = align_camera_poses_sim3(pe.unsqueeze(0), ge.unsqueeze(0))
        return align_cameras_sim3(pe.unsqueeze(0), r, t, s)[0], intr
    return pe, intr


def build_point_cloud(
    result: BackboneResult,
    images: Tensor,
    *,
    color_mode: str = "RGB",
    stride: int = 1,
    conf_thresh: float = 0.0,
    align_mode: AlignMode | str = AlignMode.NONE,
    crop_lo=None,
    crop_hi=None,
    voxel_size: float = 0.0,
    sfc_stride: int = 1,
    max_points: int = -1,
) -> tuple[np.ndarray, np.ndarray]:
    """``(points (N, 3) float32, colors (N, 3) uint8)`` numpy arrays for viser.

    Point-count controls apply in order: stride + confidence (at unprojection), crop
    box, voxel pooling, space-filling-curve stride (grid ``max(voxel_size, 0.004)``),
    furthest-point sampling to ``max_points`` (``<= 0`` disables).
    """
    depth = result.depth
    if color_mode == "confidence" and result.conf is not None:
        rgb = conf_to_rgb(result.conf)
    elif color_mode == "per-view":
        rgb = view_palette_image(depth.shape[0], (depth.shape[1], depth.shape[2]))
    else:
        rgb = images

    pc = metric_unproject(result, rgb, align_mode, stride=stride, conf_thresh=conf_thresh)
    points, colors = pc.points, pc.colors
    if crop_lo is not None and crop_hi is not None:
        points, colors = crop_to_aabb(points, colors, crop_lo, crop_hi)
    if voxel_size > 0:
        points, colors = voxel_pool(points, colors, voxel_size)
    if sfc_stride > 1:
        points, colors = space_filling_stride(points, colors, sfc_stride, voxel_size)
    if max_points > 0:
        points, colors = furthest_point_sample(points, colors, max_points)

    points_np = points.numpy().astype(np.float32)
    colors_np = (colors.clamp(0.0, 1.0) * 255).to(torch.uint8).numpy()
    return points_np, colors_np
