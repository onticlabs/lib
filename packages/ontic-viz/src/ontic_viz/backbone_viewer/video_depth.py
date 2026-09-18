"""Whole-clip video depth and optional calibration shared across time."""

from dataclasses import dataclass

import torch

from ontic_lib.depth.alignment import fit_depth_scale
from ontic_nn.video_depth import VIDEO_DEPTH_MODELS

VIDEO_METRIC = "Video depth · metric"
VIDEO_SENSOR_SCALE = "Video depth · clip sensor scale"
VIDEO_SOURCES = (VIDEO_METRIC, VIDEO_SENSOR_SCALE)
VIDEO_MODEL_LABELS = {
    "Video Depth Anything Small": "vda_small",
    "Video Depth Anything Base": "vda_base",
    "Video Depth Anything Large": "vda_large",
    "VeloDepth": "velodepth",
    "DA3 Nested Giant-Large 1.1 (joint clip)": "da3_nested",
}


@dataclass(frozen=True)
class VideoDepthSettings:
    name: str = "vda_small"
    checkpoint_path: str = ""
    repo_path: str = ""
    allow_download: bool = False
    input_size: int | None = None  # None keeps the selected model's registered default.
    fp32: bool = False
    resolution_level: int = 0


def build_video_depth(settings: VideoDepthSettings, device: str):
    if settings.name not in VIDEO_DEPTH_MODELS:
        raise ValueError(f"Unknown video depth model: {settings.name}")
    resolution = (
        {"resolution_level": settings.resolution_level}
        if settings.name == "velodepth"
        else {}
        if settings.input_size is None
        else {"input_size": settings.input_size}
    )
    cfg = VIDEO_DEPTH_MODELS[settings.name](
        checkpoint_path=settings.checkpoint_path or None,
        repo_path=settings.repo_path or None,
        allow_download=settings.allow_download,
        fp32=settings.fp32,
        **resolution,
    )
    return cfg.build().to(device).eval()


def prepare_video_depth(frames, settings, source, device, builder, progress, check_cancel):
    """Return (T,V,H,W) depth plus one calibration scale per camera.

    Every camera's full ordered sequence goes through the temporal model. Sensor
    calibration fits a single scale per camera over the entire clip, so calibration
    cannot introduce independent frame-to-frame scale jumps.
    """
    if source == VIDEO_SENSOR_SCALE and any(f.depth is None for f in frames):
        raise ValueError("Video clip sensor scale requires recorded depth on every clip frame")
    images = torch.stack([f.images for f in frames])[None]
    check_cancel()
    model = builder(settings, device)
    try:
        depth = model(images, progress=progress, check_cancel=check_cancel).detach().float().cpu()
    finally:
        del model
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
    check_cancel()
    expected = (*images.shape[:3], *images.shape[-2:])
    if depth.shape != expected:
        raise ValueError(f"Video depth returned {tuple(depth.shape)}; expected {expected}")
    depth = depth[0]
    depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, 0)
    scales = []
    for vi in range(depth.shape[1]):
        check_cancel()
        predicted = depth[:, vi]
        valid = predicted > 0
        scale = 1.0
        if source == VIDEO_SENSOR_SCALE:
            reference = torch.stack([f.depth[vi, 0] for f in frames])
            valid &= torch.isfinite(reference) & (reference > 0)
            if valid.sum() < 16:
                raise ValueError(
                    f"Camera {frames[0].cam_names[vi]} has too few valid samples for clip scale"
                )
            scale = float(fit_depth_scale(predicted, reference, mask=valid))
        if not bool((predicted > 0).flatten(1).any(1).all()):
            raise ValueError(
                f"Video depth has no valid predictions on a frame for camera {frames[0].cam_names[vi]}"
            )
        if not 0 < scale < float("inf"):
            raise ValueError(f"Invalid video clip calibration scale: {scale}")
        depth[:, vi] *= scale
        scales.append(scale)
    return depth, scales
