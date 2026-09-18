"""Metric Video Depth Anything, using upstream's overlapping temporal windows.

The time and camera axes are deliberately separate: each camera supplies one
ordered video. Inputs are RGB floats (B,T,V,3,H,W) in [0,1]; outputs are CPU
float32 camera-z depth in meters (B,T,V,H,W), at the original image resolution.
Research imports and checkpoint downloads happen only on build.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
import sys

import torch
from torch import Tensor, nn

from ontic_nn.wrappers.common import import_research_module, resolve_checkpoint

from .common import camera_videos

REPO_URL = "https://github.com/DepthAnything/Video-Depth-Anything"
_ENCODERS = {
    "vits": ("Small", 64, [48, 96, 192, 384]),
    "vitb": ("Base", 128, [96, 192, 384, 768]),
    "vitl": ("Large", 256, [256, 512, 1024, 1024]),
}


@dataclass(frozen=True, kw_only=True)
class VideoDepthAnythingConfig:
    encoder: str = "vits"
    repo_path: str | None = None
    checkpoint_path: str | None = None
    cache_dir: str | None = None
    allow_download: bool = False
    input_size: int = 518  # upstream's short-side inference size, not output size
    fp32: bool = False

    def __post_init__(self):
        if self.encoder not in _ENCODERS:
            raise ValueError(f"Unknown Video Depth Anything encoder: {self.encoder}")
        if self.input_size < 14 or self.input_size % 14:
            raise ValueError("Video depth input_size must be a positive multiple of 14")

    def build(self) -> VideoDepthAnythingModel:
        return VideoDepthAnythingModel(self)


@dataclass(frozen=True, kw_only=True)
class VDABaseConfig(VideoDepthAnythingConfig):
    encoder: str = "vitb"


@dataclass(frozen=True, kw_only=True)
class VDALargeConfig(VideoDepthAnythingConfig):
    encoder: str = "vitl"


def _import_vda(repo_path):
    path = None
    if repo_path:
        root = Path(repo_path).expanduser().resolve()
        if not (root / "video_depth_anything/video_depth.py").is_file():
            raise ValueError("Video depth research checkout must be the Video-Depth-Anything root")
        # Upstream also imports a top-level utils.util. Refuse an already-loaded
        # foreign module rather than silently borrowing another research repo's code.
        for name in ("video_depth_anything", "utils", "utils.util"):
            loaded = sys.modules.get(name)
            if loaded is None:
                continue
            locations = list(getattr(loaded, "__path__", []))
            if getattr(loaded, "__file__", None):
                locations.append(loaded.__file__)
            if not locations or any(not Path(p).resolve().is_relative_to(root) for p in locations):
                raise ImportError(
                    f"Another {name} module is loaded; restart with the Video-Depth-Anything checkout"
                )
        path = str(root)
        sys.path.insert(0, path)
    try:
        return import_research_module(
            "video_depth_anything.video_depth",
            extra="video-depth",
            repo_url=REPO_URL,
            what="VideoDepthAnythingModel",
        )
    finally:
        if path is not None:
            sys.path.remove(path)


class VideoDepthAnythingModel(nn.Module):
    def __init__(self, cfg: VideoDepthAnythingConfig):
        super().__init__()
        self.cfg = cfg
        upstream = _import_vda(cfg.repo_path)
        size, features, channels = _ENCODERS[cfg.encoder]
        checkpoint = resolve_checkpoint(
            cfg.checkpoint_path,
            f"depth-anything/Metric-Video-Depth-Anything-{size}",
            filename=f"metric_video_depth_anything_{cfg.encoder}.pth",
            cache_dir=cfg.cache_dir,
            allow_download=cfg.allow_download,
            extra="video-depth",
            what=f"Metric Video Depth Anything {size} checkpoint",
        )
        self.model = upstream.VideoDepthAnything(
            encoder=cfg.encoder, features=features, out_channels=channels, metric=True
        )
        self.model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True
        )
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def forward(self, images: Tensor, *, progress=lambda msg: None, check_cancel=lambda: None):
        device = next(self.model.parameters()).device
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Video Depth Anything supports CPU or CUDA inference")
        self.model.eval()

        def infer(video):
            frames = (
                video.detach()
                .cpu()
                .permute(0, 2, 3, 1)
                .mul(255)
                .round()
                .to(torch.uint8)
                .contiguous()
                .numpy()
            )
            # Upstream uses device as torch.autocast(device_type=...), which
            # requires 'cuda', not 'cuda:1'. Select the actual device separately.
            context = (
                torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()
            )
            with context:
                depth, _ = self.model.infer_video_depth(
                    frames,
                    -1,
                    input_size=self.cfg.input_size,
                    device=device.type,
                    fp32=self.cfg.fp32 or device.type == "cpu",
                )
            return depth

        return camera_videos(images, infer, progress=progress, check_cancel=check_cancel)
