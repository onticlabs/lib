"""VeloDepth metric video depth with independent temporal state for each camera."""

import contextlib
from dataclasses import dataclass
import inspect
from pathlib import Path

import torch
from torch import nn

from ontic_nn.wrappers.common import import_research_module, offline_guard, resolve_checkpoint

from .common import camera_videos, research_checkout

REPO_URL = "https://github.com/lpiccinelli-eth/velodepth"


@dataclass(frozen=True, kw_only=True)
class VeloDepthConfig:
    repo_path: str | None = None
    checkpoint_path: str | None = None  # HF snapshot directory, including config.json
    cache_dir: str | None = None
    allow_download: bool = False
    resolution_level: int = 0  # native pixel-budget bucket, 0..9
    fp32: bool = False

    def __post_init__(self):
        if not isinstance(self.resolution_level, int) or not 0 <= self.resolution_level <= 9:
            raise ValueError("VeloDepth resolution_level must be an integer in [0,9]")

    def build(self):
        return VeloDepthModel(self)


def _bundled_hub(checkpoint):
    """``checkpoint`` itself when the snapshot ships ``checkpoints/`` torch.hub assets."""
    if checkpoint and (Path(checkpoint) / "checkpoints").is_dir():
        return Path(checkpoint)
    return None


@contextlib.contextmanager
def _torch_hub_dir(path):
    """Serve ``torch.hub`` checkpoints from ``path/checkpoints``; restores the hub dir on exit."""
    if path is None:
        yield
        return
    import torch.hub

    previous = torch.hub.get_dir()
    torch.hub.set_dir(str(path))
    try:
        yield
    finally:
        torch.hub.set_dir(previous)


class VeloDepthModel(nn.Module):
    def __init__(self, cfg: VeloDepthConfig):
        super().__init__()
        self.cfg = cfg
        with research_checkout(cfg.repo_path, "velodepth", "velodepth/models/velodepth.py"):
            upstream = import_research_module(
                "velodepth.models", extra="velodepth", repo_url=REPO_URL, what="VeloDepth"
            )
            checkpoint = resolve_checkpoint(
                cfg.checkpoint_path,
                "lpiccinelli/velodepth",
                cache_dir=cfg.cache_dir,
                allow_download=cfg.allow_download,
                extra="velodepth",
                what="VeloDepth snapshot",
            )
            # Upstream also loads ConvNeXt initializers through torch.hub while
            # constructing the model, before loading the complete HF checkpoint; the ontic
            # artifact carries them under ``checkpoints/`` in torch.hub's layout.
            with offline_guard(cfg.allow_download), _torch_hub_dir(_bundled_hub(checkpoint)):
                self.model = upstream.VeloDepth.from_pretrained(checkpoint)
        self.model.resolution_level = cfg.resolution_level
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def forward(self, images, *, progress=lambda msg: None, check_cancel=lambda: None):
        self.model.eval()
        device = next(self.model.parameters()).device
        if device.type not in ("cpu", "cuda"):
            raise ValueError("VeloDepth supports CPU or CUDA inference")

        def infer(frames):
            # Upstream decorates infer with unconditional CUDA fp16 autocast.
            # Unwrap those decorators so CPU/full-precision mode is respected;
            # inference_mode and the precision context are supplied here instead.
            inference = inspect.unwrap(type(self.model).infer)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and not self.cfg.fp32,
            ):
                result = inference(self.model, frames.to(device=device, dtype=torch.float32) * 255)
            # 'distance' is radial range; the calibrated viewer needs camera-z.
            depth = result["depth"]
            if depth.ndim != 4 or depth.shape[1] != 1:
                raise ValueError("VeloDepth depth must have shape (T,1,H,W)")
            return depth[:, 0]

        return camera_videos(images, infer, progress=progress, check_cancel=check_cancel)
