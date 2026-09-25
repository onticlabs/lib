"""Joint-clip DA3 Nested inference, including the native metric scaling branch."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ontic_nn.wrappers.common import resize_to_long_side
from ontic_nn.wrappers.da3 import IMAGENET_MEAN, IMAGENET_STD, load_da3

from .common import camera_videos, research_checkout


@dataclass(frozen=True, kw_only=True)
class DA3VideoConfig:
    repo_path: str | None = None
    checkpoint_path: str | None = None  # HF snapshot directory, including config.json
    cache_dir: str | None = None
    allow_download: bool = False
    input_size: int = 504  # long side, unlike VDA's short-side control
    fp32: bool = False

    def __post_init__(self):
        if self.input_size < 14 or self.input_size % 14:
            raise ValueError("DA3 video input_size must be a positive multiple of 14")

    def build(self):
        return DA3VideoModel(self)


class DA3VideoModel(nn.Module):
    def __init__(self, cfg: DA3VideoConfig):
        super().__init__()
        self.cfg = cfg
        with research_checkout(cfg.repo_path, "depth_anything_3", "src/depth_anything_3/api.py"):
            api = load_da3(
                "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
                "da3nested-giant-large",
                checkpoint_path=cfg.checkpoint_path,
                cache_dir=cfg.cache_dir,
                allow_download=cfg.allow_download,
            )
        # Calling the underlying network preserves both nested branches and lets
        # us control precision (the high-level API always enables autocast).
        self.model = api.model
        if not hasattr(self.model, "da3_metric"):
            raise ValueError("DA3 video depth requires a Nested checkpoint with a metric branch")
        self.model.requires_grad_(False).eval()

    @torch.inference_mode()
    def forward(self, images, *, progress=lambda msg: None, check_cancel=lambda: None):
        self.model.eval()
        device = next(self.model.parameters()).device
        if device.type not in ("cpu", "cuda"):
            raise ValueError("DA3 video depth supports CPU or CUDA inference")

        def infer(frames):
            resized = resize_to_long_side(
                frames.to(device=device, dtype=torch.float32), 14, self.cfg.input_size
            )
            mean = resized.new_tensor(IMAGENET_MEAN)[None, :, None, None]
            std = resized.new_tensor(IMAGENET_STD)[None, :, None, None]
            dtype = (
                torch.bfloat16
                if device.type == "cuda" and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda" and not self.cfg.fp32,
            ):
                # T is the joint any-view axis. Never split into separately
                # scaled windows or confuse T with the independent batch axis.
                result = self.model(((resized - mean) / std)[None], export_feat_layers=[])
            if result.get("is_metric") != 1:
                raise ValueError("DA3 video checkpoint did not produce metric depth")
            depth = result["depth"]
            if depth.ndim != 4 or depth.shape[:2] != (1, frames.shape[0]):
                raise ValueError("DA3 video depth must have shape (1,T,H,W)")
            return F.interpolate(
                depth[0, :, None].float(),
                size=frames.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[:, 0]

        return camera_videos(images, infer, progress=progress, check_cancel=check_cancel)
