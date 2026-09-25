"""Apple Depth Pro metric monocular model (self-estimates the focal length).

``model.infer`` resizes its input to the network's 1536² internally and returns depth at the
grid we pass, plus ``focallength_px`` for that width; the normalised K assumes square pixels
and a central principal point. Input normalisation is ``x * 2 - 1`` (the upstream transform).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional

import torch
from torch import Tensor

from ontic_nn.wrappers.common import import_research_module, resolve_checkpoint

from .common import MetricDepthConfig, MetricDepthModel, MetricDepthOutput, resize_to_long_side
from .registry import register_metric_model

DEPTH_PRO_REPO_URL = "https://github.com/apple/ml-depth-pro"


@register_metric_model("depthpro")
@dataclass(kw_only=True)
class DepthProMetricConfig(MetricDepthConfig):
    """``checkpoint_path`` is the local ``depth_pro.pt``; else ``hf_filename`` of ``hf_repo``."""

    hf_repo: str = "apple/DepthPro"
    hf_filename: str = "depth_pro.pt"
    long_side: int = 1536

    def build(self) -> DepthProMetricModel:
        return DepthProMetricModel(self)


class DepthProMetricModel(MetricDepthModel):
    PATCH_SIZE = 16

    def __init__(self, cfg: DepthProMetricConfig) -> None:
        super().__init__()
        self.cfg = cfg
        depth_pro = import_research_module(
            "depth_pro.depth_pro",
            extra="depthpro",
            repo_url=DEPTH_PRO_REPO_URL,
            what="DepthProMetricModel",
        )
        ckpt = resolve_checkpoint(
            cfg.checkpoint_path,
            cfg.hf_repo,
            filename=cfg.hf_filename,
            cache_dir=cfg.cache_dir,
            allow_download=cfg.allow_download,
            extra="depthpro",
            what="Depth Pro checkpoint",
        )
        config = replace(depth_pro.DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=ckpt)
        model, _transform = depth_pro.create_model_and_transforms(config=config)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, images: Tensor, intrinsics: Optional[Tensor] = None) -> MetricDepthOutput:
        images = resize_to_long_side(images, self.PATCH_SIZE, self.cfg.long_side)
        b, v, c, h, w = images.shape
        flat = images.reshape(b * v, c, h, w).to(next(self.model.parameters()).dtype)
        flat = flat * 2.0 - 1.0

        depths: List[Tensor] = []
        focals: List[Tensor] = []
        with torch.inference_mode():
            for i in range(flat.shape[0]):
                pred = self.model.infer(flat[i], f_px=None)
                depths.append(pred["depth"].reshape(h, w).float())
                focals.append(pred["focallength_px"].reshape(()).float())
        depth = torch.stack(depths).reshape(b, v, h, w)
        f_px = torch.stack(focals).reshape(b, v)

        k = torch.zeros(b, v, 3, 3, device=depth.device, dtype=depth.dtype)
        k[..., 0, 0] = f_px / w
        k[..., 1, 1] = f_px / h
        k[..., 0, 2] = 0.5
        k[..., 1, 2] = 0.5
        k[..., 2, 2] = 1.0
        return MetricDepthOutput(depth=depth, conf=None, intrinsics=k)
