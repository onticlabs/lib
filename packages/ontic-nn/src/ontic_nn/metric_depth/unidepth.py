"""UniDepthV2 metric monocular model (predicts metric depth and its own intrinsics).

``infer(normalize=True)`` takes 0-255 RGB and applies the ImageNet normalisation itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ontic_nn.wrappers.common import (
    import_research_module,
    normalize_intrinsics_pixels,
    resolve_checkpoint,
)

from .common import MetricDepthConfig, MetricDepthModel, MetricDepthOutput, resize_to_long_side
from .registry import register_metric_model

UNIDEPTH_REPO_URL = "https://github.com/lpiccinelli-eth/UniDepth"


@register_metric_model("unidepth")
@dataclass(kw_only=True)
class UniDepthMetricConfig(MetricDepthConfig):
    """``model_dir`` is the HF repo id (``PyTorchModelHubMixin``); ``checkpoint_path`` a local
    snapshot directory. 644 = 46 * 14 keeps the input near the ~480x640 training resolution."""

    model_dir: str = "lpiccinelli/unidepth-v2-vitl14"
    long_side: int = 644

    def build(self) -> UniDepthMetricModel:
        return UniDepthMetricModel(self)


class UniDepthMetricModel(MetricDepthModel):
    PATCH_SIZE = 14

    def __init__(self, cfg: UniDepthMetricConfig) -> None:
        super().__init__()
        self.cfg = cfg
        models = import_research_module(
            "unidepth.models",
            extra="unidepth",
            repo_url=UNIDEPTH_REPO_URL,
            what="UniDepthMetricModel",
        )
        path = resolve_checkpoint(
            cfg.checkpoint_path,
            cfg.model_dir,
            cache_dir=cfg.cache_dir,
            allow_download=cfg.allow_download,
            extra="unidepth",
            what="UniDepthV2 snapshot",
        )
        if path is None:
            raise ValueError(
                "UniDepthMetricModel needs a checkpoint: set model_dir or checkpoint_path"
            )
        model = models.UniDepthV2.from_pretrained(path)
        model.interpolation_mode = "bilinear"
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images: Tensor, intrinsics: Optional[Tensor] = None) -> MetricDepthOutput:
        images = resize_to_long_side(images, self.PATCH_SIZE, self.cfg.long_side)
        b, v, _c, h, w = images.shape
        rgb = (images.reshape(b * v, 3, h, w) * 255.0).to(self.model.device)
        out = self.model.infer(rgb, camera=None, normalize=True)
        depth = out["depth"].float().reshape(b, v, h, w)
        conf = out.get("confidence")
        return MetricDepthOutput(
            depth=depth,
            conf=None if conf is None else conf.float().reshape(b, v, h, w),
            intrinsics=normalize_intrinsics_pixels(
                out["intrinsics"].float().reshape(b, v, 3, 3), h, w
            ),
        )
