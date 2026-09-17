"""DA3 metric monocular model (nested checkpoint: both branches + metric alignment).

The upstream ``DepthAnything3`` forward on the nested checkpoint returns metric depth.

Self-calibrating: input ``intrinsics`` are ignored; the predicted ones are returned.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from ontic_nn.wrappers.common import normalize_intrinsics_pixels
from ontic_nn.wrappers.da3 import IMAGENET_MEAN, IMAGENET_STD, load_da3

from .common import MetricDepthConfig, MetricDepthModel, MetricDepthOutput, resize_to_long_side
from .registry import register_metric_model


@register_metric_model("da3")
@dataclass(kw_only=True)
class DA3MetricConfig(MetricDepthConfig):
    """``model_dir`` is the HF repo id (a nested checkpoint, for metric depth)."""

    model_dir: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    model_name: str = "da3-large"
    long_side: int = 504

    def build(self) -> DA3MetricModel:
        return DA3MetricModel(self)


class DA3MetricModel(MetricDepthModel):
    PATCH_SIZE = 14

    def __init__(self, cfg: DA3MetricConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.da3 = load_da3(
            cfg.model_dir,
            cfg.model_name,
            checkpoint_path=cfg.checkpoint_path,
            cache_dir=cfg.cache_dir,
            allow_download=cfg.allow_download,
        )
        self.da3.eval()
        for p in self.da3.parameters():
            p.requires_grad_(False)
        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 1, 3, 1, 1), persistent=False
        )

    def forward(self, images: Tensor, intrinsics: Optional[Tensor] = None) -> MetricDepthOutput:
        images = resize_to_long_side(images, self.PATCH_SIZE, self.cfg.long_side)
        h, w = images.shape[-2:]
        imgs_norm = (images - self.image_mean) / self.image_std
        out = self.da3(imgs_norm, export_feat_layers=[])  # metric depth (is_metric = 1)
        k = out.get("intrinsics", None)
        return MetricDepthOutput(
            depth=out["depth"],
            conf=out.get("depth_conf", None),
            intrinsics=None if k is None else normalize_intrinsics_pixels(k, h, w),
        )
