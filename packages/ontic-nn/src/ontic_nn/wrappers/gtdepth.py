"""GT-depth + DINOv2 backbone: no depth prediction.

A (by default fine-tuned) DINOv2 provides the four patch-feature taps; the ground-truth
``depth`` passed to ``forward`` is resized to the input resolution and emitted as is, with
``depth_conf = 1`` and ``sky_mask = depth <= 0`` (invalid GT pixels). GT cameras pass through.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from ontic_nn.dinov2 import DINOv2, EMBED_DIM, Variant, load_pretrained_dinov2

from .common import (
    BackboneBase,
    BackboneConfig,
    BackboneOutput,
    offline_guard,
    resize_to_long_side,
)
from .da3 import IMAGENET_MEAN, IMAGENET_STD
from .registry import register_backbone

VIT_DEPTH: Dict[str, int] = {"vits": 12, "vitb": 12, "vitl": 24, "vitg": 40}


@register_backbone("gtdepth")
@dataclass(kw_only=True)
class GTDepthBackboneConfig(BackboneConfig):
    """``checkpoint_path`` is a local DINOv2 ``.pth`` state dict; else Meta's release is loaded
    through the torch hub cache. ``intermediate_layers`` are the 4 tapped blocks (``None`` → 4
    evenly spaced blocks ending at the last one)."""

    variant: Variant = "vitb"
    use_reg: bool = False
    long_side: int = 518
    freeze_backbone: bool = False
    intermediate_layers: Optional[Tuple[int, ...]] = None

    def build(self) -> GTDepthBackbone:
        return GTDepthBackbone(self)


class GTDepthBackbone(BackboneBase):
    PATCH_SIZE = 14
    ACCEPTS_GT_CAMERAS = True

    def __init__(self, cfg: GTDepthBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._embed_dim = EMBED_DIM[cfg.variant]
        self.model = self._load(cfg)
        if cfg.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

        depth = VIT_DEPTH[cfg.variant]
        if cfg.intermediate_layers is not None:
            self.tap_layers = list(cfg.intermediate_layers)
        else:
            self.tap_layers = [round((i + 1) * depth / 4) - 1 for i in range(4)]

        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @staticmethod
    def _load(cfg: GTDepthBackboneConfig):
        if cfg.checkpoint_path:
            model = DINOv2(
                cfg.variant,
                use_checkpointing=cfg.gradient_checkpointing,
                num_register_tokens=4 if cfg.use_reg else 0,
            )
            state = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            return model
        with offline_guard(cfg.allow_download):
            return load_pretrained_dinov2(
                cfg.variant, use_reg=cfg.use_reg, use_checkpointing=cfg.gradient_checkpointing
            )

    def train(self, mode: bool = True) -> GTDepthBackbone:
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.model.eval()
        return self

    @property
    def encoder_dim(self) -> int:
        return self._embed_dim

    def forward(
        self,
        images: Tensor,
        extrinsics: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
    ) -> BackboneOutput:
        images = resize_to_long_side(images, self.PATCH_SIZE, self.cfg.long_side)
        b, v, _c, h, w = images.shape
        ph, pw = h // self.PATCH_SIZE, w // self.PATCH_SIZE

        flat = rearrange(images, "B V C H W -> (B V) C H W")
        flat = (flat - self.image_mean) / self.image_std
        feats = self.model.get_intermediate_layers(flat, n=self.tap_layers, reshape=True, norm=True)
        patch_feats = [rearrange(f, "(B V) C Ph Pw -> B V Ph Pw C", B=b, V=v) for f in feats]

        sky_mask = None
        if depth is not None:
            if depth.dim() == 5:  # (B, V, 1, H, W)
                depth = depth.squeeze(2)
            depth_flat = rearrange(depth, "B V H W -> (B V) 1 H W").float()
            depth_flat = F.interpolate(depth_flat, size=(h, w), mode="nearest")
            depth_out = rearrange(depth_flat, "(B V) 1 H W -> B V H W", B=b, V=v)
            sky_mask = (depth_out <= 0).to(images.dtype)
        else:
            depth_out = torch.ones(b, v, h, w, device=images.device, dtype=images.dtype)

        dd: BackboneOutput.DataDict = {
            "depth": depth_out,
            "depth_conf": torch.ones_like(depth_out),
            "patch_feat_0": patch_feats[0],
            "patch_feat_1": patch_feats[1],
            "patch_feat_2": patch_feats[2],
            "patch_feat_3": patch_feats[3],
        }
        if sky_mask is not None:
            dd["sky_mask"] = sky_mask
        if extrinsics is not None:
            dd["extrinsics"] = extrinsics
        if intrinsics is not None:
            dd["intrinsics"] = intrinsics
        return BackboneOutput(
            data=dd,
            input_resolution=(h, w),
            dpt_resolution=(h, w),
            patch_resolution=(ph, pw),
        )
