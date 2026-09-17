"""Metric3Dv2 metric monocular model (focal-conditioned; needs intrinsics).

Depth is predicted in a canonical camera (focal 1000 px) and de-canonicalised with the input
focal length. Pipeline per image: ``resize_to_long_side`` → keep-ratio fit into the ViT
canonical box (616 × 1064), padding with the ImageNet mean colour → ``model.inference`` →
un-pad → ``depth *= fx / 1000`` → resize back to the ``long_side`` grid. Loaded through
``torch.hub`` (``cache_dir`` becomes the hub directory's parent).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ontic_lib.camera.intrinsics import denormalize_intrinsics
from ontic_nn.wrappers.common import offline_guard

from .common import MetricDepthConfig, MetricDepthModel, MetricDepthOutput, resize_to_long_side
from .registry import register_metric_model

METRIC3D_REPO_URL = "https://github.com/YvanYin/Metric3D"
CANONICAL_FOCAL = 1000.0
VIT_INPUT_SIZE = (616, 1064)  # (H, W)
MEAN = (123.675, 116.28, 103.53)  # on 0-255 RGB; the mean doubles as the pad colour
STD = (58.395, 57.12, 57.375)
HUB_ENTRYPOINT = {
    "vit_small": "metric3d_vit_small",
    "vit_large": "metric3d_vit_large",
    "vit_giant2": "metric3d_vit_giant2",
}


@register_metric_model("metric3d")
@dataclass(kw_only=True)
class Metric3DMetricConfig(MetricDepthConfig):
    """``checkpoint_path`` is a local ``.pth`` (else the hub fetches it); ``cache_dir`` sets the
    torch hub directory to ``cache_dir/hub``."""

    variant: Literal["vit_small", "vit_large", "vit_giant2"] = "vit_large"
    hub_repo: str = "yvanyin/metric3d"
    long_side: int = 616

    def build(self) -> Metric3DMetricModel:
        return Metric3DMetricModel(self)


class Metric3DMetricModel(MetricDepthModel):
    PATCH_SIZE = 14
    REQUIRES_INTRINSICS = True

    def __init__(self, cfg: Metric3DMetricConfig) -> None:
        super().__init__()
        self.cfg = cfg
        entry = HUB_ENTRYPOINT.get(cfg.variant)
        if entry is None:
            raise ValueError(
                f"unknown Metric3D variant {cfg.variant!r}; expected {sorted(HUB_ENTRYPOINT)}"
            )
        if cfg.cache_dir:
            torch.hub.set_dir(str(Path(cfg.cache_dir).expanduser() / "hub"))
        try:
            with offline_guard(cfg.allow_download):
                hub_kwargs = dict(trust_repo=True, skip_validation=True)
                if cfg.checkpoint_path:
                    model = torch.hub.load(cfg.hub_repo, entry, pretrain=False, **hub_kwargs)
                    state = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=True)
                    sd = state.get("model_state_dict", state) if isinstance(state, dict) else state
                    model.load_state_dict(sd, strict=False)
                else:
                    model = torch.hub.load(cfg.hub_repo, entry, pretrain=True, **hub_kwargs)
        except Exception as e:
            raise RuntimeError(
                f"failed to load Metric3Dv2 via torch.hub ({type(e).__name__}: {e}); it needs "
                f"ontic-nn[metric3d] (mmengine + the mmcv 1.x shim) and the hub code + checkpoint "
                f"from {METRIC3D_REPO_URL} in the torch hub cache (allow_download / cache_dir)"
            ) from e
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, images: Tensor, intrinsics: Optional[Tensor] = None) -> MetricDepthOutput:
        if intrinsics is None:
            raise ValueError(
                "Metric3DMetricModel requires normalised intrinsics: it predicts depth in a "
                "canonical camera and needs the focal length to make it metric"
            )
        b, v = images.shape[:2]
        imgs = resize_to_long_side(images, self.PATCH_SIZE, self.cfg.long_side)
        hr, wr = imgs.shape[-2:]
        k_px = denormalize_intrinsics(intrinsics, (hr, wr))
        mean = torch.tensor(MEAN, device=images.device).view(3, 1, 1)
        std = torch.tensor(STD, device=images.device).view(3, 1, 1)

        flat = imgs.reshape(b * v, 3, hr, wr)
        flat_k = k_px.reshape(b * v, 3, 3)
        depths: List[Tensor] = [
            self._infer_one(flat[i], flat_k[i], mean, std) for i in range(b * v)
        ]
        depth = torch.stack(depths).reshape(b, v, hr, wr)
        return MetricDepthOutput(depth=depth, conf=None, intrinsics=None)

    def _infer_one(self, img: Tensor, k_px: Tensor, mean: Tensor, std: Tensor) -> Tensor:
        """``(3, Hr, Wr)`` image in ``[0, 1]`` + pixel K → metric z-depth ``(Hr, Wr)``."""
        hr, wr = img.shape[-2:]
        in_h, in_w = VIT_INPUT_SIZE
        scale = min(in_h / hr, in_w / wr)
        h2 = max(1, int(round(hr * scale)))
        w2 = max(1, int(round(wr * scale)))
        rgb = F.interpolate(img[None] * 255.0, size=(h2, w2), mode="bilinear", align_corners=False)[
            0
        ]
        fx_inner = float(k_px[0, 0]) * scale

        pad_h, pad_w = in_h - h2, in_w - w2
        pad_t, pad_l = pad_h // 2, pad_w // 2
        pad_b, pad_r = pad_h - pad_t, pad_w - pad_l
        rgb = F.pad(rgb, (pad_l, pad_r, pad_t, pad_b), mode="constant", value=0.0)
        if pad_t or pad_b or pad_l or pad_r:
            border = torch.ones_like(rgb, dtype=torch.bool)
            border[:, pad_t : pad_t + h2, pad_l : pad_l + w2] = False
            rgb = torch.where(border, mean.expand_as(rgb), rgb)

        rgb = ((rgb - mean) / std)[None]
        pred_depth, _conf, _out = self.model.inference({"input": rgb})
        pred_depth = pred_depth.squeeze()[pad_t : in_h - pad_b, pad_l : in_w - pad_r]
        pred_depth = pred_depth * (fx_inner / CANONICAL_FOCAL)
        return F.interpolate(
            pred_depth[None, None], size=(hr, wr), mode="bilinear", align_corners=False
        ).squeeze()
