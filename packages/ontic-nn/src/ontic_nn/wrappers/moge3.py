"""MoGe-3 backbone wrapper (monocular; views are folded into the batch).

MoGe returns an affine point map per image plus a validity mask and a metric scale;
``recover_focal_shift`` turns it into depth and predicted intrinsics. There is no
``extrinsics_pred`` — a single image has no shared world frame. ``depth_conf = 1 + mask`` and
``sky_mask = 1 - mask``. The ViT grid is pinned to ``num_tokens = Ph * Pw`` of the
``long_side``-resized image, so patch features match the output resolution exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .common import (
    BackboneBase,
    BackboneConfig,
    BackboneOutput,
    amp_dtype,
    import_research_module,
    resize_to_long_side,
    resolve_checkpoint,
    set_frozen,
)
from .registry import register_backbone

MOGE_REPO_URL = "https://github.com/microsoft/MoGe"


def install_triton_autotuner_shim() -> None:
    """Let FlexGEMM (a MoGe-3 import-time dependency) import on Triton < 3.1.

    ``flex_gemm`` passes ``do_bench`` to ``triton.runtime.autotuner.Autotuner``, which only
    accepts it from Triton 3.1; the argument is dropped on older versions. Idempotent.
    """
    import inspect

    try:
        import triton.runtime.autotuner as autotuner
    except ImportError:
        return
    init = autotuner.Autotuner.__init__
    if getattr(init, "_ontic_shimmed", False):
        return
    params = inspect.signature(init).parameters
    if "do_bench" in params:
        return
    n_accepted = len(params) - 1

    def patched(self, *args, **kwargs):
        kwargs.pop("do_bench", None)
        return init(self, *args[:n_accepted], **kwargs)

    patched._ontic_shimmed = True
    autotuner.Autotuner.__init__ = patched


@register_backbone("moge3")
@dataclass(kw_only=True)
class MoGe3BackboneConfig(BackboneConfig):
    """``model_dir`` is the HF repo holding ``model.pt``; empty → random init from the
    ``model_config`` JSON (a MoGe train-config ``model`` block).

    ``refine_steps > 0`` builds the sparse volumetric refiner (needs a working FlexGEMM);
    ``dino_layers`` picks the DINOv2 blocks tapped for the patch features (``None`` → MoGe's
    quartile taps).
    """

    model_dir: str = "Ruicheng/moge-3-vitl"
    long_side: int = 966  # a 4:3 frame lands at ~3600 tokens, MoGe's resolution_level 9
    model_config: str = ""
    refine_steps: int = 0
    dino_layers: Optional[Tuple[int, ...]] = None
    use_amp: bool = False

    def build(self) -> MoGe3Backbone:
        return MoGe3Backbone(self)


class MoGe3Backbone(BackboneBase):
    """MoGe-3 with ``requires_grad`` freezing: encoder (``freeze_backbone``), neck + heads +
    refiner (``freeze_dpt_head``); the camera flags do not apply."""

    PATCH_SIZE = 14

    def __init__(self, cfg: MoGe3BackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.moge, self.loaded_checkpoint = self._load(cfg)
        self.moge.eval()
        set_frozen([self.moge.encoder], cfg.freeze_backbone)
        heads = [self.moge.neck] + [
            getattr(self.moge, name)
            for name in ("points_head", "mask_head", "normal_head", "scale_head", "refiner")
            if getattr(self.moge, name, None) is not None
        ]
        set_frozen(heads, cfg.freeze_dpt_head)

    @staticmethod
    def _load(cfg: MoGe3BackboneConfig):
        install_triton_autotuner_shim()
        v3 = import_research_module(
            "moge.model.v3", extra="moge3", repo_url=MOGE_REPO_URL, what="MoGe3Backbone"
        )
        model_kwargs = None if cfg.refine_steps > 0 else {"refiner": None}
        path = resolve_checkpoint(
            cfg.checkpoint_path,
            cfg.model_dir,
            filename="model.pt",
            cache_dir=cfg.cache_dir,
            allow_download=cfg.allow_download,
            extra="moge3",
            what="MoGe-3 checkpoint",
        )
        if path is not None:
            return v3.MoGeModel.from_pretrained(path, model_kwargs=model_kwargs), path
        if not cfg.model_config:
            raise ValueError(
                "MoGe3Backbone: no checkpoint (model_dir / checkpoint_path) and no model_config"
            )
        arch = json.loads(Path(cfg.model_config).read_text())
        arch = arch.get("model", arch)
        if model_kwargs:
            arch = {**arch, **model_kwargs}
        return v3.MoGeModel(**arch), None

    @property
    def encoder_dim(self) -> int:
        return int(self.moge.encoder.dim_features)

    def _dino_layers(self) -> List[int]:
        if self.cfg.dino_layers is not None:
            return list(self.cfg.dino_layers)
        n = len(self.moge.encoder.backbone.blocks)
        return [n // 4 - 1, n // 2 - 1, (3 * n) // 4 - 1, n - 1]

    def forward(
        self,
        images: Tensor,
        extrinsics: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
    ) -> BackboneOutput:
        utils3d = import_research_module(
            "utils3d_moge", extra="moge3", repo_url=MOGE_REPO_URL, what="MoGe3Backbone"
        )
        geometry = import_research_module(
            "moge.utils.geometry_torch", extra="moge3", repo_url=MOGE_REPO_URL, what="MoGe3Backbone"
        )
        images = resize_to_long_side(images, self.PATCH_SIZE, self.cfg.long_side)
        b, v, c, h, w = images.shape
        ph, pw = h // self.PATCH_SIZE, w // self.PATCH_SIZE
        npatch = ph * pw
        aspect_ratio = w / h
        flat = images.reshape(b * v, c, h, w)

        dino_idx = self._dino_layers()
        vit = self.moge.encoder.backbone
        states: Dict[int, Tensor] = {}
        handles = [
            vit.blocks[j].register_forward_hook(self._make_store_hook(states, j))
            for j in set(dino_idx)
        ]
        any_trainable = not (self.cfg.freeze_backbone and self.cfg.freeze_dpt_head)
        try:
            with (
                torch.autocast(images.device.type, dtype=amp_dtype(), enabled=self.cfg.use_amp),
                torch.set_grad_enabled(any_trainable),
            ):
                out = self.moge(flat, num_tokens=npatch, refine_steps=self.cfg.refine_steps)
        finally:
            for hd in handles:
                hd.remove()

        points = out["points"].float()  # (B*V, H, W, 3) affine point map
        mask = out.get("mask", None)  # (B*V, H, W) validity probability
        metric_scale = out.get("metric_scale", None)  # (B*V,)
        mask_binary = None if mask is None else mask.float() > 0.5

        focal, shift = geometry.recover_focal_shift(points, mask_binary)
        half_diag = (1 + aspect_ratio**2) ** 0.5
        fx, fy = focal / 2 * half_diag / aspect_ratio, focal / 2 * half_diag
        intrinsics_pred = utils3d.pt.intrinsics_from_focal_center(fx, fy, 0.5, 0.5).float()

        depth_out = points[..., 2] + shift[:, None, None]
        if metric_scale is not None:
            depth_out = depth_out * metric_scale.float()[:, None, None]
        depth_out = torch.nan_to_num(depth_out, nan=0.0, posinf=0.0, neginf=0.0)
        depth_out = depth_out.reshape(b, v, h, w)
        valid = torch.ones_like(depth_out) if mask is None else mask.float().reshape(b, v, h, w)

        def patch(state: Tensor) -> Tensor:  # (B*V, 1 + registers + npatch, C)
            return vit.norm(state[:, -npatch:]).reshape(b, v, ph, pw, self.encoder_dim).float()

        feats = [patch(states[j]) for j in dino_idx]
        dd: BackboneOutput.DataDict = {
            "depth": depth_out,
            "depth_conf": 1.0 + valid,
            "sky_mask": 1.0 - valid,
            "patch_feat_0": feats[0],
            "patch_feat_1": feats[1],
            "patch_feat_2": feats[2],
            "patch_feat_3": feats[3],
            "intrinsics_pred": intrinsics_pred.reshape(b, v, 3, 3),
        }
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

    @staticmethod
    def _make_store_hook(store: Dict[int, Tensor], idx: int):
        def hook(_module, _inp, output):
            store[idx] = output

        return hook
