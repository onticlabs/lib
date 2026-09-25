"""VGGT-Omega backbone wrapper (pose-free; GT cameras are passed through)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from ontic_lib.transforms.rigid import invert_rigid_transform

from .common import (
    BackboneBase,
    BackboneConfig,
    BackboneOutput,
    amp_dtype,
    as_homogeneous_4x4,
    import_research_module,
    normalize_intrinsics_pixels,
    resize_to_long_side,
    resolve_checkpoint,
    set_frozen,
)
from .registry import register_backbone

VGGT_REPO_URL = "https://github.com/facebookresearch/vggt-omega"


@register_backbone("vggt")
@dataclass(kw_only=True)
class VGGTBackboneConfig(BackboneConfig):
    """``model_dir`` is the (gated) HF repo holding ``checkpoint_file``; empty → random init.

    ``normalize_features`` layer-norms the exposed aggregator tokens (their raw magnitude
    grows to ~±170 by the last layer). ``depth_scale_mul`` multiplies the raw depth.
    """

    model_dir: str = "facebook/VGGT-Omega"
    checkpoint_file: str = "vggt_omega_1b_512.pt"
    long_side: int = 512
    embed_dim: int = 1024
    enable_alignment: bool = False
    normalize_features: bool = True
    depth_scale_mul: float = 1.0

    def build(self) -> VGGTBackbone:
        return VGGTBackbone(self)


class VGGTBackbone(BackboneBase):
    """VGGT-Omega with ``requires_grad`` freezing: aggregator / dense head / camera head."""

    PATCH_SIZE = 16

    def __init__(self, cfg: VGGTBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.vggt = self._load(cfg)
        self.vggt.eval()
        set_frozen([self.vggt.aggregator], cfg.freeze_backbone)
        if self.vggt.dense_head is not None:
            set_frozen([self.vggt.dense_head], cfg.freeze_dpt_head)
        if self.vggt.camera_head is not None:
            set_frozen([self.vggt.camera_head], cfg.freeze_cam_dec)

    @staticmethod
    def _load(cfg: VGGTBackboneConfig):
        vggt_omega = import_research_module(
            "vggt_omega", extra="vggt", repo_url=VGGT_REPO_URL, what="VGGTBackbone"
        )
        model = vggt_omega.VGGTOmega(
            patch_size=VGGTBackbone.PATCH_SIZE,
            embed_dim=cfg.embed_dim,
            enable_camera=True,
            enable_depth=True,
            enable_alignment=cfg.enable_alignment,
        )
        path = resolve_checkpoint(
            cfg.checkpoint_path,
            cfg.model_dir,
            filename=cfg.checkpoint_file,
            cache_dir=cfg.cache_dir,
            allow_download=cfg.allow_download,
            extra="vggt",
            what="VGGT-Omega checkpoint",
        )
        if path is None:
            return model
        sd = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(sd, dict):
            for key in ("model", "state_dict", "weights"):
                if key in sd and isinstance(sd[key], dict):
                    sd = sd[key]
                    break
        model.load_state_dict(sd, strict=False)
        return model

    @property
    def encoder_dim(self) -> int:
        return 2 * self.cfg.embed_dim

    def train(self, mode: bool = True) -> VGGTBackbone:
        super().train(mode)
        if self.cfg.freeze_backbone:
            self.vggt.aggregator.eval()
        if self.cfg.freeze_dpt_head and self.vggt.dense_head is not None:
            self.vggt.dense_head.eval()
        if self.cfg.freeze_cam_dec and self.vggt.camera_head is not None:
            self.vggt.camera_head.eval()
        return self

    def forward(
        self,
        images: Tensor,
        extrinsics: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
    ) -> BackboneOutput:
        pose_enc_mod = import_research_module(
            "vggt_omega.utils.pose_enc", extra="vggt", repo_url=VGGT_REPO_URL, what="VGGTBackbone"
        )
        images = resize_to_long_side(images, self.PATCH_SIZE, self.cfg.long_side)
        b, v, _c, h, w = images.shape
        ph, pw = h // self.PATCH_SIZE, w // self.PATCH_SIZE
        device_type = images.device.type

        with (
            torch.autocast(device_type, dtype=amp_dtype()),
            torch.set_grad_enabled(not self.cfg.freeze_backbone),
        ):
            tokens_list, patch_token_start = self.vggt.aggregator(images)

        cached = [t for t in tokens_list if t is not None]  # the 4 cached aggregator layers
        if len(cached) != 4:
            raise RuntimeError(f"expected 4 cached aggregator layers, got {len(cached)}")
        raw_feats: List[Tensor] = []
        for t in cached:
            patch_tok = t[:, :, patch_token_start:]  # (B, V, Ph*Pw, 2C)
            feat = patch_tok.reshape(b, v, ph, pw, patch_tok.shape[-1]).float()
            if self.cfg.normalize_features:
                feat = F.layer_norm(feat, (feat.shape[-1],))
            raw_feats.append(feat)

        with (
            torch.autocast(device_type, enabled=False),
            torch.set_grad_enabled(not self.cfg.freeze_dpt_head),
        ):
            pred_depth, depth_conf = self.vggt.dense_head(
                tokens_list, images=images, patch_token_start=patch_token_start
            )
        pred_depth = pred_depth.squeeze(-1)  # (B, V, H, W)

        with (
            torch.autocast(device_type, enabled=False),
            torch.set_grad_enabled(not self.cfg.freeze_cam_dec),
        ):
            pose_enc = self.vggt.camera_head(tokens_list, patch_token_start=patch_token_start)
        w2c, k_px = pose_enc_mod.encoding_to_camera(pose_enc, image_size_hw=(h, w))
        extrinsics_pred = invert_rigid_transform(as_homogeneous_4x4(w2c))
        intrinsics_pred = normalize_intrinsics_pixels(k_px, h, w)

        if self.cfg.depth_scale_mul != 1.0:
            pred_depth = pred_depth * self.cfg.depth_scale_mul

        dd: BackboneOutput.DataDict = {
            "depth": pred_depth,
            "depth_conf": depth_conf,
            "patch_feat_0": raw_feats[0],
            "patch_feat_1": raw_feats[1],
            "patch_feat_2": raw_feats[2],
            "patch_feat_3": raw_feats[3],
            "extrinsics_pred": extrinsics_pred,
            "intrinsics_pred": intrinsics_pred,
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
