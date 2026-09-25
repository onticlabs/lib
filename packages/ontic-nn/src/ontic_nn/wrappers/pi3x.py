"""Pi3X backbone wrapper.

Calls the real ``Pi3X.forward`` (conditioned on GT cameras when ``use_multimodal``) and taps
four decoder layers with forward hooks for the hierarchical patch features. Depth is the z
component of the predicted local point map; the sigmoid-logit confidence is exposed as
``1 + exp(logit)`` to match the ``expp1`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ontic_lib.camera.intrinsics import denormalize_intrinsics

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

PI3_REPO_URL = "https://github.com/yyfz/Pi3"
FEATURE_LAYERS: Tuple[int, ...] = (8, 17, 26, 35)  # of the 36 decoder blocks


@register_backbone("pi3x")
@dataclass(kw_only=True)
class Pi3XBackboneConfig(BackboneConfig):
    """``model_dir`` is the HF repo id (``PyTorchModelHubMixin``); empty → random init."""

    model_dir: str = "yyfz233/Pi3X"
    long_side: int = 518
    use_multimodal: bool = True

    def build(self) -> Pi3XBackbone:
        return Pi3XBackbone(self)


class Pi3XBackbone(BackboneBase):
    """Pi3X with ``requires_grad`` freezing.

    ``freeze_backbone`` → encoder + decoder (+ register token); ``freeze_dpt_head`` → point,
    confidence and metric heads; ``freeze_cam_dec`` → camera head; ``freeze_cam_enc`` → the
    multimodal conditioning encoders.
    """

    PATCH_SIZE = 14

    def __init__(self, cfg: Pi3XBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.pi3x = self._load(cfg)
        self.pi3x.eval()
        self._dec_dim = self.pi3x.dec_embed_dim
        self._patch_start = self.pi3x.patch_start_idx

        m = self.pi3x
        set_frozen([m.encoder, m.decoder, m.register_token], cfg.freeze_backbone)
        set_frozen(
            [
                m.point_decoder,
                m.point_head,
                m.conf_decoder,
                m.conf_head,
                m.metric_decoder,
                m.metric_head,
                m.metric_token,
            ],
            cfg.freeze_dpt_head,
        )
        set_frozen([m.camera_decoder, m.camera_head], cfg.freeze_cam_dec)
        if cfg.use_multimodal:
            encoders = [
                getattr(m, a, None)
                for a in ("depth_encoder", "depth_emb", "ray_embed", "pose_inject_blk")
            ]
            set_frozen([e for e in encoders if e is not None], cfg.freeze_cam_enc)

    @staticmethod
    def _load(cfg: Pi3XBackboneConfig):
        pi3x_mod = import_research_module(
            "pi3.models.pi3x", extra="pi3x", repo_url=PI3_REPO_URL, what="Pi3XBackbone"
        )
        path = resolve_checkpoint(
            cfg.checkpoint_path,
            cfg.model_dir,
            cache_dir=cfg.cache_dir,
            allow_download=cfg.allow_download,
            extra="pi3x",
            what="Pi3X snapshot",
        )
        if path is None:
            model = pi3x_mod.Pi3X(use_multimodal=cfg.use_multimodal)
        else:
            model = pi3x_mod.Pi3X.from_pretrained(path)
        if not cfg.use_multimodal and getattr(model, "use_multimodal", False):
            model.disable_multimodal()
        return model

    @property
    def accepts_gt_cameras(self) -> bool:
        return bool(self.cfg.use_multimodal)

    @property
    def encoder_dim(self) -> int:
        return self._dec_dim

    @staticmethod
    def intrinsics_from_rays(rays: Tensor, h: int, w: int) -> Tensor:
        """Normalised pinhole intrinsics ``(B, V, 3, 3)`` least-squares fitted to unit rays
        ``(B, V, H, W, 3)``: ``ray ∝ [(u - cx) / fx, (v - cy) / fy, 1]`` is linear in the pixel
        coordinates, so ``fx, cx`` / ``fy, cy`` come from a line fit along each axis."""
        b, n = rays.shape[:2]
        dev, dt = rays.device, rays.dtype
        xy = rays[..., :2] / (rays[..., 2:3] + 1e-6)
        u = torch.arange(w, device=dev, dtype=dt)
        v = torch.arange(h, device=dev, dtype=dt)

        def fit(prof: Tensor, coord: Tensor) -> Tuple[Tensor, Tensor]:
            cm = coord.mean()
            pm = prof.mean(-1, keepdim=True)
            a = ((coord - cm) * (prof - pm)).sum(-1) / (((coord - cm) ** 2).sum() + 1e-8)
            return a, pm.squeeze(-1) - a * cm

        ax, bx = fit(xy[..., 0].mean(dim=2), u)
        ay, by = fit(xy[..., 1].mean(dim=3), v)
        fx = 1.0 / (ax + 1e-8)
        fy = 1.0 / (ay + 1e-8)
        k = torch.zeros(b, n, 3, 3, device=dev, dtype=dt)
        k[..., 0, 0] = fx / w
        k[..., 1, 1] = fy / h
        k[..., 0, 2] = -bx * fx / w
        k[..., 1, 2] = -by * fy / h
        k[..., 2, 2] = 1.0
        return k

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

        cond_intr = cond_poses = None
        with_prior = False
        if extrinsics is not None and intrinsics is not None and self.cfg.use_multimodal:
            cond_intr = denormalize_intrinsics(intrinsics, (h, w))
            cond_poses = extrinsics
            with_prior = True

        captured: Dict[int, Tensor] = {}
        handles = [
            self.pi3x.decoder[i].register_forward_hook(self._make_hook(captured, i))
            for i in FEATURE_LAYERS
        ]
        cfg = self.cfg
        any_trainable = not (
            cfg.freeze_backbone
            and cfg.freeze_dpt_head
            and cfg.freeze_cam_dec
            and cfg.freeze_cam_enc
        )
        try:
            with (
                torch.autocast(images.device.type, dtype=amp_dtype()),
                torch.set_grad_enabled(any_trainable),
            ):
                out = self.pi3x(
                    images, intrinsics=cond_intr, poses=cond_poses, with_prior=with_prior
                )
        finally:
            for hd in handles:
                hd.remove()

        hw = ph * pw + self._patch_start
        raw_feats: List[Tensor] = []
        for i in FEATURE_LAYERS:
            f = captured[i].reshape(b, v, hw, self._dec_dim)[:, :, self._patch_start :]
            raw_feats.append(f.reshape(b, v, ph, pw, self._dec_dim).float())

        pred_depth = out["local_points"][..., 2].float()
        depth_conf = 1.0 + torch.exp(out["conf"][..., 0].float().clamp(max=15.0))
        dd: BackboneOutput.DataDict = {
            "depth": pred_depth,
            "depth_conf": depth_conf,
            "patch_feat_0": raw_feats[0],
            "patch_feat_1": raw_feats[1],
            "patch_feat_2": raw_feats[2],
            "patch_feat_3": raw_feats[3],
            "extrinsics_pred": out["camera_poses"].float(),
            "intrinsics_pred": self.intrinsics_from_rays(out["rays"].float(), h, w),
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
    def _make_hook(store: Dict[int, Tensor], idx: int):
        def hook(_module, _inp, output):
            store[idx] = output if output.requires_grad else output.detach()

        return hook
