"""Depth Anything 3 backbone wrapper.

Runs the upstream ``DepthAnything3`` ViT (with camera-token conditioning when GT cameras are
given), our :class:`ontic_nn.dpt.DPTHead` / :class:`DualDPTHead` loaded from the DA3 head
weights, and DA3's camera decoder. On the nested checkpoint only the anyview branch runs:
depth is raw (up to scale), the metric branch is the job of ``ontic_nn.metric_depth.da3``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from ontic_lib.camera.intrinsics import denormalize_intrinsics
from ontic_lib.transforms.rigid import invert_rigid_transform
from ontic_nn.dpt import DPTHead, DualDPTHead

from .common import (
    BackboneBase,
    BackboneConfig,
    BackboneOutput,
    as_homogeneous_4x4,
    buffers_to_params,
    convert_to_buffer,
    extract_weights,
    import_research_module,
    normalize_intrinsics_pixels,
    resize_to_long_side,
    resolve_checkpoint,
)
from .registry import register_backbone

DA3_REPO_URL = "https://github.com/ByteDance-Seed/Depth-Anything-3"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@register_backbone("da3")
@dataclass(kw_only=True)
class DA3BackboneConfig(BackboneConfig):
    """``model_dir`` is the HF repo id; empty → random init of the ``model_name`` preset."""

    model_name: str = "da3-large"
    model_dir: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    long_side: int = 504

    def build(self) -> DA3Backbone:
        return DA3Backbone(self)


def load_da3(
    model_dir: str,
    model_name: str,
    *,
    checkpoint_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_download: bool = True,
):
    """Upstream ``DepthAnything3`` from a local snapshot / the hub, or the ``model_name`` preset."""
    os.environ.setdefault("DA3_LOG_LEVEL", "WARN")
    api = import_research_module(
        "depth_anything_3.api", extra="da3", repo_url=DA3_REPO_URL, what="DA3"
    )
    path = resolve_checkpoint(
        checkpoint_path,
        model_dir or None,
        cache_dir=cache_dir,
        allow_download=allow_download,
        extra="da3",
        what="DA3 snapshot",
    )
    if path is None:
        return api.DepthAnything3(model_name=model_name)
    return api.DepthAnything3.from_pretrained(path)


def normalize_extrinsics(w2c: Tensor) -> Tensor:
    """DA3's pose normalisation for ``(B, V, 4, 4)`` w2c: first camera at the origin, median
    camera distance one."""
    transform = invert_rigid_transform(w2c[:, :1])
    w2c_norm = w2c @ transform
    c2w = invert_rigid_transform(w2c_norm)
    dists = c2w[..., :3, 3].norm(dim=-1)
    median = torch.median(dists, dim=-1, keepdim=True).values.clamp(min=1e-1)
    w2c_norm[..., :3, 3] = w2c_norm[..., :3, 3] / median.unsqueeze(-1)
    return w2c_norm


class DA3Backbone(BackboneBase):
    """Frozen DA3 (parameters become buffers) with selectively trainable head / camera modules."""

    PATCH_SIZE = 14
    ACCEPTS_GT_CAMERAS = True

    def __init__(self, cfg: DA3BackboneConfig) -> None:
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

        vit_dim = self._vit_dim()
        self._is_nested = hasattr(self.da3.model, "da3")

        # Our head is built from DA3's state dict before the (non-persistent) buffer conversion.
        self.dpt_head = self._build_head(vit_dim)
        convert_to_buffer(self.da3, persistent=False)
        if cfg.freeze_dpt_head:
            convert_to_buffer(self.dpt_head, persistent=False)

        net = self._net()
        if not cfg.freeze_cam_dec and net.cam_dec is not None:
            buffers_to_params(net.cam_dec)
        if not cfg.freeze_cam_enc and net.cam_enc is not None:
            buffers_to_params(net.cam_enc)

        self.register_buffer(
            "image_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "image_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    # -- introspection -----------------------------------------------------------------
    def _net(self):
        """Inner ``DepthAnything3Net`` (the anyview branch on the nested model)."""
        net = self.da3.model
        return net.da3 if hasattr(net, "da3") else net

    def _vit_dim(self) -> int:
        backbone = self._net().backbone
        dim = backbone.pretrained.embed_dim
        return dim * 2 if getattr(backbone, "cat_token", False) else dim

    def _build_head(self, vit_dim: int) -> nn.Module:
        da3_head = self._net().head
        is_dual = hasattr(da3_head, "scratch") and hasattr(da3_head.scratch, "refinenet1_aux")
        output_dim = list(da3_head.scratch.output_conv2.children())[-1].out_channels
        common = dict(
            dim_in=vit_dim,
            patch_size=self.PATCH_SIZE,
            output_dim=output_dim,
            activation=getattr(da3_head, "activation", "exp"),
            conf_activation=getattr(da3_head, "conf_activation", "expp1"),
            features=da3_head.scratch.output_conv1.in_channels,
            out_channels=tuple(p.out_channels for p in da3_head.projects),
            down_ratio=getattr(da3_head, "down_ratio", 1),
        )
        if is_dual:
            head: nn.Module = DualDPTHead(
                pos_embed=getattr(da3_head, "pos_embed", True),
                aux_pyramid_levels=getattr(da3_head, "aux_levels", 4),
                aux_out1_conv_num=getattr(da3_head, "aux_out1_conv_num", 5),
                **common,
            )
        else:
            head = DPTHead(
                pos_embed=getattr(da3_head, "pos_embed", False),
                use_sky_head=hasattr(da3_head.scratch, "sky_output_conv2"),
                **common,
            )
        prefix = "model.da3.head." if self._is_nested else "model.head."
        missing, _unexpected = head.load_state_dict(
            extract_weights(self.da3.state_dict(), prefix), strict=False
        )
        if missing:
            raise RuntimeError(f"DA3 head weights missing for {sorted(missing)[:5]}...")
        return head

    @property
    def encoder_dim(self) -> int:
        return self._vit_dim()

    # -- pieces of the upstream forward ------------------------------------------------
    def _run_backbone(
        self, imgs_norm: Tensor, w2c_norm: Optional[Tensor], k_px: Optional[Tensor]
    ) -> List[Tuple[Tensor, Tensor]]:
        """ViT with camera-token injection → list of ``(patch_tokens, cam_token)`` per tap.

        A trainable camera encoder records a graph only while autograd is already on: an
        outer ``no_grad`` (validation) is never overridden.
        """
        net = self._net()
        h, w = imgs_norm.shape[-2:]
        cam_token = None
        if w2c_norm is not None and net.cam_enc is not None:
            with (
                torch.autocast(device_type=imgs_norm.device.type, enabled=False),
                torch.set_grad_enabled(not self.cfg.freeze_cam_enc and torch.is_grad_enabled()),
            ):
                cam_token = net.cam_enc(w2c_norm, k_px, (h, w))
        with torch.inference_mode(self.cfg.freeze_backbone):
            feats, _aux = net.backbone(
                imgs_norm,
                cam_token=cam_token,
                export_feat_layers=[],
                ref_view_strategy="saddle_balanced",
            )
        if self.cfg.freeze_backbone:  # leave inference mode so trainable heads can use them
            feats = [(f[0].clone(), f[1].clone()) for f in feats]
        return feats

    def _run_heads(
        self, feats: List[Tuple[Tensor, Tensor]], h: int, w: int
    ) -> Tuple[Dict[str, Tensor], Optional[Tensor], Optional[Tensor]]:
        """Our DPT head + DA3's camera decoder → ``(head outputs, c2w_pred, K_px_pred)``."""
        net = self._net()
        c2w = k_px = None
        with torch.autocast(device_type=feats[0][0].device.type, enabled=False):
            with torch.inference_mode(self.cfg.freeze_dpt_head):
                output = self.dpt_head([f[0] for f in feats], h, w, patch_start_idx=0)
            if net.cam_dec is not None:
                transform = import_research_module(
                    "depth_anything_3.model.utils.transform",
                    extra="da3",
                    repo_url=DA3_REPO_URL,
                    what="DA3",
                )
                with torch.inference_mode(self.cfg.freeze_cam_dec):
                    pose_enc = net.cam_dec(feats[-1][1])
                    c2w, k_px = transform.pose_encoding_to_extri_intri(pose_enc, (h, w))
        output = self._fill_sky_depth(output)
        return output, c2w, k_px

    @staticmethod
    def _fill_sky_depth(output: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """DA3's mono sky handling: sky pixels get the 99th-percentile non-sky depth."""
        if "sky" not in output:
            return output
        alignment = import_research_module(
            "depth_anything_3.utils.alignment", extra="da3", repo_url=DA3_REPO_URL, what="DA3"
        )
        non_sky = alignment.compute_sky_mask(output["sky"], threshold=0.3)
        if non_sky.sum() <= 10 or (~non_sky).sum() <= 10:
            return output
        non_sky_depth = output["depth"][non_sky]
        if non_sky_depth.numel() > 100000:
            idx = torch.randint(0, non_sky_depth.numel(), (100000,), device=non_sky_depth.device)
            non_sky_depth = non_sky_depth[idx]
        non_sky_max = torch.quantile(non_sky_depth, 0.99)
        output["depth"], _ = alignment.set_sky_regions_to_max_depth(
            output["depth"], None, non_sky, max_depth=non_sky_max
        )
        return output

    # -- forward -----------------------------------------------------------------------
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
        imgs_norm = (images - self.image_mean) / self.image_std

        w2c_norm = k_px = None
        if extrinsics is not None and intrinsics is not None:
            w2c_norm = normalize_extrinsics(invert_rigid_transform(extrinsics))
            k_px = denormalize_intrinsics(intrinsics, (h, w))

        feats = self._run_backbone(imgs_norm, w2c_norm, k_px)
        outputs, c2w_pred, k_px_pred = self._run_heads(feats, h, w)

        pred_depth = outputs["depth"]
        hd, wd = pred_depth.shape[-2:]

        raw_feats: List[Tensor] = []
        for patch_tok, _cam in feats:  # (B, V, tokens, C); drop leading non-patch tokens
            if patch_tok.shape[2] > ph * pw:
                patch_tok = patch_tok[:, :, -ph * pw :]
            raw_feats.append(patch_tok.reshape(b, v, ph, pw, patch_tok.shape[-1]))

        dd: BackboneOutput.DataDict = {
            "depth": pred_depth,
            "depth_conf": outputs["depth_conf"],
            "patch_feat_0": raw_feats[0],
            "patch_feat_1": raw_feats[1],
            "patch_feat_2": raw_feats[2],
            "patch_feat_3": raw_feats[3],
        }
        if "sky" in outputs:
            dd["sky_mask"] = outputs["sky"]
        if extrinsics is not None:
            dd["extrinsics"] = extrinsics
        if intrinsics is not None:
            dd["intrinsics"] = intrinsics
        if c2w_pred is not None:
            dd["extrinsics_pred"] = as_homogeneous_4x4(c2w_pred)
        if k_px_pred is not None:
            dd["intrinsics_pred"] = normalize_intrinsics_pixels(k_px_pred, hd, wd)

        return BackboneOutput(
            data=dd,
            input_resolution=(h, w),
            dpt_resolution=(hd, wd),
            patch_resolution=(ph, pw),
        )
