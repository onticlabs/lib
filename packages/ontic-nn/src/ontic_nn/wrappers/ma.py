"""MapAnything backbone wrapper.

Runs the upstream ``MapAnything`` model decomposed into encoder → geometric-input fusion →
info-sharing transformer → dense / pose / scale heads, so the four hierarchical features
(encoder output, two intermediate transformer outputs, final output) can be tapped. Depth is
the z component of the predicted camera-frame point map.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor

from ontic_lib.camera.intrinsics import denormalize_intrinsics

from .common import (
    BackboneBase,
    BackboneConfig,
    BackboneOutput,
    buffers_to_params,
    convert_to_buffer,
    import_research_module,
    normalize_intrinsics_pixels,
    resize_to_long_side,
    resolve_checkpoint,
)
from .registry import register_backbone

MA_REPO_URL = "https://github.com/facebookresearch/map-anything"

_GEOMETRIC_INPUTS_WITH_CAMERAS = {
    "overall_prob": 1.0,
    "dropout_prob": 0.0,
    "ray_dirs_prob": 1.0,
    "depth_prob": 0.0,
    "cam_prob": 1.0,
    "sparse_depth_prob": 0.0,
    # 0.0 keeps the metric-scale information (the upstream ``infer`` setting with metric cameras).
    "depth_scale_norm_all_prob": 0.0,
    "pose_scale_norm_all_prob": 0.0,
}
_GEOMETRIC_INPUTS_POSE_FREE = {
    "overall_prob": 0.0,
    "dropout_prob": 1.0,
    "ray_dirs_prob": 0.0,
    "depth_prob": 0.0,
    "cam_prob": 0.0,
    "sparse_depth_prob": 0.0,
    "depth_scale_norm_all_prob": 0.0,
    "pose_scale_norm_all_prob": 0.0,
}


@register_backbone("ma")
@dataclass(kw_only=True)
class MABackboneConfig(BackboneConfig):
    model_dir: str = "facebook/map-anything"
    long_side: int = 518

    def build(self) -> MABackbone:
        return MABackbone(self)


class MABackbone(BackboneBase):
    """Frozen MapAnything (parameters become buffers) with selectively trainable heads.

    ``freeze_dpt_head`` → dense head, ``freeze_cam_dec`` → pose + scale heads,
    ``freeze_cam_enc`` → the geometric-input encoders and fusion norm.
    """

    PATCH_SIZE = 14
    ACCEPTS_GT_CAMERAS = True

    def __init__(self, cfg: MABackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ma = self._load(cfg)
        self.ma.eval()

        self._encoder_dim = self.ma.encoder.enc_embed_dim
        self._use_encoder_features_for_dpt = getattr(self.ma, "use_encoder_features_for_dpt", True)
        self._scene_rep_type = self.ma.scene_rep_type

        convert_to_buffer(self.ma, persistent=False)
        if not cfg.freeze_dpt_head:
            buffers_to_params(self.ma.dense_head)
        if not cfg.freeze_cam_dec:
            for attr in ("pose_head", "scale_head"):
                if hasattr(self.ma, attr):
                    buffers_to_params(getattr(self.ma, attr))
        if not cfg.freeze_cam_enc:
            for attr in (
                "ray_dirs_encoder",
                "cam_rot_encoder",
                "cam_trans_encoder",
                "cam_trans_scale_encoder",
                "depth_scale_encoder",
                "depth_encoder",
                "fusion_norm_layer",
            ):
                mod = getattr(self.ma, attr, None)
                if mod is not None:
                    buffers_to_params(mod)

    @staticmethod
    def _load(cfg: MABackboneConfig):
        """``MapAnything.from_pretrained`` without downloading the DINOv2 weights it overwrites."""
        model_mod = import_research_module(
            "mapanything.models.mapanything.model",
            extra="ma",
            repo_url=MA_REPO_URL,
            what="MABackbone",
        )
        path = resolve_checkpoint(
            cfg.checkpoint_path,
            cfg.model_dir,
            cache_dir=cfg.cache_dir,
            allow_download=cfg.allow_download,
            extra="ma",
            what="MapAnything snapshot",
        )
        if path is None:
            raise ValueError("MABackbone needs a checkpoint: set model_dir or checkpoint_path")

        orig_hub_load = torch.hub.load

        @functools.wraps(orig_hub_load)
        def hub_load_no_weights(repo, model, *args, **kwargs):
            if "dinov2" in str(repo).lower():
                kwargs["pretrained"] = False
            return orig_hub_load(repo, model, *args, **kwargs)

        torch.hub.load = hub_load_no_weights
        try:
            return model_mod.MapAnything.from_pretrained(path)
        finally:
            torch.hub.load = orig_hub_load

    @property
    def encoder_dim(self) -> int:
        return self._encoder_dim

    # -- upstream input / output conversion -------------------------------------------
    def _prepare_views(
        self,
        images: Tensor,
        extrinsics: Optional[Tensor],
        intrinsics: Optional[Tensor],
    ) -> List[dict]:
        """``(B, V, 3, H, W)`` → MapAnything's one-dict-per-view list (``img (B, 3, H, W)``)."""
        inference = import_research_module(
            "mapanything.utils.inference", extra="ma", repo_url=MA_REPO_URL, what="MABackbone"
        )
        b, v, _c, h, w = images.shape
        data_norm_type = self.ma.encoder.data_norm_type
        has_cameras = extrinsics is not None and intrinsics is not None
        views = []
        for i in range(v):
            view = {"img": images[:, i], "data_norm_type": [data_norm_type] * b}
            if has_cameras:
                view["camera_poses"] = extrinsics[:, i]  # c2w 4x4
                view["intrinsics"] = denormalize_intrinsics(intrinsics[:, i], (h, w))
                view["is_metric_scale"] = torch.ones(b, dtype=torch.bool, device=images.device)
            views.append(view)
        if has_cameras:
            views = inference.preprocess_input_views_for_inference(views)
        return views

    def _assemble_scene_outputs(
        self, dense_out, pose_out, scale: Tensor, num_views: int
    ) -> List[dict]:
        """Raw head outputs → per-view dicts (mirrors the upstream ``MapAnything.forward``)."""
        geometry = import_research_module(
            "mapanything.utils.geometry", extra="ma", repo_url=MA_REPO_URL, what="MABackbone"
        )
        scene_rep = self._scene_rep_type
        dense = dense_out.value.permute(0, 2, 3, 1).contiguous()
        if "pointmap+raydirs+depth+pose" in scene_rep:
            _pts_direct, ray_dirs, depth_along_ray = dense.split([3, 3, 1], dim=-1)
        elif "raydirs+depth+pose" in scene_rep:
            ray_dirs, depth_along_ray = dense.split([3, 1], dim=-1)
        elif "campointmap+pose" in scene_rep:
            depth_along_ray = torch.norm(dense, dim=-1, keepdim=True)
            ray_dirs = dense / (depth_along_ray + 1e-8)
        elif "pointmap" in scene_rep and "raydirs" not in scene_rep:
            res = [
                {"pts3d": p * scale.unsqueeze(-1).unsqueeze(-1), "metric_scaling_factor": scale}
                for p in dense.chunk(num_views, dim=0)
            ]
            return self._add_conf_and_mask(res, dense_out, num_views)
        else:
            raise ValueError(f"unsupported MapAnything scene_rep_type {scene_rep!r}")

        cam_trans, cam_quats = pose_out.value.split([3, 4], dim=-1)
        pts3d = geometry.convert_ray_dirs_depth_along_ray_pose_trans_quats_to_pointmap(
            ray_dirs, depth_along_ray, cam_trans, cam_quats
        )
        pts3d_cam = ray_dirs * depth_along_ray
        per_view = {
            k: t.chunk(num_views, dim=0)
            for k, t in {
                "ray_directions": ray_dirs,
                "depth_along_ray": depth_along_ray,
                "cam_trans": cam_trans,
                "cam_quats": cam_quats,
                "pts3d": pts3d,
                "pts3d_cam": pts3d_cam,
            }.items()
        }
        s_map = scale.unsqueeze(-1).unsqueeze(-1)
        res = [
            {
                "pts3d": per_view["pts3d"][i] * s_map,
                "pts3d_cam": per_view["pts3d_cam"][i] * s_map,
                "ray_directions": per_view["ray_directions"][i],
                "depth_along_ray": per_view["depth_along_ray"][i] * s_map,
                "cam_trans": per_view["cam_trans"][i] * scale,
                "cam_quats": per_view["cam_quats"][i],
                "metric_scaling_factor": scale,
            }
            for i in range(num_views)
        ]
        return self._add_conf_and_mask(res, dense_out, num_views)

    @staticmethod
    def _add_conf_and_mask(res: List[dict], dense_out, num_views: int) -> List[dict]:
        if getattr(dense_out, "confidence", None) is not None:
            conf = dense_out.confidence.permute(0, 2, 3, 1).squeeze(-1).contiguous()
            for i, c in enumerate(conf.chunk(num_views, dim=0)):
                res[i]["conf"] = c
        if getattr(dense_out, "mask", None) is not None:
            mask = dense_out.mask.permute(0, 2, 3, 1).squeeze(-1).contiguous() > 0.5
            for i, m in enumerate(mask.chunk(num_views, dim=0)):
                res[i]["non_ambiguous_mask"] = m
        return res

    def _extract_depth_and_cameras(
        self, results: List[dict], h: int, w: int
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
        """Per-view dicts → ``(depth, conf, c2w_pred, K_pred)`` stacked as ``(V, B, ...)``."""
        geometry = import_research_module(
            "mapanything.utils.geometry", extra="ma", repo_url=MA_REPO_URL, what="MABackbone"
        )
        depths, confs, exts, ints = [], [], [], []
        for res in results:
            if "pts3d_cam" in res:
                depth_z = res["pts3d_cam"][..., 2]
            elif "depth_along_ray" in res and "ray_directions" in res:
                depth_z = (res["depth_along_ray"] * res["ray_directions"][..., 2:3].abs())[..., 0]
            elif "pts3d" in res:
                depth_z = res["pts3d"][..., 2]
            else:
                raise ValueError(f"cannot extract depth from MapAnything keys {list(res)}")
            depths.append(depth_z.abs())
            confs.append(res["conf"] if "conf" in res else torch.ones_like(depth_z))
            if "cam_trans" in res and "cam_quats" in res:
                rot = geometry.quaternion_to_rotation_matrix(res["cam_quats"])  # (B, 3, 3)
                c2w = torch.eye(4, device=rot.device, dtype=rot.dtype).repeat(rot.shape[0], 1, 1)
                c2w[:, :3, :3] = rot
                c2w[:, :3, 3] = res["cam_trans"]
                exts.append(c2w)
            if "ray_directions" in res:
                k_px = geometry.recover_pinhole_intrinsics_from_ray_directions(
                    res["ray_directions"]
                )
                ints.append(normalize_intrinsics_pixels(k_px, h, w))
        depth = torch.stack(depths)
        conf = torch.stack(confs)
        return (
            depth,
            conf,
            torch.stack(exts) if exts else None,
            torch.stack(ints) if ints else None,
        )

    # -- forward -----------------------------------------------------------------------
    def forward(
        self,
        images: Tensor,
        extrinsics: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
    ) -> BackboneOutput:
        info_sharing_base = import_research_module(
            "uniception.models.info_sharing.base",
            extra="ma",
            repo_url=MA_REPO_URL,
            what="MABackbone",
        )
        heads_base = import_research_module(
            "uniception.models.prediction_heads.base",
            extra="ma",
            repo_url=MA_REPO_URL,
            what="MABackbone",
        )

        images = resize_to_long_side(images, self.PATCH_SIZE, self.cfg.long_side)
        b, v, _c, h, w = images.shape
        ph, pw = h // self.PATCH_SIZE, w // self.PATCH_SIZE
        has_cameras = extrinsics is not None and intrinsics is not None
        device_type = images.device.type

        views = self._prepare_views(images, extrinsics, intrinsics)
        self.ma.geometric_input_config.update(
            _GEOMETRIC_INPUTS_WITH_CAMERAS if has_cameras else _GEOMETRIC_INPUTS_POSE_FREE
        )

        with torch.set_grad_enabled(not self.cfg.freeze_backbone):
            enc_feats, enc_registers = self.ma._encode_n_views(views)
        with (
            torch.autocast(device_type, enabled=False),
            torch.set_grad_enabled(not self.cfg.freeze_cam_enc),
        ):
            fused = self.ma._encode_and_fuse_optional_geometric_inputs(views, enc_feats)

        scale_token = self.ma.scale_token.unsqueeze(0).unsqueeze(-1).repeat(b, 1, 1)
        info_input = info_sharing_base.MultiViewTransformerInput(
            features=fused,
            additional_input_tokens_per_view=enc_registers,
            additional_input_tokens=scale_token,
        )
        if self.ma.info_sharing_return_type != "intermediate_features":
            raise ValueError(
                "MapAnything must be configured with info_sharing_return_type="
                f"'intermediate_features', got {self.ma.info_sharing_return_type!r}"
            )
        with torch.set_grad_enabled(not self.cfg.freeze_backbone):
            final_feat, intermediate = self.ma.info_sharing(info_input)

        h1 = torch.cat(list(enc_feats), dim=0)  # (V*B, C, Ph, Pw)
        h2 = torch.cat(intermediate[0].features, dim=0)
        h3 = torch.cat(intermediate[1].features, dim=0)
        h4 = torch.cat(final_feat.features, dim=0)
        if self._use_encoder_features_for_dpt:
            dense_inputs = [h1, h2, h3, h4]
        else:
            dense_inputs = [h2, h3, torch.cat(intermediate[2].features, dim=0), h4]

        output_shape = (h, w)
        with torch.autocast(device_type, enabled=False):
            with torch.set_grad_enabled(not self.cfg.freeze_dpt_head):
                dense_out = self.ma.downstream_dense_head(dense_inputs, output_shape)
            pose_out = None
            if self.ma.pred_head_type == "dpt+pose":
                with torch.set_grad_enabled(not self.cfg.freeze_cam_dec):
                    pose_head_out = self.ma.pose_head(
                        heads_base.PredictionHeadInput(last_feature=dense_inputs[-1])
                    )
                    pose_out = self.ma.pose_adaptor(
                        heads_base.AdaptorInput(
                            adaptor_feature=pose_head_out.decoded_channels,
                            output_shape_hw=output_shape,
                        )
                    )
            with torch.set_grad_enabled(not self.cfg.freeze_cam_dec):
                scale_head_out = self.ma.scale_head(
                    heads_base.PredictionHeadTokenInput(
                        last_feature=final_feat.additional_token_features
                    )
                )
                scale = self.ma.scale_adaptor(
                    heads_base.AdaptorInput(
                        adaptor_feature=scale_head_out.decoded_channels,
                        output_shape_hw=output_shape,
                    )
                ).value.squeeze(-1)  # (B, 1)

        results = self._assemble_scene_outputs(dense_out, pose_out, scale, v)
        pred_depth, conf, ext_pred, int_pred = self._extract_depth_and_cameras(results, h, w)

        dd: BackboneOutput.DataDict = {
            "depth": rearrange(pred_depth, "V B H W -> B V H W"),
            "depth_conf": rearrange(conf, "V B H W -> B V H W"),
        }
        for i, f in enumerate((h1, h2, h3, h4)):
            dd[f"patch_feat_{i}"] = rearrange(f, "(V B) C Ph Pw -> B V Ph Pw C", V=v, B=b)
        if extrinsics is not None:
            dd["extrinsics"] = extrinsics
        if intrinsics is not None:
            dd["intrinsics"] = intrinsics
        if ext_pred is not None:
            dd["extrinsics_pred"] = rearrange(ext_pred, "V B i j -> B V i j")
        if int_pred is not None:
            dd["intrinsics_pred"] = rearrange(int_pred, "V B i j -> B V i j")

        return BackboneOutput(
            data=dd,
            input_resolution=(h, w),
            dpt_resolution=(h, w),
            patch_resolution=(ph, pw),
        )
