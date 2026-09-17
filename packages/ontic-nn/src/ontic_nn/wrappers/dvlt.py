"""DVLT backbone wrapper (NVIDIA "Déjà View" looping transformer; pose-free).

A DINOv2 patch embedder feeds one shared recurrent block applied ``inference_steps`` times.
Patch features are tapped either from the recurrent state (``"loop"``) or from DINOv2 blocks
(``"dino"``), per feature level. Cameras come from the predicted ray field.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .common import (
    BackboneBase,
    BackboneConfig,
    BackboneOutput,
    amp_dtype,
    import_research_module,
    normalize_intrinsics_pixels,
    resize_to_long_side,
    resolve_checkpoint,
    set_frozen,
)
from .registry import register_backbone

DVLT_REPO_URL = "https://github.com/nv-tlabs/dvlt"


def install_flex_attention_shim() -> None:
    """Import-only ``torch.nn.attention.flex_attention`` stub for torch < 2.5.

    DVLT imports ``flex_attention`` at module import but only calls it on the
    ``block_mask`` path, which pose-free inference never takes. No-op when torch has it.
    """
    import torch.nn.attention as tna

    if hasattr(tna, "flex_attention") or "torch.nn.attention.flex_attention" in sys.modules:
        return
    mod = types.ModuleType("torch.nn.attention.flex_attention")

    def unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "flex_attention needs torch >= 2.5; DVLT's block_mask path is unavailable"
        )

    mod.flex_attention = unavailable
    sys.modules["torch.nn.attention.flex_attention"] = mod


@register_backbone("dvlt")
@dataclass(kw_only=True)
class DVLTBackboneConfig(BackboneConfig):
    """``model_dir`` is the HF repo holding ``model.safetensors``; empty → random init.

    ``feature_sources`` has one entry per feature level, ``"loop"`` (recurrent state at
    ``loop_steps[i]``; ``None`` → 4 evenly spaced steps) or ``"dino"`` (DINOv2 block
    ``dino_layers[i]``).
    """

    model_dir: str = "nvidia/dvlt"
    long_side: int = 504
    img_size: int = 504
    patch_embed: str = "dinov2_vitb14_reg"
    embed_dim: int = 768
    num_steps: int = 16
    min_steps: int = 8
    inference_steps: int = 12
    num_register_tokens: int = 4
    load_patch_embed_weights: bool = False
    feature_sources: Tuple[str, ...] = ("loop", "loop", "loop", "loop")
    dino_layers: Tuple[int, ...] = (2, 5, 8, 11)
    loop_steps: Optional[Tuple[int, ...]] = None

    def build(self) -> DVLTBackbone:
        return DVLTBackbone(self)


class DVLTBackbone(BackboneBase):
    """DVLT with ``requires_grad`` freezing: patch embedder + recurrent block, ray + depth
    decoders (``freeze_dpt_head``) and camera head (``freeze_cam_dec``). Fully frozen runs under
    ``no_grad``; unfreezing anything makes the same fixed-K forward differentiable."""

    PATCH_SIZE = 14

    def __init__(self, cfg: DVLTBackboneConfig) -> None:
        super().__init__()
        self.cfg = cfg
        if len(cfg.feature_sources) != 4:
            raise ValueError(f"feature_sources needs 4 entries, got {len(cfg.feature_sources)}")
        if any(s not in ("dino", "loop") for s in cfg.feature_sources):
            raise ValueError(
                f"feature_sources entries must be 'dino' or 'loop': {cfg.feature_sources}"
            )

        self.dvlt, self.loaded_checkpoint = self._load(cfg)
        self.dvlt.eval()
        m = self.dvlt
        set_frozen(
            [m.patch_embed_encoder, m.recurrent_blocks, m.camera_token, m.register_token],
            cfg.freeze_backbone,
        )
        set_frozen([m.ray_decoder, m.depth_decoder], cfg.freeze_dpt_head)
        if m.camera_head is not None:
            set_frozen([m.camera_head], cfg.freeze_cam_dec)
        if hasattr(m.patch_embed_encoder, "mask_token"):  # unused upstream; keep it frozen
            m.patch_embed_encoder.mask_token.requires_grad_(False)

    @staticmethod
    def _load(cfg: DVLTBackboneConfig):
        install_flex_attention_shim()
        model_mod = import_research_module(
            "dvlt.model.dvlt.model", extra="dvlt", repo_url=DVLT_REPO_URL, what="DVLTBackbone"
        )
        # ``DVLT.load_pretrained`` logs through accelerate, which needs its global state.
        accelerate = import_research_module(
            "accelerate", extra="dvlt", repo_url=DVLT_REPO_URL, what="DVLTBackbone"
        )
        accelerate.PartialState()

        wrap = model_mod.DVLT(
            img_size=cfg.img_size,
            patch_embed=cfg.patch_embed,
            embed_dim=cfg.embed_dim,
            num_steps=cfg.num_steps,
            min_steps=cfg.min_steps,
            inference_steps=cfg.inference_steps,
            num_register_tokens=cfg.num_register_tokens,
            load_patch_embed_weights=cfg.load_patch_embed_weights,
        )
        path = resolve_checkpoint(
            cfg.checkpoint_path,
            cfg.model_dir,
            filename="model.safetensors",
            cache_dir=cfg.cache_dir,
            allow_download=cfg.allow_download,
            extra="dvlt",
            what="DVLT checkpoint",
        )
        if path is not None:
            wrap.load_pretrained(path, strict=False)
        return wrap.model, path

    @property
    def encoder_dim(self) -> int:
        return self.cfg.embed_dim

    def _loop_steps(self) -> List[int]:
        if self.cfg.loop_steps is not None:
            return list(self.cfg.loop_steps)
        k = self.dvlt.inference_steps
        return torch.linspace(0, max(k - 1, 0), 4).round().long().tolist()

    def forward(
        self,
        images: Tensor,
        extrinsics: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
    ) -> BackboneOutput:
        rays_mod = import_research_module(
            "dvlt.common.rays", extra="dvlt", repo_url=DVLT_REPO_URL, what="DVLTBackbone"
        )
        images = resize_to_long_side(images, self.PATCH_SIZE, self.cfg.long_side)
        b, v, _c, h, w = images.shape
        ph, pw = h // self.PATCH_SIZE, w // self.PATCH_SIZE
        npatch = ph * pw
        cfg = self.cfg

        loop_idx = self._loop_steps()
        dino_idx = list(cfg.dino_layers)
        loop_states: List[Tensor] = []
        dino_states: Dict[int, Tensor] = {}
        handles = []
        if "loop" in cfg.feature_sources:
            handles.append(
                self.dvlt.recurrent_blocks[0].register_forward_hook(
                    lambda _m, _i, o: loop_states.append(o)
                )
            )
        for j in {dino_idx[i] for i, s in enumerate(cfg.feature_sources) if s == "dino"}:
            handles.append(
                self.dvlt.patch_embed_encoder.blocks[j].register_forward_hook(
                    self._make_store_hook(dino_states, j)
                )
            )

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
                out = self._forward_inference(images, any_trainable)
        finally:
            for hd in handles:
                hd.remove()

        pred_depth = out["depth"].squeeze(-1).float()  # (B, V, H, W)
        depth_conf = out["depth_conf"].float()  # already expp1
        rays = out["rays"].float()  # (B, V, H, W, 6)

        c2w, k_px = rays_mod.rays_to_pose(
            rays, torch.ones_like(depth_conf), h, w, patch_size=self.PATCH_SIZE
        )

        def patch(state: Tensor) -> Tensor:  # (B*V, T, C); patch tokens are the last npatch
            return state[:, -npatch:].reshape(b, v, ph, pw, cfg.embed_dim).float()

        feats = [
            patch(loop_states[loop_idx[i]] if src == "loop" else dino_states[dino_idx[i]])
            for i, src in enumerate(cfg.feature_sources)
        ]
        dd: BackboneOutput.DataDict = {
            "depth": pred_depth,
            "depth_conf": depth_conf,
            "patch_feat_0": feats[0],
            "patch_feat_1": feats[1],
            "patch_feat_2": feats[2],
            "patch_feat_3": feats[3],
            "extrinsics_pred": c2w,
            "intrinsics_pred": normalize_intrinsics_pixels(k_px, h, w),
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

    def _forward_inference(self, images: Tensor, any_trainable: bool):
        """Upstream ``forward_inference`` under the caller's grad mode.

        It is decorated ``@torch.no_grad()``; the undecorated body is reachable through
        ``__wrapped__`` so finetuning can run the identical solver with gradients.
        """
        fn = getattr(type(self.dvlt).forward_inference, "__wrapped__", None)
        if fn is None:
            if any_trainable:
                raise RuntimeError(
                    "DVLTModel.forward_inference has no __wrapped__; its @torch.no_grad cannot be "
                    "bypassed, so DVLT cannot be finetuned with this dvlt version"
                )
            return self.dvlt.forward_inference(images)
        return fn(self.dvlt, images)

    @staticmethod
    def _make_store_hook(store: Dict[int, Tensor], idx: int):
        def hook(_module, _inp, output):
            store[idx] = output

        return hook
