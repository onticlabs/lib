"""Pretrained DINOv2 weights (Meta's public release) and a frozen feature-extractor wrapper.

Weights come from the ontic store when :data:`ontic_nn.weights.ONTIC_WEIGHTS` lists the
release URL, else through ``torch.hub`` on first use (cached under the hub directory);
nothing is fetched at import time.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, unpack
from torch import Tensor

from ontic_nn.weights import ontic_path

from .model import DINOv2, DinoVisionTransformer

Variant = Literal["vits", "vitb", "vitl", "vitg"]

_BASE = "https://dl.fbaipublicfiles.com/dinov2"
_URLS: Dict[Tuple[str, bool], str] = {
    ("vits", False): f"{_BASE}/dinov2_vits14/dinov2_vits14_pretrain.pth",
    ("vits", True): f"{_BASE}/dinov2_vits14/dinov2_vits14_reg4_pretrain.pth",
    ("vitb", False): f"{_BASE}/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    ("vitb", True): f"{_BASE}/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth",
    ("vitl", False): f"{_BASE}/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    ("vitl", True): f"{_BASE}/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth",
    ("vitg", False): f"{_BASE}/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
    ("vitg", True): f"{_BASE}/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth",
}

EMBED_DIM: Dict[str, int] = {"vits": 384, "vitb": 768, "vitl": 1024, "vitg": 1536}
PATCH_SIZE = 14


def load_pretrained_dinov2(
    variant: Variant = "vitl",
    use_reg: bool = False,
    use_checkpointing: bool = False,
) -> DinoVisionTransformer:
    """DINOv2 ``variant`` (patch 14, 4 registers if ``use_reg``) with Meta's released weights."""
    if (variant, use_reg) not in _URLS:
        raise ValueError(f"variant must be one of {sorted(EMBED_DIM)}, got {variant!r}")
    url = _URLS[(variant, use_reg)]
    model = DINOv2(
        variant, use_checkpointing=use_checkpointing, num_register_tokens=4 if use_reg else 0
    )
    stored = ontic_path(url, url.rsplit("/", 1)[-1])
    if stored is not None:
        state = torch.load(stored, map_location="cpu", weights_only=True)
    else:
        from torch.hub import load_state_dict_from_url  # lazy: network/cache access

        state = load_state_dict_from_url(
            url, map_location="cpu", check_hash=False, file_name=url.rsplit("/", 1)[-1]
        )
    model.load_state_dict(state, strict=True)
    return model


class StandaloneDinoExtractor(nn.Module):
    """Wrap a :class:`DinoVisionTransformer` as ``(images, mask=None) -> patch features``.

    ``images (..., 3, H, W)`` (ImageNet-normalised, ``H, W`` multiples of the patch
    size) -> ``(..., H // p, W // p, D)`` last-block normed patch tokens. ``mask``
    is a pixel-resolution bool ``(..., H, W)`` or ``(..., 1, H, W)``; a patch is
    replaced by the learned mask token if any of its pixels is True. With
    ``freeze`` the wrapped model has no trainable parameters and stays in eval
    mode across ``train()`` calls.
    """

    def __init__(self, model: DinoVisionTransformer, freeze: bool = True) -> None:
        super().__init__()
        self.model = model
        self.patch_size = int(model.patch_size)
        self.embed_dim = int(model.embed_dim)
        self.frozen = freeze
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)
            self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        variant: Variant = "vitl",
        use_reg: bool = False,
        freeze: bool = True,
        use_checkpointing: bool = False,
    ) -> StandaloneDinoExtractor:
        model = load_pretrained_dinov2(
            variant, use_reg=use_reg, use_checkpointing=use_checkpointing
        )
        return cls(model, freeze=freeze)

    def train(self, mode: bool = True) -> StandaloneDinoExtractor:
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    @staticmethod
    def _pixel_mask_to_patch_mask(mask: Tensor, patch_size: int, ph: int, pw: int) -> Tensor:
        """Max-pool a pixel mask ``(..., H, W)`` / ``(..., 1, H, W)`` to ``(..., ph, pw)`` bool."""
        if mask.dim() >= 4 and mask.shape[-3] == 1:
            mask = mask.squeeze(-3)
        flat, ps = pack([mask.float()], "* H W")
        reduced = F.max_pool2d(flat.unsqueeze(1), kernel_size=patch_size, stride=patch_size)
        reduced = reduced.squeeze(1)
        if tuple(reduced.shape[-2:]) != (ph, pw):
            raise ValueError(
                f"mask reduced to {tuple(reduced.shape[-2:])}, expected ({ph}, {pw}); "
                "mask must have the image's H, W"
            )
        [reduced] = unpack(reduced, ps, "* Ph Pw")
        return reduced > 0.5

    def forward(self, images: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        h, w = images.shape[-2], images.shape[-1]
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"DINOv2 requires H, W divisible by {self.patch_size}, got ({h}, {w})")
        ph, pw = h // self.patch_size, w // self.patch_size

        flat_images, ps = pack([images], "* C H W")

        patch_mask: Optional[Tensor] = None
        if mask is not None:
            pm = self._pixel_mask_to_patch_mask(mask, self.patch_size, ph, pw)
            patch_mask_flat, _ = pack([pm], "* Ph Pw")
            patch_mask = patch_mask_flat.reshape(patch_mask_flat.shape[0], ph * pw)

        out = self.model.forward_features(flat_images, masks=patch_mask)
        tokens = rearrange(out["x_norm_patchtokens"], "n (ph pw) c -> n ph pw c", ph=ph, pw=pw)
        [tokens] = unpack(tokens, ps, "* Ph Pw C")
        return tokens
