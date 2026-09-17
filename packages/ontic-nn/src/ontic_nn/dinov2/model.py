# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""DINOv2 vision transformer (CLS + optional register tokens + patch tokens)."""

from __future__ import annotations

import math
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch import Tensor
from torch.nn.init import trunc_normal_

from ontic_nn.layers import Block, Mlp, PatchEmbed, SwiGLUFFNFused


def named_apply(
    fn: Callable, module: nn.Module, name: str = "", depth_first: bool = True, include_root=False
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn, module=child_module, name=child_name, depth_first=depth_first, include_root=True
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


class BlockChunk(nn.ModuleList):
    def forward(self, x: Tensor) -> Tensor:
        for b in self:
            x = b(x)
        return x


class DinoVisionTransformer(nn.Module):
    """ViT with CLS token, ``num_register_tokens`` registers and bicubic pos-embed resizing.

    Input images are ``(B, in_chans, H, W)`` with ``H, W`` multiples of
    ``patch_size``. Token layout is ``[cls, reg_0..reg_{R-1}, patch_0..]``.
    ``forward`` returns the dict of :meth:`forward_features`.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        ffn_bias: bool = True,
        proj_bias: bool = True,
        drop_path_rate: float = 0.0,
        drop_path_uniform: bool = False,
        init_values: Optional[float] = None,  # for layerscale: None or 0 => no layerscale
        embed_layer: Callable[..., nn.Module] = PatchEmbed,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        block_fn: Callable[..., nn.Module] = Block,
        ffn_layer: str = "mlp",
        block_chunks: int = 1,
        num_register_tokens: int = 0,
        interpolate_antialias: bool = False,
        interpolate_offset: float = 0.1,
        use_checkpointing: bool = False,
    ) -> None:
        """
        Args:
            img_size: input image size the positional-embedding table is sized for.
            patch_size: patch size.
            in_chans: number of input channels.
            embed_dim: embedding dimension.
            depth: number of transformer blocks.
            num_heads: number of attention heads.
            mlp_ratio: ratio of FFN hidden dim to embedding dim.
            qkv_bias / proj_bias / ffn_bias: bias flags for the qkv, attention output
                and FFN projections.
            drop_path_rate: stochastic-depth rate (linearly increasing over depth unless
                ``drop_path_uniform``).
            init_values: LayerScale init; ``None`` or 0 disables LayerScale.
            ffn_layer: ``"mlp"``, ``"swiglu"``/``"swiglufused"`` or ``"identity"``.
            block_chunks: split the block list into chunks (FSDP wrapping); 0 keeps a
                flat ``nn.ModuleList``.
            num_register_tokens: number of register tokens inserted after CLS.
            interpolate_antialias / interpolate_offset: positional-embedding resize options.
            use_checkpointing: activation checkpointing per block.
        """
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        self.use_checkpointing = use_checkpointing

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        if num_register_tokens < 0:
            raise ValueError("num_register_tokens must be non-negative")
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))
            if num_register_tokens
            else None
        )

        if drop_path_uniform:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        ffn_class: Callable[..., nn.Module]
        if ffn_layer == "mlp":
            ffn_class = Mlp
        elif ffn_layer in ("swiglufused", "swiglu"):
            ffn_class = SwiGLUFFNFused
        elif ffn_layer == "identity":

            def ffn_class(*args, **kwargs):
                return nn.Identity()

        else:
            raise ValueError(f"unknown ffn_layer {ffn_layer!r}")

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_class,
                init_values=init_values,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # keep the block index consistent when chunking
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))

        self.init_weights()

    def init_weights(self) -> None:
        trunc_normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=1e-6)
        if self.register_tokens is not None:
            nn.init.normal_(self.register_tokens, std=1e-6)
        named_apply(init_weights_vit_timm, self)

    def _run_block(self, blk: nn.Module, x: Tensor) -> Tensor:
        if self.use_checkpointing:
            return torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def interpolate_pos_encoding(self, x: Tensor, w: int, h: int) -> Tensor:
        """Resize the ``(1, 1 + M*M, D)`` table to the ``(w // p, h // p)`` patch grid."""
        previous_dtype = x.dtype
        npatch = x.shape[1] - 1
        n = self.pos_embed.shape[1] - 1
        if npatch == n and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        m = int(math.sqrt(n))  # patches per side of the stored table
        if n != m * m:
            raise ValueError(f"pos_embed holds {n} patches, expected a square grid")
        kwargs = {}
        if self.interpolate_offset:
            # Historical kludge: add a small number to avoid floating point error in the
            # interpolation, see https://github.com/facebookresearch/dino/issues/8
            sx = float(w0 + self.interpolate_offset) / m
            sy = float(h0 + self.interpolate_offset) / m
            kwargs["scale_factor"] = (sx, sy)
        else:
            kwargs["size"] = (w0, h0)
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, m, m, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=self.interpolate_antialias,
            **kwargs,
        )
        if (w0, h0) != tuple(patch_pos_embed.shape[-2:]):
            raise RuntimeError("positional-embedding interpolation produced the wrong grid")
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x: Tensor, masks: Optional[Tensor] = None) -> Tensor:
        """``(B, C, H, W)`` image -> ``(B, 1 + R + N, D)`` tokens; ``masks (B, N)`` bool
        replaces patch tokens with the learned mask token."""
        _, _, w, h = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)

        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)

        if self.register_tokens is not None:
            x = torch.cat(
                (x[:, :1], self.register_tokens.expand(x.shape[0], -1, -1), x[:, 1:]), dim=1
            )

        return x

    def forward_features(self, x: Tensor, masks: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """Run all blocks. Keys: ``x_norm_clstoken (B, D)``, ``x_norm_regtokens (B, R, D)``,
        ``x_norm_patchtokens (B, N, D)``, ``x_prenorm (B, 1+R+N, D)``, ``masks``."""
        x = self.prepare_tokens_with_masks(x, masks)

        for blk in self.blocks:
            x = self._run_block(blk, x)

        x_norm = self.norm(x)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            "masks": masks,
        }

    def _get_intermediate_layers_not_chunked(
        self, x: Tensor, n: Union[int, Sequence[int]] = 1
    ) -> List[Tensor]:
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a sequence, take those.
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x = self._run_block(blk, x)
            if i in blocks_to_take:
                output.append(x)
        if len(output) != len(blocks_to_take):
            raise ValueError(f"only {len(output)} / {len(blocks_to_take)} blocks found")
        return output

    def _get_intermediate_layers_chunked(
        self, x: Tensor, n: Union[int, Sequence[int]] = 1
    ) -> List[Tensor]:
        x = self.prepare_tokens_with_masks(x)
        output, i, total_block_len = [], 0, len(self.blocks[-1])
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for block_chunk in self.blocks:
            for blk in block_chunk[i:]:  # skips the leading nn.Identity() placeholders
                x = self._run_block(blk, x)
                if i in blocks_to_take:
                    output.append(x)
                i += 1
        if len(output) != len(blocks_to_take):
            raise ValueError(f"only {len(output)} / {len(blocks_to_take)} blocks found")
        return output

    def get_intermediate_layers(
        self,
        x: Tensor,
        n: Union[int, Sequence[int]] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        norm: bool = True,
    ):
        """Patch tokens after the last ``n`` blocks (or the listed block indices).

        Each entry is ``(B, N, D)``, or ``(B, D, H // p, W // p)`` with ``reshape``;
        with ``return_class_token`` entries are ``(patch_tokens, cls_token (B, D))``.
        """
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            b, _, w, h = x.shape
            outputs = [
                out.reshape(b, w // self.patch_size, h // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)

    def forward(self, x: Tensor, masks: Optional[Tensor] = None) -> Dict[str, Tensor]:
        return self.forward_features(x, masks)


def init_weights_vit_timm(module: nn.Module, name: str = "") -> None:
    """ViT weight initialization, original timm impl (for reproducibility)."""
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def vit_small(
    patch_size: int = 16, num_register_tokens: int = 0, **kwargs
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_base(patch_size: int = 16, num_register_tokens: int = 0, **kwargs) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_large(
    patch_size: int = 16, num_register_tokens: int = 0, **kwargs
) -> DinoVisionTransformer:
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


def vit_giant2(
    patch_size: int = 16, num_register_tokens: int = 0, **kwargs
) -> DinoVisionTransformer:
    """Close to ViT-giant: embed dim 1536 and 24 heads (64 per head)."""
    return DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        mlp_ratio=4.0,
        num_register_tokens=num_register_tokens,
        **kwargs,
    )


_MODEL_ZOO = {"vits": vit_small, "vitb": vit_base, "vitl": vit_large, "vitg": vit_giant2}


def DINOv2(  # noqa: N802 — upstream factory name
    model_name: str,
    use_checkpointing: bool = False,
    num_register_tokens: int = 0,
) -> DinoVisionTransformer:
    """Randomly initialised DINOv2 model (patch 14, img 518, LayerScale) for ``vits/b/l/g``."""
    if model_name not in _MODEL_ZOO:
        raise ValueError(f"model_name must be one of {sorted(_MODEL_ZOO)}, got {model_name!r}")
    return _MODEL_ZOO[model_name](
        img_size=518,
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp" if model_name != "vitg" else "swiglufused",
        block_chunks=0,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=num_register_tokens > 0,
        interpolate_offset=0.0 if num_register_tokens > 0 else 0.1,
        use_checkpointing=use_checkpointing,
    )
