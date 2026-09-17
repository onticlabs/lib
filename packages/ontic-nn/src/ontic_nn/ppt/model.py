"""Plain point transformer: kNN attention interleaved with multi-view low-res attention.

kNN attention runs on a packed cloud; the low-resolution attention is global
over the ``(V, H, W)`` multi-view grid the cloud came from.

Point clouds use the packed layout of :mod:`ontic_nn.ppt.knn`: ``p (N, 3)``,
``x (N, C)``, ``offset (B,)`` with ``N = B * V * H * W`` ordered ``b, v, h, w``.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor

from ontic_nn.layers import Block

from .knn import Impl, knn_query


class KNNAttention(nn.Module):
    """Single-head attention of each point over its ``knn_samples`` neighbours.

    ``forward(p, x, offset, knn_idx=None)`` with ``x (N, C)`` returns ``(N, C)``;
    ``knn_idx (N, K)`` long indices are computed with ``impl`` when not given.
    """

    def __init__(
        self,
        channels: int,
        knn_samples: int = 16,
        proj_channels: Optional[int] = None,
        impl: Impl = "torch",
    ) -> None:
        super().__init__()
        self.knn_samples = knn_samples
        self.proj_channels = proj_channels
        self.impl = impl
        inner = proj_channels if proj_channels is not None else channels
        self.qkv = nn.Linear(channels, inner * 3, bias=False)
        self.proj = nn.Linear(inner, channels)

    def forward(
        self, p: Tensor, x: Tensor, offset: Tensor, knn_idx: Optional[Tensor] = None
    ) -> Tensor:
        if knn_idx is None:
            knn_idx, _ = knn_query(self.knn_samples, p, offset, impl=self.impl)
        x_q, x_k, x_v = torch.chunk(self.qkv(x), chunks=3, dim=-1)  # (N, C')
        scale = x_q.shape[-1] ** -0.5
        x_k = x_k[knn_idx]  # (N, K, C')
        x_v = x_v[knn_idx]
        scores = torch.matmul(x_q.unsqueeze(1), x_k.transpose(1, 2)) * scale  # (N, 1, K)
        out = torch.matmul(torch.softmax(scores, dim=2), x_v).squeeze(1)  # (N, C')
        return self.proj(out)


class MLP(nn.Module):
    """``fc2(gelu(fc1(x)))`` with 4x expansion on the last dim."""

    def __init__(self, channels: int, expansion: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels * expansion)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(channels * expansion, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """Pre-norm kNN-attention block: ``x + attn(norm1(x)); x + mlp(norm2(x))``."""

    def __init__(
        self,
        channels: int,
        knn_samples: int = 16,
        attn_proj_channels: Optional[int] = None,
        impl: Impl = "torch",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = KNNAttention(
            channels, knn_samples=knn_samples, proj_channels=attn_proj_channels, impl=impl
        )
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = MLP(channels)

    def forward(
        self, p: Tensor, x: Tensor, offset: Tensor, knn_idx: Optional[Tensor] = None
    ) -> Tensor:
        x = x + self.attn(p, self.norm1(x), offset, knn_idx=knn_idx)
        x = x + self.mlp(self.norm2(x))
        return x


class MultiViewLowResAttention(nn.Module):
    """Global attention over all views at ``1 / down_factor`` resolution.

    ``forward(x, v, h, w, context=None)`` with ``x (B, V*H*W, C)`` pixel-unshuffles
    each view by ``down_factor`` (8 = bilinear /2 then unshuffle 4), attends over
    the pooled tokens of all views, shuffles back and adds a residual.
    ``context (B, V'*H*W, C)`` switches to cross-attention (V' may differ from V).
    """

    def __init__(
        self,
        channels: int,
        down_factor: int = 4,
        attn_proj_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if down_factor not in (1, 2, 4, 8):
            raise ValueError(f"down_factor must be 1, 2, 4 or 8, got {down_factor}")
        self.down_factor = down_factor
        self.attn_proj_channels = attn_proj_channels

        if attn_proj_channels:
            ori_channels = channels
            self.proj0 = nn.Linear(channels, attn_proj_channels)
            channels = attn_proj_channels

        shuffle = 4 if down_factor == 8 else down_factor
        self.shuffle = shuffle

        self.proj1 = nn.Linear(channels * shuffle**2, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.proj2 = nn.Linear(channels, channels * shuffle**2)
        self.norm2 = nn.LayerNorm(channels * shuffle**2)
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

        if attn_proj_channels:
            self.proj3 = nn.Linear(channels, ori_channels)

        num_heads = 1 if attn_proj_channels else 4
        if channels % 32 != 0 and channels % 16 == 0:
            num_heads = 2
        if channels in (48, 3):
            num_heads = 1
        self.attn = Block(channels, num_heads)

    def _pool(self, x: Tensor, v: int, h: int, w: int) -> Tensor:
        x = rearrange(x, "b (v h w) c -> (b v) c h w", v=v, h=h, w=w)
        if self.down_factor == 8:
            x = F.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=True)
        x = F.pixel_unshuffle(x, self.shuffle)
        x = rearrange(x, "(b v) c h w -> b (v h w) c", v=v)
        return self.norm1(self.proj1(x))

    def _unpool(self, x: Tensor, v: int, h: int, w: int) -> Tensor:
        x = self.norm2(self.proj2(x))
        x = rearrange(
            x, "b (v h w) c -> (b v) c h w", v=v, h=h // self.down_factor, w=w // self.down_factor
        )
        x = F.pixel_shuffle(x, self.shuffle)
        x = self.conv(x)
        if self.down_factor == 8:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return rearrange(x, "(b v) c h w -> b (v h w) c", v=v)

    def forward(
        self, x: Tensor, v: int, h: int, w: int, context: Optional[Tensor] = None
    ) -> Tensor:
        if h % self.down_factor or w % self.down_factor:
            raise ValueError(
                f"(h, w)=({h}, {w}) must be divisible by down_factor {self.down_factor}"
            )
        residual = x
        if self.attn_proj_channels:
            x = self.proj0(x)
        tokens = self._pool(x, v, h, w)
        if context is None:
            tokens = self.attn(tokens)
        else:
            ctx_v = context.shape[1] // (h * w)
            tokens = self.attn(tokens, context=self._pool(context, ctx_v, h, w))
        x = self._unpool(tokens, v, h, w)
        if self.attn_proj_channels:
            x = self.proj3(x)
        return x + residual


class PlainPointTransformer(nn.Module):
    """``num_blocks`` x (kNN attention block, multi-view low-res attention).

    ``forward(p, x, offset, *, b, v, h, w, knn_idx=None)`` with ``p (N, 3)``,
    ``x (N, channels)``, ``offset (B,)`` and ``N = b * v * h * w`` returns
    ``(N, channels)``; ``return_knn_idx`` also returns the ``(N, K)`` indices used.
    ``impl`` selects the kNN backend (``"torch"`` brute force, ``"cuda"`` pointops).
    """

    def __init__(
        self,
        channels: int,
        knn_samples: int = 16,
        num_blocks: int = 4,
        attn_proj_channels: Optional[int] = None,
        cache_knn_idx: bool = True,
        use_checkpointing: bool = False,
        mvattn_down_factor: int = 4,
        impl: Impl = "torch",
    ) -> None:
        super().__init__()
        self.cache_knn_idx = cache_knn_idx
        self.knn_samples = knn_samples
        self.use_checkpointing = use_checkpointing
        self.impl = impl

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    channels,
                    knn_samples=knn_samples,
                    attn_proj_channels=attn_proj_channels,
                    impl=impl,
                )
                for _ in range(num_blocks)
            ]
        )
        self.mv_blocks = nn.ModuleList(
            [
                MultiViewLowResAttention(channels, down_factor=mvattn_down_factor)
                for _ in range(num_blocks)
            ]
        )

    def compute_knn(self, p: Tensor, offset: Tensor) -> Tensor:
        """``(N, K)`` long neighbour indices for the packed cloud ``p``."""
        knn_idx, _ = knn_query(self.knn_samples, p, offset, impl=self.impl)
        return knn_idx

    def forward(
        self,
        p: Tensor,
        x: Tensor,
        offset: Tensor,
        *,
        b: int,
        v: int,
        h: int,
        w: int,
        knn_idx: Optional[Tensor] = None,
        return_knn_idx: bool = False,
    ):
        if x.shape[0] != b * v * h * w:
            raise ValueError(f"expected {b * v * h * w} points for (b, v, h, w), got {x.shape[0]}")
        if knn_idx is None and self.cache_knn_idx:
            knn_idx = self.compute_knn(p, offset)

        for blk, mv_blk in zip(self.blocks, self.mv_blocks):
            if self.use_checkpointing and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(
                    blk, p, x, offset, knn_idx, use_reentrant=False
                )
            else:
                x = blk(p, x, offset, knn_idx=knn_idx)
            x = rearrange(x, "(b v h w) c -> b (v h w) c", b=b, v=v, h=h, w=w)
            if self.use_checkpointing and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(mv_blk, x, v, h, w, use_reentrant=False)
            else:
                x = mv_blk(x, v, h, w)
            x = rearrange(x, "b (v h w) c -> (b v h w) c", b=b, v=v, h=h, w=w)

        if return_knn_idx:
            return x, knn_idx
        return x


__all__ = [
    "KNNAttention",
    "MLP",
    "MultiViewLowResAttention",
    "PlainPointTransformer",
    "TransformerBlock",
]
