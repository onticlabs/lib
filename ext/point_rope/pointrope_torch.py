"""Pure-PyTorch implementation of 3D RoPE for point clouds.

Continuous-coord variant: positions are float xyz (not integer grid indices),
so gradients flow through `coord` and you can apply coord-space augmentation
without quantisation.

Design follows the standard 3D-RoPE family (Utonia / LitePT-PointROPE):
head_dim is split into 3 equal chunks, one per spatial axis (x, y, z). Each
chunk gets standard 1D RoPE: a `rotate_half` trick gives a per-axis rotation
of pairs whose angle is `coord_axis * inv_freq`.

Compatible with FlashAttention: this is a Q/K linear map, not a per-pair bias.

Usage in attention:
    rope = Point3DRoPE(head_dim=channels // num_heads, base=100.0)
    # once per stage (cache on Point dict so blocks share it):
    cos_unord, sin_unord = rope.get_cos_sin(point.coord)         # (N, 3, D/3)
    # gather to attention's patched/ordered layout (same `order` as qkv):
    cos = cos_unord[order]                                        # (N_pad, 3, D/3)
    sin = sin_unord[order]
    # apply to q, k (NOT v):
    q, k = rope.apply(q, k, cos, sin)
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class Point3DRoPE(nn.Module):
    """Continuous-coord 3D RoPE module (pure PyTorch).

    Args:
        head_dim: per-head feature dim. Must be divisible by 6 (3 axes × 2 for
            rotate_half pairs).
        base: frequency base, same role as LLM RoPE's `base=10000`. Smaller
            values wrap faster, appropriate for sub-meter metric coords.
            Default 100 matches LitePT's default.
        F0: overall frequency scale. Effective per-axis freq table is
            `F0 / base^(2i/(head_dim/3))`. Useful to decouple workspace scale
            from `base`. Default 1.0.
        use_cuda: if True, route `forward()` calls through the compiled CUDA
            extension when available (forward-only w.r.t. coord — no coord
            grad along this path). The `get_cos_sin` / `apply` API stays pure
            PyTorch regardless and remains autograd-traceable through coord.
    """

    def __init__(
        self,
        head_dim: int,
        base: float = 100.0,
        F0: float = 1.0,
        use_cuda: bool = False,
    ):
        super().__init__()
        assert head_dim % 6 == 0, (
            f"head_dim must be divisible by 6 for 3D RoPE "
            f"(3 axes × 2 for rotate_half pairs), got {head_dim}. "
            f"Adjust channels or num_heads so channels // num_heads is a multiple of 6."
        )
        self.head_dim = head_dim
        self.chunk_dim = head_dim // 3
        self.base = float(base)
        self.F0 = float(F0)

        # Lazy import — only used by forward() when use_cuda is True; importing
        # at module top would pull the compiled extension and fail when it
        # isn't built (which is the common case).
        self.use_cuda = False
        if use_cuda:
            try:
                from .pointrope_cuda import apply_rope as _cuda_apply_rope  # noqa: F401

                self.use_cuda = True
            except Exception:
                self.use_cuda = False

        # inv_freq has chunk_dim / 2 entries; cos/sin double-up via cat([f, f]).
        half = self.chunk_dim // 2
        inv_freq = self.F0 / (self.base ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim))
        assert inv_freq.shape == (half,)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        """Standard RoPE rotate_half over the last dim: [-x2, x1]."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def get_cos_sin(self, coord: Tensor) -> tuple[Tensor, Tensor]:
        """Compute cos/sin from continuous coords.

        Args:
            coord: (N, 3) float xyz positions.
        Returns:
            cos, sin: both (N, 3, chunk_dim). Differentiable in coord.
        """
        # (N, 3, 1) * (chunk_dim/2,) → (N, 3, chunk_dim/2)
        freqs = coord.unsqueeze(-1) * self.inv_freq.to(coord.dtype)
        # standard RoPE: duplicate so rotate_half works
        freqs = torch.cat((freqs, freqs), dim=-1)  # (N, 3, chunk_dim)
        return freqs.cos(), freqs.sin()

    def apply(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate q and k.

        Args:
            q, k: (..., head_dim). Any leading shape; rotation acts on last dim.
            cos, sin: must broadcast against q.view(..., 3, chunk_dim).
                Typical shape: (N, 1, 3, chunk_dim) where N matches q's N-prefix.
        Returns:
            q_rot, k_rot: same shape as inputs.
        """
        q_shape, k_shape = q.shape, k.shape
        q_v = q.unflatten(-1, (3, self.chunk_dim))
        k_v = k.unflatten(-1, (3, self.chunk_dim))
        q_rot = q_v * cos + self.rotate_half(q_v) * sin
        k_rot = k_v * cos + self.rotate_half(k_v) * sin
        return q_rot.reshape(q_shape), k_rot.reshape(k_shape)

    def forward(self, q: Tensor, k: Tensor, coord: Tensor) -> tuple[Tensor, Tensor]:
        """Apply 3D RoPE end-to-end without caching.

        For per-stage caching of cos/sin across attention blocks, call
        `get_cos_sin` once and pass cos/sin into `apply` directly.

        Args:
            q, k: (N, H, head_dim).
            coord: (N, 3) continuous positions matching q/k's N axis.
        """
        if self.use_cuda and q.is_cuda and not q.requires_grad and not coord.requires_grad:
            from .pointrope_cuda import apply_rope as _cuda_apply_rope

            return _cuda_apply_rope(q, k, coord, self.base, self.F0)

        cos, sin = self.get_cos_sin(coord)
        cos = cos.unsqueeze(-3)  # (N, 1, 3, chunk_dim) when q is (N, H, head_dim)
        sin = sin.unsqueeze(-3)
        return self.apply(q, k, cos, sin)


__all__ = ["Point3DRoPE"]
