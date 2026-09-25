"""3D rotary position embedding on continuous point coordinates.

``head_dim`` is split into three equal chunks (x, y, z); each chunk gets 1-D
RoPE with angle ``coord_axis * inv_freq`` via the ``rotate_half`` trick.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

RopeImpl = Literal["torch", "cuda"]


def _cuda_ext():
    try:
        from point_rope_cuda import _C
    except ImportError as e:
        raise ImportError(
            "Point3DRoPE(impl='cuda') requires the point_rope_cuda extension; "
            "run scripts/install_cuda_ext.sh point_rope"
        ) from e
    return _C


class _CudaRope(torch.autograd.Function):
    """In-place rotation of ``(B, N, H, D)`` tokens; backward applies the inverse rotation."""

    @staticmethod
    def forward(ctx, tokens, positions, base, f0):
        ctx.save_for_backward(positions)
        ctx.base, ctx.f0 = base, f0
        _cuda_ext().pointrope(tokens, positions, base, f0)
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_tokens):
        (positions,) = ctx.saved_tensors
        _cuda_ext().pointrope(grad_tokens, positions, ctx.base, -ctx.f0)
        ctx.mark_dirty(grad_tokens)
        return grad_tokens, None, None, None


class Point3DRoPE(nn.Module):
    """RoPE over ``(N, 3)`` float coordinates for ``(..., head_dim)`` queries/keys.

    ``head_dim`` must be divisible by 6. Per-axis frequencies are
    ``f0 / base ** (2i / (head_dim / 3))``. ``impl="cuda"`` routes
    :meth:`forward` through the vendored kernel (no coordinate gradient).
    """

    def __init__(
        self, head_dim: int, base: float = 100.0, f0: float = 1.0, impl: RopeImpl = "torch"
    ):
        super().__init__()
        if head_dim % 6:
            raise ValueError(f"head_dim must be divisible by 6 for 3D RoPE, got {head_dim}")
        if impl not in ("torch", "cuda"):
            raise ValueError(f"impl must be 'torch' or 'cuda', got {impl!r}")
        self.head_dim = head_dim
        self.chunk_dim = head_dim // 3
        self.base = float(base)
        self.f0 = float(f0)
        self.impl = impl
        if impl == "cuda":
            _cuda_ext()
        inv_freq = self.f0 / (
            self.base ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def get_cos_sin(self, coord: Tensor) -> tuple[Tensor, Tensor]:
        """``(N, 3)`` coordinates -> ``cos, sin`` of shape ``(N, 3, chunk_dim)``."""
        freqs = coord.unsqueeze(-1) * self.inv_freq.to(coord.dtype)
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos(), freqs.sin()

    def apply(self, q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate ``q, k (..., head_dim)`` with ``cos, sin`` broadcastable to ``(..., 3, chunk_dim)``."""
        q_shape, k_shape = q.shape, k.shape
        q_v = q.unflatten(-1, (3, self.chunk_dim))
        k_v = k.unflatten(-1, (3, self.chunk_dim))
        q_rot = q_v * cos + self.rotate_half(q_v) * sin
        k_rot = k_v * cos + self.rotate_half(k_v) * sin
        return q_rot.reshape(q_shape), k_rot.reshape(k_shape)

    def forward(self, q: Tensor, k: Tensor, coord: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate ``q, k (N, H, head_dim)`` by ``coord (N, 3)`` end to end."""
        if self.impl == "cuda":
            if not (q.is_cuda and k.is_cuda and coord.is_cuda):
                raise ValueError("Point3DRoPE(impl='cuda') requires CUDA tensors")
            pos = coord.to(torch.float32).unsqueeze(0).contiguous()
            q4 = q.unsqueeze(0).contiguous().clone()
            k4 = k.unsqueeze(0).contiguous().clone()
            _CudaRope.apply(q4, pos, self.base, self.f0)
            _CudaRope.apply(k4, pos, self.base, self.f0)
            return q4.squeeze(0), k4.squeeze(0)
        cos, sin = self.get_cos_sin(coord)
        return self.apply(q, k, cos.unsqueeze(-3), sin.unsqueeze(-3))
