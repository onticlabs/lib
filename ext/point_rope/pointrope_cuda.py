"""CUDA wrapper for the compiled `point_rope_cuda._C` extension.

Forward-only w.r.t. positions (no coord gradient). Backward through tokens uses
the kernel with negated F0 (rotation is orthogonal; inverse = applying the
opposite-sign phase).

Importing this module raises ImportError if the extension isn't built —
callers should fall back to `.pointrope_torch.Point3DRoPE`.
"""

from __future__ import annotations

import torch

# Will raise ImportError if the extension isn't built; the parent
# `__init__.py` catches this and exposes only the PyTorch path.
from point_rope_cuda import _C as _ext  # type: ignore  # noqa: F401


class _PointRopeFunction(torch.autograd.Function):
    """In-place RoPE rotation on (B, N, H, D) tokens."""

    @staticmethod
    def forward(ctx, tokens, positions, base, F0):
        ctx.save_for_backward(positions)
        ctx.saved_base = base
        ctx.saved_F0 = F0
        _ext.pointrope(tokens, positions, base, F0)
        ctx.mark_dirty(tokens)
        return tokens

    @staticmethod
    def backward(ctx, grad_tokens):
        (positions,) = ctx.saved_tensors
        _ext.pointrope(grad_tokens, positions, ctx.saved_base, -ctx.saved_F0)
        ctx.mark_dirty(grad_tokens)
        return grad_tokens, None, None, None


def apply_rope(q: torch.Tensor, k: torch.Tensor, coord: torch.Tensor, base: float, F0: float):
    """Apply 3D RoPE to q and k via CUDA kernel.

    Args:
        q, k: (N, H, head_dim) float tensors on CUDA. head_dim % 6 == 0.
        coord: (N, 3) tensor of positions on CUDA. Float **or** integer dtype
            — integer values are cast to float32 here (LitePT-style integer
            `grid_coord` vs continuous metric xyz both work; the kernel itself
            does `pos * inv_freq` in float regardless, so there is no kernel-
            level perf difference between the two).
        base, F0: frequency hyperparameters (see Point3DRoPE).

    Returns:
        (q_rot, k_rot): rotated copies (the kernel writes in place on cloned
        contiguous buffers, so the originals are untouched).
    """
    assert q.is_cuda and k.is_cuda and coord.is_cuda
    if coord.dtype != torch.float32:
        coord = coord.to(torch.float32)

    # Kernel expects (B, N, H, D); add singleton batch and clone (in-place op).
    q4 = q.unsqueeze(0).contiguous().clone()
    k4 = k.unsqueeze(0).contiguous().clone()
    pos = coord.unsqueeze(0).contiguous()

    _PointRopeFunction.apply(q4, pos, base, F0)
    _PointRopeFunction.apply(k4, pos, base, F0)
    return q4.squeeze(0), k4.squeeze(0)


__all__ = ["apply_rope"]
