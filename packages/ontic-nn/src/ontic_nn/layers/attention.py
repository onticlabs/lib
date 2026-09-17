# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Multi-head attention on ``torch.nn.functional.scaled_dot_product_attention``."""

from __future__ import annotations

from typing import Optional

import torch.nn.functional as F
from torch import Tensor, nn


class Attention(nn.Module):
    """Multi-head self-attention with a fused ``qkv`` projection.

    ``forward(x, context=None)``: ``x (B, Nq, C)``; with ``context (B, Nk, C)``
    queries come from ``x`` and keys/values from ``context`` (cross-attention),
    using the k/v slices of the same ``qkv`` weight. Returns ``(B, Nq, C)``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tensor:
        b, nq, c = x.shape
        if context is None:
            qkv = self.qkv(x).reshape(b, nq, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        else:
            weight, bias = self.qkv.weight, self.qkv.bias
            q = F.linear(x, weight[:c], None if bias is None else bias[:c])
            kv = F.linear(context, weight[c:], None if bias is None else bias[c:])
            q = self._split_heads(q)
            k, v = kv.chunk(2, dim=-1)
            k, v = self._split_heads(k), self._split_heads(v)

        dropout = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout)
        out = out.transpose(1, 2).reshape(b, nq, c)
        out = self.proj(out)
        return self.proj_drop(out)
