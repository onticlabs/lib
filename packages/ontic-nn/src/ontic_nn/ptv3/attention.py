"""Serialized (windowed) multi-head self-attention with optional RPE / RoPE."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import AttentionCfg
from .rope import Point3DRoPE
from .state import StageState


class RPE(nn.Module):
    """Bucketed relative position bias from integer grid offsets: ``(W, K, K, 3) -> (W, H, K, K)``."""

    def __init__(self, patch_size: int, num_heads: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord: Tensor) -> Tensor:
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)
            + self.pos_bnd
            + torch.arange(3, device=coord.device) * self.rpe_num
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        return out.permute(0, 3, 1, 2)


class RPEv2(nn.Module):
    """Learned distance from ``(dx, dy, dz, |d|)``: ``(W, pairs, 4) -> (W, H, pairs)``."""

    def __init__(self, patch_size: int, num_heads: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.distance_mlp = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1)
        )
        for layer in self.distance_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, rel_pos: Tensor) -> Tensor:
        distances = self.distance_mlp(rel_pos)  # (W, pairs, 1)
        return distances.squeeze(-1).unsqueeze(1).expand(-1, self.num_heads, -1)


def _flash_varlen():
    try:
        from flash_attn import flash_attn_varlen_qkvpacked_func
    except ImportError as e:
        raise ImportError(
            "SerializedAttention(backend='flash') requires flash-attn; "
            "install it against your torch build (pip install flash-attn)"
        ) from e
    return flash_attn_varlen_qkvpacked_func


class SerializedAttention(nn.Module):
    """Attention within ``patch_size`` windows of serialization order ``order_index``.

    ``backend="sdpa"`` runs ``F.scaled_dot_product_attention`` on
    ``(W, H, P, D)`` windows with a key mask for short windows;
    ``backend="flash"`` runs ``flash_attn_varlen_qkvpacked_func`` in fp16.
    """

    def __init__(
        self,
        channels: int,
        heads: int,
        patch_size: int,
        order_index: int,
        cfg: AttentionCfg,
        stage_grid_size: float = 1.0,
    ):
        super().__init__()
        if channels % heads:
            raise ValueError(f"channels {channels} not divisible by heads {heads}")
        if cfg.backend not in ("sdpa", "flash"):
            raise ValueError(f"backend must be 'sdpa' or 'flash', got {cfg.backend!r}")
        self.channels = channels
        self.heads = heads
        self.patch_size = patch_size
        self.order_index = order_index
        self.cfg = cfg
        self.scale = cfg.qk_scale or (channels // heads) ** -0.5
        self.stage_grid_size = float(stage_grid_size)
        self._flash = _flash_varlen() if cfg.backend == "flash" else None

        self.qkv = nn.Linear(channels, channels * 3, bias=cfg.qkv_bias)
        self.proj = nn.Linear(channels, channels)
        self.proj_drop = nn.Dropout(cfg.proj_drop)
        self.rpe = None
        if cfg.rpe == "v1":
            self.rpe = RPE(patch_size, heads)
        elif cfg.rpe == "v2":
            self.rpe = RPEv2(patch_size, heads)
        elif cfg.rpe != "none":
            raise ValueError(f"rpe must be 'none', 'v1' or 'v2', got {cfg.rpe!r}")
        self.rope = None
        if cfg.rope:
            self.rope = Point3DRoPE(
                channels // heads, base=cfg.rope_base, f0=cfg.rope_f0, impl=cfg.rope_impl
            )

    def _rope_source(self, state: StageState) -> Tensor:
        if self.cfg.rope_coord_source == "coord":
            return state.points.coord
        return state.grid_coord.to(state.points.coord.dtype) * self.stage_grid_size

    def _apply_rope(self, qkv: Tensor, order: Tensor, state: StageState) -> Tensor:
        h, d = self.heads, self.channels // self.heads
        q, k, v = qkv.view(-1, 3, h, d).unbind(dim=1)
        if self.cfg.rope_impl == "cuda":
            q, k = self.rope(q, k, self._rope_source(state)[order])
        else:
            if state.rope_cache is None:
                state.rope_cache = self.rope.get_cos_sin(self._rope_source(state))
            cos, sin = state.rope_cache
            q, k = self.rope.apply(q, k, cos[order].unsqueeze(1), sin[order].unsqueeze(1))
        return torch.stack([q, k, v], dim=1).reshape(-1, 3 * self.channels)

    def _rpe_bias(self, order: Tensor, state: StageState, window_index: Tensor) -> Tensor:
        w, p = window_index.shape
        if self.cfg.rpe == "v1":
            grid = state.grid_coord[order][window_index]  # (W, P, 3)
            return self.rpe(grid.unsqueeze(2) - grid.unsqueeze(1))
        coord = state.points.coord[order][window_index]
        i, j = torch.triu_indices(p, p, offset=0, device=coord.device)
        rel = coord[:, i] - coord[:, j]
        rel = torch.cat([rel, torch.norm(rel, dim=-1, keepdim=True)], dim=-1)
        distance = self.rpe(rel)  # (W, H, pairs)
        bias = torch.zeros(w, self.heads, p, p, dtype=distance.dtype, device=coord.device)
        bias[:, :, i, j] -= distance
        bias[:, :, j, i] -= distance
        return bias

    def _attend_sdpa(self, qkv: Tensor, order: Tensor, state: StageState, window) -> Tensor:
        h, c = self.heads, self.channels
        d = c // h
        x = qkv[window.window_index]  # (W, P, 3C)
        w, p = x.shape[:2]
        q, k, v = x.view(w, p, 3, h, d).permute(2, 0, 3, 1, 4).unbind(0)  # (W, H, P, D)
        key_mask = window.key_mask[:, None, None, :]  # (W, 1, 1, P)
        bias = self._rpe_bias(order, state, window.window_index) if self.rpe is not None else None
        dropout = self.cfg.attn_drop if self.training else 0.0
        if self.cfg.upcast_attention or self.cfg.upcast_softmax:
            if self.cfg.upcast_attention:
                q, k = q.float(), k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)
            if bias is not None:
                attn = attn + bias
            attn = attn.masked_fill(~key_mask, float("-inf"))
            if self.cfg.upcast_softmax:
                attn = attn.float()
            attn = F.dropout(attn.softmax(dim=-1), p=dropout, training=self.training).to(v.dtype)
            out = attn @ v
        else:
            if bias is None:
                attn_mask = key_mask
            else:
                attn_mask = bias.to(q.dtype).masked_fill(~key_mask, float("-inf"))
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout, scale=self.scale
            )
        out = out.transpose(1, 2).reshape(w * p, c)
        return out[window.slot_index]

    def _attend_flash(self, qkv: Tensor, window) -> Tensor:
        h, c = self.heads, self.channels
        out = self._flash(
            qkv.half().reshape(-1, 3, h, c // h),
            window.cu_seqlens,
            max_seqlen=self.patch_size,
            dropout_p=self.cfg.attn_drop if self.training else 0.0,
            softmax_scale=self.scale,
        )
        return out.reshape(-1, c).to(qkv.dtype)

    def forward(self, feat: Tensor, state: StageState) -> Tensor:
        serialization = state.require_serialization()
        window = state.window(self.patch_size)
        order = serialization.order[self.order_index][window.pad]
        inverse = window.unpad[serialization.inverse[self.order_index]]

        qkv = self.qkv(feat)[order]
        if self.rope is not None:
            qkv = self._apply_rope(qkv, order, state)
        if self._flash is not None:
            out = self._attend_flash(qkv, window)
        else:
            out = self._attend_sdpa(qkv, order, state, window)
        out = out[inverse]
        return self.proj_drop(self.proj(out))
