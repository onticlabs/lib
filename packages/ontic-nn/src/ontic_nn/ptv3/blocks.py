"""PTv3 transformer block: sparse-conv positional encoding, serialized attention, MLP."""

from __future__ import annotations

from typing import Callable

import torch.utils.checkpoint
from torch import Tensor, nn

from ..layers.drop_path import DropPath
from .attention import SerializedAttention
from .config import AttentionCfg, ConvImpl
from .sparse_conv import SubmanifoldConv3d
from .state import StageState

NormFactory = Callable[[int], nn.Module]
ActFactory = Callable[[], nn.Module]


class MLP(nn.Module):
    """``fc1 -> act -> drop -> fc2 -> drop``, optionally preceded by ``norm_pre -> act_pre``."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | None = None,
        out_channels: int | None = None,
        act_layer: ActFactory = nn.GELU,
        drop: float = 0.0,
        pre_norm_act: tuple[NormFactory, ActFactory] | None = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)
        self.norm_pre = self.act_pre = None
        if pre_norm_act is not None:
            norm_pre, act_pre = pre_norm_act
            self.norm_pre, self.act_pre = norm_pre(in_channels), act_pre()

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_pre is not None:
            x = self.act_pre(self.norm_pre(x))
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class ConditionalPositionEncoding(nn.Module):
    """``SubmanifoldConv3d(k=3) -> Linear -> norm`` on the stage's voxel grid."""

    def __init__(
        self,
        channels: int,
        norm_layer: NormFactory,
        conv_impl: ConvImpl = "torch",
        indice_key: str | None = None,
    ):
        super().__init__()
        self.conv = SubmanifoldConv3d(
            channels, channels, kernel_size=3, bias=True, impl=conv_impl, indice_key=indice_key
        )
        self.linear = nn.Linear(channels, channels)
        self.norm = norm_layer(channels)

    def forward(self, feat: Tensor, state: StageState) -> Tensor:
        table = state.sparse() if self.conv.impl == "spconv" else state.neighbor_table(3)
        return self.norm(self.linear(self.conv(feat, table)))


class Block(nn.Module):
    """One PTv3 block. ``conv=False`` swaps the CPE for a LayerNorm (``norm0``);
    ``attn=False`` skips the attention + MLP branch."""

    def __init__(
        self,
        channels: int,
        heads: int,
        patch_size: int,
        order_index: int,
        cfg: AttentionCfg,
        *,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: NormFactory = nn.LayerNorm,
        act_layer: ActFactory = nn.GELU,
        pre_norm: bool = True,
        conv: bool = True,
        attn: bool = True,
        conv_impl: ConvImpl = "torch",
        indice_key: str | None = None,
        stage_grid_size: float = 1.0,
        checkpoint: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.enable_conv = conv
        self.enable_attn = attn
        self.checkpoint = checkpoint
        if conv:
            self.cpe = ConditionalPositionEncoding(channels, norm_layer, conv_impl, indice_key)
        else:
            self.norm0 = norm_layer(channels)
        if attn:
            self.norm1 = norm_layer(channels)
            self.attn = SerializedAttention(
                channels, heads, patch_size, order_index, cfg, stage_grid_size=stage_grid_size
            )
            self.norm2 = norm_layer(channels)
            self.mlp = MLP(
                channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=cfg.proj_drop,
            )
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _forward_impl(self, feat: Tensor, state: StageState) -> Tensor:
        if self.enable_conv:
            feat = feat + self.cpe(feat, state)
        else:
            feat = self.norm0(feat)
        if self.enable_attn:
            shortcut = feat
            x = self.norm1(feat) if self.pre_norm else feat
            feat = shortcut + self.drop_path(self.attn(x, state))
            if not self.pre_norm:
                feat = self.norm1(feat)
            shortcut = feat
            x = self.norm2(feat) if self.pre_norm else feat
            feat = shortcut + self.drop_path(self.mlp(x))
            if not self.pre_norm:
                feat = self.norm2(feat)
        return feat

    def forward(self, state: StageState) -> StageState:
        feat = state.points.feat
        if self.checkpoint and self.training:
            feat = torch.utils.checkpoint.checkpoint(
                self._forward_impl, feat, state, use_reentrant=False
            )
        else:
            feat = self._forward_impl(feat, state)
        return state.with_feat(feat)
