"""PointTransformerV3: serialized point transformer U-Net over :class:`PointBatch`."""

from __future__ import annotations

from functools import partial

import torch
from torch import Tensor, nn

from ontic_lib.structures import PointBatch

from .blocks import MLP, Block
from .config import PointTransformerV3Cfg
from .pooling import GridPooling, SerializedPooling, Unpooling
from .sparse_conv import SubmanifoldConv3d
from .state import MergeRecord, PoolRecord, StageState
from .temporal import Merger, Unmerger


class Stem(nn.Module):
    """``SubmanifoldConv3d(k=5) -> norm -> act``."""

    def __init__(self, in_channels, embed_channels, norm_layer, act_layer, conv_impl):
        super().__init__()
        self.conv = SubmanifoldConv3d(
            in_channels,
            embed_channels,
            kernel_size=5,
            bias=False,
            impl=conv_impl,
            indice_key="stem",
        )
        self.norm = norm_layer(embed_channels)
        self.act = act_layer()

    def forward(self, feat: Tensor, state: StageState) -> Tensor:
        table = state.sparse() if self.conv.impl == "spconv" else state.neighbor_table(5)
        return self.act(self.norm(self.conv(feat, table)))


class Embedding(nn.Module):
    def __init__(self, in_channels, embed_channels, norm_layer, act_layer, conv_impl="torch"):
        super().__init__()
        self.stem = Stem(in_channels, embed_channels, norm_layer, act_layer, conv_impl)

    def forward(self, state: StageState) -> StageState:
        return state.with_feat(self.stem(state.points.feat, state))


class EncoderStage(nn.Module):
    """Optional ``down`` (pooling) and ``merger``, then ``block0..block{depth-1}``."""

    def __init__(self, down: nn.Module | None, merger: Merger | None, blocks: list[Block]):
        super().__init__()
        self.down = down
        self.merger = merger
        self.depth = len(blocks)
        for i, block in enumerate(blocks):
            self.add_module(f"block{i}", block)

    def forward(
        self, state: StageState, generator: torch.Generator | None = None
    ) -> tuple[StageState, PoolRecord | None, MergeRecord | None]:
        pool_record = merge_record = None
        if self.down is not None:
            state, pool_record = self.down(state, generator=generator)
        if self.merger is not None:
            state, merge_record = self.merger(state)
        for i in range(self.depth):
            state = getattr(self, f"block{i}")(state)
        return state, pool_record, merge_record


class DecoderStage(nn.Module):
    """Optional ``unmerger``, ``up`` (unpooling), then ``block0..block{depth-1}``."""

    def __init__(self, unmerger: Unmerger | None, up: Unpooling, blocks: list[Block]):
        super().__init__()
        self.unmerger = unmerger
        self.up = up
        self.depth = len(blocks)
        for i, block in enumerate(blocks):
            self.add_module(f"block{i}", block)

    def forward(
        self, state: StageState, pool_record: PoolRecord, merge_record: MergeRecord | None
    ) -> StageState:
        if self.unmerger is not None:
            if merge_record is None:
                raise ValueError("decoder stage has an unmerger but no merge record")
            state = self.unmerger(state, merge_record)
        state = self.up(state, pool_record)
        for i in range(self.depth):
            state = getattr(self, f"block{i}")(state)
        return state


class PyramidDecoder(nn.Module):
    """Concatenate every encoder stage's features at full resolution, then a linear head."""

    def __init__(self, backbone_out_channels: int, out_channels: int):
        super().__init__()
        self.dynamics_head = nn.Linear(backbone_out_channels, out_channels)

    def forward(
        self, state: StageState, pools: list[PoolRecord], merges: list[MergeRecord]
    ) -> StageState:
        while pools:
            if merges:
                state = merges.pop().parent.with_feat(state.points.feat)
            record = pools.pop()
            parent = record.parent
            state = parent.with_feat(
                torch.cat([parent.points.feat, state.points.feat[record.cluster]], dim=-1)
            )
        return state.with_feat(self.dynamics_head(state.points.feat))


class PointTransformerV3(nn.Module):
    """Encoder/decoder point transformer. ``forward`` keeps the input point order."""

    def __init__(self, cfg: PointTransformerV3Cfg):
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        n = cfg.num_stages
        enc = cfg.encoder
        temporal = cfg.temporal

        if cfg.pooling.norm == "bn":
            pool_norm = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        else:
            pool_norm = partial(nn.LayerNorm, eps=1e-3)
        ln = nn.LayerNorm
        act = nn.GELU

        self.embedding = Embedding(cfg.in_channels, enc[0].channels, pool_norm, act, cfg.conv_impl)

        if cfg.attention.rope_stage_rescale:
            stage_grid_sizes = [cfg.grid_size]
            for s in range(1, n):
                stage_grid_sizes.append(stage_grid_sizes[-1] * cfg.pooling.strides[s - 1])
        else:
            stage_grid_sizes = [1.0] * n

        def make_blocks(stage, channels, drop_paths, indice_key, grid_size) -> list[Block]:
            return [
                Block(
                    channels,
                    stage.heads,
                    stage.patch_size,
                    order_index=i % len(cfg.orders),
                    cfg=cfg.attention,
                    mlp_ratio=cfg.mlp_ratio,
                    drop_path=drop_paths[i],
                    norm_layer=ln,
                    act_layer=act,
                    pre_norm=cfg.pre_norm,
                    conv=stage.conv,
                    attn=stage.attn,
                    conv_impl=cfg.conv_impl,
                    indice_key=indice_key,
                    stage_grid_size=grid_size,
                    checkpoint=cfg.gradient_checkpointing,
                )
                for i in range(stage.depth)
            ]

        enc_depths = [stage.depth for stage in enc]
        enc_drop = [x.item() for x in torch.linspace(0, cfg.drop_path, sum(enc_depths))]
        self.enc = nn.ModuleDict()
        for s in range(n):
            drop_paths = enc_drop[sum(enc_depths[:s]) : sum(enc_depths[: s + 1])]
            down = merger = None
            if s > 0:
                stride = cfg.pooling.strides[s - 1]
                if cfg.pooling.kind == "grid":
                    down = GridPooling(
                        enc[s - 1].channels,
                        enc[s].channels,
                        stride=stride,
                        norm_layer=pool_norm,
                        act_layer=act,
                        reduce=cfg.pooling.reduce,
                        shuffle_orders=cfg.shuffle_orders,
                        re_serialize=enc[s].attn,
                        orders=cfg.orders,
                        neighbor_attender=cfg.pooling.neighbor_attender,
                    )
                else:
                    down = SerializedPooling(
                        enc[s - 1].channels,
                        enc[s].channels,
                        stride=stride,
                        norm_layer=pool_norm,
                        act_layer=act,
                        reduce=cfg.pooling.reduce,
                        shuffle_orders=cfg.shuffle_orders,
                        neighbor_attender=cfg.pooling.neighbor_attender,
                    )
                if temporal is not None:
                    merger = Merger(
                        temporal.merge_window[s - 1],
                        enc[s].channels if temporal.time_embedding else None,
                    )
            blocks = make_blocks(
                enc[s], enc[s].channels, drop_paths, f"stage{s}", stage_grid_sizes[s]
            )
            self.enc[f"enc{s}"] = EncoderStage(down, merger, blocks)

        self.dec = nn.ModuleDict()
        if cfg.cls_mode:
            return
        if cfg.feature_pyramid:
            self.dec["pyramid_decoder"] = PyramidDecoder(
                sum(stage.channels for stage in enc), cfg.out_channels
            )
            return
        dec = cfg.decoder
        dec_depths = [stage.depth for stage in dec]
        dec_drop = [x.item() for x in torch.linspace(0, cfg.drop_path, sum(dec_depths))]
        dec_channels = [stage.channels for stage in dec] + [enc[-1].channels]
        for s in reversed(range(n - 1)):
            drop_paths = dec_drop[sum(dec_depths[:s]) : sum(dec_depths[: s + 1])]
            drop_paths.reverse()
            up = Unpooling(
                dec_channels[s + 1],
                enc[s].channels,
                dec_channels[s],
                norm_layer=pool_norm,
                act_layer=act,
            )
            blocks = make_blocks(
                dec[s], dec_channels[s], drop_paths, f"stage{s}", stage_grid_sizes[s]
            )
            unmerger = Unmerger() if temporal is not None else None
            self.dec[f"dec{s}"] = DecoderStage(unmerger, up, blocks)
        self.dec["head"] = MLP(
            dec_channels[0],
            hidden_channels=dec_channels[0],
            out_channels=cfg.out_channels,
            act_layer=nn.Identity,
            pre_norm_act=(ln, act),
        )

    def forward(
        self, points: PointBatch, *, generator: torch.Generator | None = None
    ) -> PointBatch:
        """``points.feat (N, in_channels) -> (N, out_channels)`` in the input row order.

        ``cls_mode`` returns the coarsest stage's points instead; ``generator``
        drives the serialization-order shuffles.
        """
        cfg = self.cfg
        state = StageState.from_points(
            points,
            cfg.grid_size,
            cfg.orders,
            shuffle=cfg.shuffle_orders,
            generator=generator,
            temporal=cfg.temporal is not None,
        )
        state = self.embedding(state)
        pools: list[PoolRecord] = []
        merges: list[MergeRecord] = []
        for s in range(cfg.num_stages):
            state, pool_record, merge_record = self.enc[f"enc{s}"](state, generator=generator)
            if pool_record is not None:
                pools.append(pool_record)
            if merge_record is not None:
                merges.append(merge_record)
        if cfg.cls_mode:
            return state.points
        if cfg.feature_pyramid:
            return self.dec["pyramid_decoder"](state, pools, merges).points
        for s in reversed(range(cfg.num_stages - 1)):
            merge_record = merges.pop() if merges else None
            state = self.dec[f"dec{s}"](state, pools.pop(), merge_record)
        return state.points.replace(feat=self.dec["head"](state.points.feat))
