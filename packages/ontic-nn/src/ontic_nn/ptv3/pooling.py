"""Pooling (down-sampling) and unpooling between PTv3 stages."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import Tensor, nn

from ontic_lib.pointops import (
    SpaceFillingOrder,
    batch_to_offset,
    cluster_reduce,
    code_clusters,
    grid_clusters,
    pool_serialization,
    serialize,
)
from ontic_lib.pointops.grid import Clusters
from ontic_lib.structures import PointBatch

from .state import PoolRecord, StageState

NormFactory = Callable[[int], nn.Module]
ActFactory = Callable[[], nn.Module]


class LinearNormAct(nn.Module):
    """``Linear`` followed by optional norm and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: NormFactory | None = None,
        act_layer: ActFactory | None = None,
    ):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


def _pool_extras(extras: dict[str, Tensor], clusters: Clusters) -> dict[str, Tensor]:
    """bool -> any, floating -> mean, integer -> value of the cluster head."""
    out = {}
    m = clusters.num_clusters
    for name, value in extras.items():
        if value.dtype == torch.bool:
            out[name] = cluster_reduce(value, clusters.cluster, m, "any")
        elif value.is_floating_point():
            out[name] = cluster_reduce(value, clusters.cluster, m, "mean")
        else:
            out[name] = value[clusters.head]
    return out


class _Pooling(nn.Module):
    """Shared projection/reduction of :class:`SerializedPooling` and :class:`GridPooling`."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        norm_layer: NormFactory | None,
        act_layer: ActFactory | None,
        reduce: str,
        shuffle_orders: bool,
        neighbor_attender: bool,
    ):
        super().__init__()
        if reduce not in ("sum", "mean", "min", "max"):
            raise ValueError(f"reduce must be sum/mean/min/max, got {reduce!r}")
        if stride != 2 ** (math.ceil(stride) - 1).bit_length():
            raise ValueError(f"stride must be a power of two, got {stride}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.neighbor_attender = neighbor_attender
        proj_in = in_channels + 4 if neighbor_attender else in_channels
        if neighbor_attender:
            self.weight_mlp = nn.Sequential(nn.Linear(proj_in, 8), nn.ReLU(), nn.Linear(8, 1))
        self.proj = nn.Linear(proj_in, out_channels)
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def _reduce(self, state: StageState, clusters: Clusters) -> tuple[Tensor, Tensor]:
        """Pooled ``(coord, feat)`` of every cluster."""
        coord, feat = state.points.coord, state.points.feat
        cluster, m = clusters.cluster, clusters.num_clusters
        new_coord = cluster_reduce(coord, cluster, m, "mean")
        if self.neighbor_attender:
            delta = new_coord[cluster] - coord
            feat_ij = torch.cat([delta, torch.norm(delta, dim=-1, keepdim=True), feat], dim=-1)
            logits = torch.clamp(self.weight_mlp(feat_ij), min=-10.0, max=10.0)
            exp_w = torch.exp(logits)
            sum_exp = torch.clamp(cluster_reduce(exp_w, cluster, m, "sum")[cluster], min=1e-8)
            new_feat = cluster_reduce(self.proj(feat_ij) * (exp_w / sum_exp), cluster, m, "sum")
        else:
            new_feat = cluster_reduce(self.proj(feat), cluster, m, self.reduce)
        if self.norm is not None:
            new_feat = self.norm(new_feat)
        if self.act is not None:
            new_feat = self.act(new_feat)
        return new_coord, new_feat

    def _pooled_state(
        self, state: StageState, clusters: Clusters, grid_coord: Tensor, serialization
    ) -> tuple[StageState, PoolRecord]:
        head = clusters.head
        new_coord, new_feat = self._reduce(state, clusters)
        points = state.points
        pooled = PointBatch(
            coord=new_coord,
            feat=new_feat,
            batch=points.batch[head],
            time=None if points.time is None else points.time[head],
            extras=_pool_extras(points.extras, clusters),
        )
        group = state.group[head]
        new_state = StageState(
            points=pooled,
            group=group,
            group_offset=batch_to_offset(group),
            grid_coord=grid_coord,
            serialization=serialization,
            time_depth=state.time_depth,
            time_shift=state.time_shift,
        )
        return new_state, PoolRecord(parent=state, cluster=clusters.cluster)


class SerializedPooling(_Pooling):
    """Pool rows sharing a serialization code truncated by ``log2(stride)`` bits per axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_layer: NormFactory | None = None,
        act_layer: ActFactory | None = None,
        reduce: str = "max",
        shuffle_orders: bool = True,
        neighbor_attender: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            stride,
            norm_layer,
            act_layer,
            reduce,
            shuffle_orders,
            neighbor_attender,
        )

    def forward(
        self, state: StageState, generator: torch.Generator | None = None
    ) -> tuple[StageState, PoolRecord]:
        s = state.require_serialization()
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > s.depth:
            pooling_depth = 0
        clusters = code_clusters(s.code[0] >> (pooling_depth * 3))
        serialization = pool_serialization(
            s, clusters.head, pooling_depth, shuffle=self.shuffle_orders, generator=generator
        )
        grid_coord = state.grid_coord[clusters.head] >> pooling_depth
        return self._pooled_state(state, clusters, grid_coord, serialization)


class GridPooling(_Pooling):
    """Pool rows sharing ``(group, grid_coord // stride)``; optionally re-serialize."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        norm_layer: NormFactory | None = None,
        act_layer: ActFactory | None = None,
        reduce: str = "max",
        shuffle_orders: bool = True,
        re_serialize: bool = False,
        orders: Sequence[SpaceFillingOrder] = ("z",),
        neighbor_attender: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            stride,
            norm_layer,
            act_layer,
            reduce,
            shuffle_orders,
            neighbor_attender,
        )
        self.re_serialize = re_serialize
        self.orders = tuple(orders)

    def forward(
        self, state: StageState, generator: torch.Generator | None = None
    ) -> tuple[StageState, PoolRecord]:
        grid_coord, clusters = grid_clusters(state.grid_coord, state.group, stride=self.stride)
        serialization = None
        if self.re_serialize:
            serialization = serialize(
                grid_coord,
                state.group[clusters.head],
                orders=self.orders,
                shuffle=self.shuffle_orders,
                generator=generator,
            )
        return self._pooled_state(state, clusters, grid_coord, serialization)


class Unpooling(nn.Module):
    """``parent.feat = proj_skip(parent.feat) + proj(feat)[cluster]``; returns the parent state."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_layer: NormFactory | None = None,
        act_layer: ActFactory | None = None,
    ):
        super().__init__()
        self.proj = LinearNormAct(in_channels, out_channels, norm_layer, act_layer)
        self.proj_skip = LinearNormAct(skip_channels, out_channels, norm_layer, act_layer)

    def forward(self, state: StageState, record: PoolRecord) -> StageState:
        parent = record.parent
        feat = self.proj_skip(parent.points.feat) + self.proj(state.points.feat)[record.cluster]
        return parent.with_feat(feat)
