"""Per-stage state carried through the PTv3 encoder/decoder."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from torch import Tensor

from ontic_lib.pointops import SpaceFillingOrder, Serialization, batch_to_offset, serialize
from ontic_lib.pointops.grid import voxel_coords
from ontic_lib.structures import PointBatch

from .sparse_conv import NeighborTable, build_neighbor_table, make_sparse_tensor
from .windows import WindowPadding, window_padding


def dense_groups(batch: Tensor, time: Tensor) -> Tensor:
    """Dense id of every distinct ``(batch, time)`` pair, ordered by batch then time."""
    key = batch.long() << 32 | time.long()
    _, group = torch.unique(key, sorted=True, return_inverse=True)
    return group


@dataclass
class StageState:
    """Points of one stage plus the derived, cached quantities the blocks need.

    ``group (N,)`` is the dense attention/convolution group id: the batch id,
    or the id of ``(batch, time >> time_shift)`` in temporal mode. ``time_depth``
    is the bit width of the input time index (0 when non-temporal).
    """

    points: PointBatch
    group: Tensor
    group_offset: Tensor
    grid_coord: Tensor
    serialization: Serialization | None
    time_depth: int = 0
    time_shift: int = 0
    neighbors: dict[int, NeighborTable] = field(default_factory=dict)
    windows: dict[int, WindowPadding] = field(default_factory=dict)
    rope_cache: tuple[Tensor, Tensor] | None = None
    sparse_tensor: Any = None

    @classmethod
    def from_points(
        cls,
        points: PointBatch,
        grid_size: float,
        orders: Sequence[SpaceFillingOrder],
        shuffle: bool = False,
        generator: torch.Generator | None = None,
        temporal: bool = False,
    ) -> StageState:
        grid_coord = voxel_coords(points.coord, grid_size)
        if temporal:
            if points.time is None:
                raise ValueError("temporal PointTransformerV3 needs PointBatch.time")
            time_max = int(points.time.max().item()) if len(points) else 0
            time_depth = max(time_max.bit_length(), 1)
            group = dense_groups(points.batch, points.time)
        else:
            time_depth = 0
            group = points.batch
        serialization = serialize(
            grid_coord, group, orders=orders, shuffle=shuffle, generator=generator
        )
        return cls(
            points=points,
            group=group,
            group_offset=batch_to_offset(group),
            grid_coord=grid_coord,
            serialization=serialization,
            time_depth=time_depth,
        )

    @property
    def feat(self) -> Tensor:
        return self.points.feat

    @property
    def num_groups(self) -> int:
        return int(self.group_offset.shape[0])

    @property
    def time_depth_current(self) -> int:
        return self.time_depth - self.time_shift

    def with_feat(self, feat: Tensor) -> StageState:
        """Same state with new features (caches shared)."""
        return dataclasses.replace(self, points=self.points.replace(feat=feat))

    def neighbor_table(self, kernel_size: int) -> NeighborTable:
        table = self.neighbors.get(kernel_size)
        if table is None:
            table = build_neighbor_table(self.grid_coord, self.group, kernel_size)
            self.neighbors[kernel_size] = table
        return table

    def sparse(self) -> Any:
        """Cached ``spconv.SparseConvTensor`` (features are replaced per call)."""
        if self.sparse_tensor is None:
            self.sparse_tensor = make_sparse_tensor(self.points.feat, self.grid_coord, self.group)
        return self.sparse_tensor

    def window(self, patch_size: int) -> WindowPadding:
        padding = self.windows.get(patch_size)
        if padding is None:
            padding = window_padding(self.group_offset, patch_size)
            self.windows[patch_size] = padding
        return padding

    def require_serialization(self) -> Serialization:
        if self.serialization is None:
            raise ValueError("this stage has no serialization (grid pooling without re-serialize)")
        return self.serialization


@dataclass
class PoolRecord:
    """What unpooling needs: the parent state and each parent row's cluster in the child."""

    parent: StageState
    cluster: Tensor


@dataclass
class MergeRecord:
    """What unmerging needs: the state before the time merge."""

    parent: StageState
