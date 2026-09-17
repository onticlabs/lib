"""Time merging: fold ``window_size`` consecutive time steps into one attention group."""

from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn

from ontic_lib.pointops import batch_to_offset, reserialize

from .state import MergeRecord, StageState, dense_groups


class Merger(nn.Module):
    """Shift the time index right by ``log2(window_size)`` bits and regroup.

    Points are not moved; only ``group``, the serialization's group bits and
    (optionally) a learned relative-time embedding change. A no-op when the
    window is 1 or the time index has no bits left.
    """

    def __init__(self, window_size: int = 2, time_emb_dim: int | None = None):
        super().__init__()
        if window_size < 1 or window_size & (window_size - 1):
            raise ValueError(f"window_size must be a power of two, got {window_size}")
        self.window_size = window_size
        self.shift = int(math.log2(window_size))
        self.pos_emb = None
        if time_emb_dim is not None and window_size > 1:
            self.pos_emb = nn.Parameter(torch.zeros(window_size, time_emb_dim))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def forward(self, state: StageState) -> tuple[StageState, MergeRecord]:
        depth = state.time_depth_current
        if self.shift == 0 or depth == 0:
            return dataclasses.replace(state), MergeRecord(parent=state)
        if state.points.time is None:
            raise ValueError("Merger needs PointBatch.time")
        shift = min(self.shift, depth)
        time_cur = state.points.time >> state.time_shift
        group = dense_groups(state.points.batch, time_cur >> shift)

        feat = state.points.feat
        if self.pos_emb is not None:
            relative = time_cur & ((1 << shift) - 1)
            feat = feat + self.pos_emb[relative]

        serialization = state.serialization
        if serialization is not None:
            serialization = reserialize(serialization, group)
        merged = StageState(
            points=state.points.replace(feat=feat),
            group=group,
            group_offset=batch_to_offset(group),
            grid_coord=state.grid_coord,
            serialization=serialization,
            time_depth=state.time_depth,
            time_shift=state.time_shift + shift,
            rope_cache=state.rope_cache,
        )
        return merged, MergeRecord(parent=state)


class Unmerger(nn.Module):
    """Return the pre-merge state carrying the merged features (same rows)."""

    def forward(self, state: StageState, record: MergeRecord) -> StageState:
        return record.parent.with_feat(state.points.feat)
