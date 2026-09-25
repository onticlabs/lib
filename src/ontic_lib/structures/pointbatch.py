"""Packed batch of points with per-point features, group ids and optional time index."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..pointops.packing import (
    batch_to_offset,
    offset_to_counts,
    pack_padded,
    unpack_to_padded,
)


@dataclass
class PointBatch:
    """``N`` points of ``B`` groups in packed layout.

    ``coord (N, 3)`` float, ``feat (N, C)``, ``batch (N,) int64`` sorted
    ascending, optional ``time (N,) int64`` and per-point ``extras`` of shape
    ``(N, ...)``. ``offset`` follows the cumulative-count convention of
    :mod:`ontic_lib.pointops.packing`.
    """

    coord: Tensor
    feat: Tensor
    batch: Tensor
    time: Tensor | None = None
    extras: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.coord.ndim != 2 or self.coord.shape[-1] != 3:
            raise ValueError(f"coord must be (N, 3), got {tuple(self.coord.shape)}")
        if not self.coord.is_floating_point():
            raise ValueError(f"coord must be floating point, got {self.coord.dtype}")
        n = self.coord.shape[0]
        if self.feat.ndim != 2 or self.feat.shape[0] != n:
            raise ValueError(f"feat must be (N, C) with N={n}, got {tuple(self.feat.shape)}")
        if self.batch.shape != (n,) or self.batch.dtype != torch.int64:
            raise ValueError(
                f"batch must be (N,) int64 with N={n}, got {tuple(self.batch.shape)} "
                f"{self.batch.dtype}"
            )
        if n > 0 and bool((self.batch[1:] < self.batch[:-1]).any()):
            raise ValueError("batch must be sorted ascending")
        if n > 0 and bool(self.batch[0] < 0):
            raise ValueError("batch ids must be non-negative")
        if self.time is not None and (self.time.shape != (n,) or self.time.dtype != torch.int64):
            raise ValueError(
                f"time must be (N,) int64 with N={n}, got {tuple(self.time.shape)} "
                f"{self.time.dtype}"
            )
        for name, value in self.extras.items():
            if not isinstance(value, Tensor) or value.ndim < 1 or value.shape[0] != n:
                raise ValueError(f"extra {name!r} must be a tensor of shape (N, ...) with N={n}")

    def __len__(self) -> int:
        return int(self.coord.shape[0])

    @property
    def num_groups(self) -> int:
        return int(self.batch.max().item()) + 1 if len(self) > 0 else 0

    @property
    def offset(self) -> Tensor:
        return batch_to_offset(self.batch)

    @property
    def counts(self) -> Tensor:
        return offset_to_counts(self.offset)

    @property
    def device(self) -> torch.device:
        return self.coord.device

    def fields(self) -> dict[str, Tensor]:
        """Every per-point tensor by name (``coord``, ``feat``, ``time`` if set, extras)."""
        out = {"coord": self.coord, "feat": self.feat}
        if self.time is not None:
            out["time"] = self.time
        out.update(self.extras)
        return out

    @classmethod
    def from_padded(
        cls,
        coord: Tensor,
        feat: Tensor,
        mask: Tensor | None = None,
        extras: dict[str, Tensor] | None = None,
        time: Tensor | None = None,
    ) -> PointBatch:
        """Pack ``(B, K, ...)`` or ``(B, T, K, ...)`` padded tensors.

        4-D input sets ``batch = b`` and ``time = t`` unless ``time`` is given
        explicitly (``(B, T, K)`` int64). Points are ordered by ``b``, then
        ``t``, then slot ``k``.
        """
        if coord.ndim not in (3, 4) or coord.shape[-1] != 3:
            raise ValueError(f"coord must be (B, K, 3) or (B, T, K, 3), got {tuple(coord.shape)}")
        group_shape = coord.shape[:-1]
        if mask is None:
            mask = torch.ones(group_shape, dtype=torch.bool, device=coord.device)
        if mask.shape != group_shape:
            raise ValueError(f"mask must be {tuple(group_shape)}, got {tuple(mask.shape)}")
        extras = dict(extras or {})
        temporal = coord.ndim == 4
        if time is None and temporal:
            b, t, k = group_shape
            time = torch.arange(t, device=coord.device).view(1, t, 1).expand(b, t, k)
        if time is not None and time.shape != group_shape:
            raise ValueError(f"time must be {tuple(group_shape)}, got {tuple(time.shape)}")

        names = list(extras)
        tensors = [coord, feat] + [extras[n] for n in names] + ([time] if time is not None else [])
        group, packed = pack_padded(mask, *tensors)
        batch = group // group_shape[1] if temporal else group
        packed_time = packed[-1].to(torch.int64) if time is not None else None
        return cls(
            coord=packed[0],
            feat=packed[1],
            batch=batch,
            time=packed_time,
            extras=dict(zip(names, packed[2 : 2 + len(names)])),
        )

    def to_padded(self, *, pad_value: float = 0.0) -> tuple[dict[str, Tensor], Tensor]:
        """Inverse of :meth:`from_padded`.

        Returns ``(fields, mask)`` with ``(B, K, ...)`` tensors, or
        ``(B, T, K, ...)`` when ``time`` is set (``T = time.max() + 1``).
        Rows keep their relative order within a group.
        """
        num_batches = self.num_groups
        if self.time is None:
            group, num_groups = self.batch, num_batches
        else:
            num_times = int(self.time.max().item()) + 1 if len(self) > 0 else 0
            group = self.batch * num_times + self.time
            num_groups = num_batches * num_times
        fields: dict[str, Tensor] = {}
        mask = None
        for name, value in self.fields().items():
            padded, mask = unpack_to_padded(
                value, group, num_groups=num_groups, pad_value=pad_value
            )
            if self.time is not None:
                padded = padded.reshape(num_batches, num_times, *padded.shape[1:])
            fields[name] = padded
        if self.time is not None:
            mask = mask.reshape(num_batches, num_times, -1)
        return fields, mask

    def select(self, index_or_mask: Tensor) -> PointBatch:
        """Row subset (a bool mask or a sorted index tensor keeps the sorted order)."""
        return self.replace(
            **{name: value[index_or_mask] for name, value in self.fields().items()},
            batch=self.batch[index_or_mask],
        )

    def cat(self, other: PointBatch, *, batch_offset: int = 0) -> PointBatch:
        """Append ``other`` (its batch ids shifted by ``batch_offset``) and re-sort by batch."""
        if (self.time is None) != (other.time is None):
            raise ValueError("cannot concatenate a temporal and a non-temporal PointBatch")
        if set(self.extras) != set(other.extras):
            raise ValueError(f"extras differ: {sorted(self.extras)} vs {sorted(other.extras)}")
        batch = torch.cat([self.batch, other.batch + batch_offset])
        order = torch.argsort(batch, stable=True)
        mine, theirs = self.fields(), other.fields()
        merged = {name: torch.cat([mine[name], theirs[name]])[order] for name in mine}
        return self.replace(batch=batch[order], **merged)

    def replace(self, **changes: Any) -> PointBatch:
        """Return a copy with the named fields replaced (extras given by name are merged)."""
        known = {f.name for f in dataclasses.fields(self)}
        extras = dict(self.extras)
        for name in list(changes):
            if name not in known:
                extras[name] = changes.pop(name)
        return dataclasses.replace(self, extras=changes.pop("extras", extras), **changes)

    def _map(self, fn) -> PointBatch:
        return PointBatch(
            coord=fn(self.coord),
            feat=fn(self.feat),
            batch=fn(self.batch),
            time=None if self.time is None else fn(self.time),
            extras={k: fn(v) for k, v in self.extras.items()},
        )

    def to(self, *args: Any, **kwargs: Any) -> PointBatch:
        """Move/cast every tensor; non-floating fields only move device."""
        probe = torch.empty(0).to(*args, **kwargs)
        return self._map(
            lambda x: x.to(*args, **kwargs) if x.is_floating_point() else x.to(probe.device)
        )

    def detach(self) -> PointBatch:
        return self._map(lambda x: x.detach())

    def clone(self) -> PointBatch:
        return self._map(lambda x: x.clone())

    def __repr__(self) -> str:
        shapes = ", ".join(f"{k}={tuple(v.shape)}" for k, v in self.fields().items())
        return f"PointBatch(N={len(self)}, groups={self.num_groups}, {shapes})"
