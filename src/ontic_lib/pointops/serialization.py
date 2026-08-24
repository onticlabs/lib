"""Morton and Hilbert serialization for integer point grids."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from ._hilbert import decode as _hilbert_decode
from ._hilbert import encode as _hilbert_encode
from ._z_order import key2xyz as _morton_decode
from ._z_order import xyz2key as _morton_encode

SpaceFillingOrder = Literal["z", "z-trans", "hilbert", "hilbert-trans"]


def morton_encode(grid_coordinates: Tensor, *, depth: int = 16) -> Tensor:
    if grid_coordinates.ndim != 2 or grid_coordinates.shape[-1] != 3:
        raise ValueError(f"expected (N, 3) grid coordinates, got {grid_coordinates.shape}")
    if not 1 <= depth <= 16:
        raise ValueError(f"Morton depth must be in [1, 16], got {depth}")
    x, y, z = grid_coordinates.long().unbind(dim=-1)
    return _morton_encode(x, y, z, b=None, depth=depth)


def morton_decode(code: Tensor, *, depth: int = 16) -> Tensor:
    x, y, z, _ = _morton_decode(code, depth=depth)
    return torch.stack((x, y, z), dim=-1)


def hilbert_encode(grid_coordinates: Tensor, *, depth: int = 16) -> Tensor:
    if grid_coordinates.ndim != 2 or grid_coordinates.shape[-1] != 3:
        raise ValueError(f"expected (N, 3) grid coordinates, got {grid_coordinates.shape}")
    if not 1 <= depth <= 21:
        raise ValueError(f"Hilbert depth must be in [1, 21], got {depth}")
    return _hilbert_encode(grid_coordinates.long(), num_dims=3, num_bits=depth)


def hilbert_decode(code: Tensor, *, depth: int = 16) -> Tensor:
    return _hilbert_decode(code, num_dims=3, num_bits=depth)


def encode_grid(
    grid_coordinates: Tensor,
    *,
    batch: Tensor | None = None,
    depth: int = 16,
    order: SpaceFillingOrder = "z",
) -> Tensor:
    if order not in ("z", "z-trans", "hilbert", "hilbert-trans"):
        raise ValueError(f"unsupported space-filling order {order!r}")
    if grid_coordinates.ndim != 2 or grid_coordinates.shape[-1] not in (3, 4):
        raise ValueError(f"expected (N, 3) or (N, 4) coordinates, got {grid_coordinates.shape}")
    coordinates = grid_coordinates[:, :3]
    temporal_index = grid_coordinates[:, 3].long() if grid_coordinates.shape[-1] == 4 else None
    if order.endswith("-trans"):
        coordinates = coordinates[:, [1, 0, 2]]
    code = (
        morton_encode(coordinates, depth=depth)
        if order.startswith("z")
        else hilbert_encode(coordinates, depth=depth)
    )
    if temporal_index is not None:
        code = temporal_index << depth * 3 | code
    if batch is not None:
        code = batch.long() << (depth * 3 + 10) | code
    return code
