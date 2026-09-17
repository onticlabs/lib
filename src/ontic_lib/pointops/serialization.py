"""Morton and Hilbert serialization for integer point grids.

The encoders accept ``impl="torch" | "cuda" | "auto"`` (default ``"auto"``):
the vendored ``serialize_cuda`` kernels are measured bit-exact against these
references, so automatic routing on CUDA tensors cannot change results — it
is purely a speedup (~10x Morton, ~350-470x Hilbert; see
docs/pointops_benchmarks.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor

from ._hilbert import decode as _hilbert_decode
from ._hilbert import encode as _hilbert_encode
from ._z_order import key2xyz as _morton_decode
from ._z_order import xyz2key as _morton_encode

SpaceFillingOrder = Literal["z", "z-trans", "hilbert", "hilbert-trans"]
Impl = Literal["torch", "cuda", "auto"]


def _cuda_route(tensor: Tensor, impl: str):
    """Return the serialize_cuda module if this call should use it, else None."""
    from .accel import cuda_serialization, resolve_impl  # lazy: avoid import cycle

    module = cuda_serialization()
    return module if resolve_impl(tensor, impl, module, "point_serialization") else None


def morton_encode(grid_coordinates: Tensor, *, depth: int = 16, impl: Impl = "auto") -> Tensor:
    if grid_coordinates.ndim != 2 or grid_coordinates.shape[-1] != 3:
        raise ValueError(f"expected (N, 3) grid coordinates, got {grid_coordinates.shape}")
    if not 1 <= depth <= 16:
        raise ValueError(f"Morton depth must be in [1, 16], got {depth}")
    cuda = _cuda_route(grid_coordinates, impl)
    if cuda is not None:
        return cuda.morton_encode(grid_coordinates.to(torch.int32).contiguous()).to(torch.int64)
    x, y, z = grid_coordinates.long().unbind(dim=-1)
    return _morton_encode(x, y, z, b=None, depth=depth)


def morton_decode(code: Tensor, *, depth: int = 16) -> Tensor:
    x, y, z, _ = _morton_decode(code, depth=depth)
    return torch.stack((x, y, z), dim=-1)


def hilbert_encode(grid_coordinates: Tensor, *, depth: int = 16, impl: Impl = "auto") -> Tensor:
    if grid_coordinates.ndim != 2 or grid_coordinates.shape[-1] != 3:
        raise ValueError(f"expected (N, 3) grid coordinates, got {grid_coordinates.shape}")
    if not 1 <= depth <= 21:
        raise ValueError(f"Hilbert depth must be in [1, 21], got {depth}")
    cuda = _cuda_route(grid_coordinates, impl)
    if cuda is not None:
        return cuda.hilbert_encode(grid_coordinates.to(torch.int32).contiguous(), depth).to(
            torch.int64
        )
    return _hilbert_encode(grid_coordinates.long(), num_dims=3, num_bits=depth)


def hilbert_decode(code: Tensor, *, depth: int = 16) -> Tensor:
    return _hilbert_decode(code, num_dims=3, num_bits=depth)


def encode_grid(
    grid_coordinates: Tensor,
    *,
    batch: Tensor | None = None,
    depth: int = 16,
    order: SpaceFillingOrder = "z",
    impl: Impl = "auto",
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
        morton_encode(coordinates, depth=depth, impl=impl)
        if order.startswith("z")
        else hilbert_encode(coordinates, depth=depth, impl=impl)
    )
    if temporal_index is not None:
        code = temporal_index << depth * 3 | code
    if batch is not None:
        code = batch.long() << (depth * 3 + 10) | code
    return code


@dataclass(frozen=True)
class Serialization:
    """Space-filling-curve codes of one packed point set under ``k`` orders.

    ``code (k, N) int64`` with the group id in the high bits (see
    :func:`encode_grid`), ``order (k, N)`` the argsort of each code row and
    ``inverse (k, N)`` its inverse permutation. ``depth`` bits per axis.
    """

    depth: int
    code: Tensor
    order: Tensor
    inverse: Tensor

    @property
    def num_orders(self) -> int:
        return int(self.code.shape[0])


def _sort_codes(code: Tensor) -> tuple[Tensor, Tensor]:
    order = torch.argsort(code, dim=1, stable=True)
    inverse = torch.zeros_like(order).scatter_(
        dim=1,
        index=order,
        src=torch.arange(code.shape[1], device=order.device).repeat(code.shape[0], 1),
    )
    return order, inverse


def _shuffle_orders(
    depth: int, code: Tensor, order: Tensor, inverse: Tensor, generator: torch.Generator | None
) -> Serialization:
    perm = torch.randperm(code.shape[0], generator=generator).to(code.device)
    return Serialization(depth=depth, code=code[perm], order=order[perm], inverse=inverse[perm])


def serialize(
    grid_coord: Tensor,
    batch: Tensor,
    *,
    orders: Sequence[SpaceFillingOrder],
    depth: int | None = None,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
    impl: Impl = "auto",
) -> Serialization:
    """Serialize ``(N, 3)`` integer grid coordinates grouped by ``batch (N,)``.

    ``depth`` defaults to ``grid_coord.max().bit_length() + 1`` and must be
    at most 16 with ``depth * 3 + num_groups.bit_length() <= 63``. With
    ``shuffle`` the ``k`` orders are permuted (drawn from ``generator``).
    """
    if grid_coord.ndim != 2 or grid_coord.shape[-1] != 3:
        raise ValueError(f"expected (N, 3) grid coordinates, got {tuple(grid_coord.shape)}")
    if batch.shape != grid_coord.shape[:1]:
        raise ValueError(f"batch {tuple(batch.shape)} must be (N,) with N={grid_coord.shape[0]}")
    if len(orders) == 0:
        raise ValueError("at least one serialization order is required")
    if depth is None:
        depth = int(grid_coord.max()).bit_length() + 1 if grid_coord.shape[0] > 0 else 1
    num_groups = int(batch.max()) + 1 if batch.shape[0] > 0 else 0
    if depth * 3 + num_groups.bit_length() > 63:
        raise ValueError(f"depth={depth} with {num_groups} groups overflows a 63-bit code")
    if depth > 16:
        raise ValueError(f"serialization depth must be <= 16, got {depth}")
    code = torch.stack(
        [encode_grid(grid_coord, batch=batch, depth=depth, order=o, impl=impl) for o in orders]
    )
    order, inverse = _sort_codes(code)
    if shuffle:
        return _shuffle_orders(depth, code, order, inverse, generator)
    return Serialization(depth=depth, code=code, order=order, inverse=inverse)


def reserialize(s: Serialization, batch: Tensor) -> Serialization:
    """Replace the group bits of every code with ``batch`` and re-sort; spatial bits are kept."""
    shift = s.depth * 3 + 10
    spatial = s.code & ((1 << shift) - 1)
    code = spatial | (batch.long() << shift)
    order, inverse = _sort_codes(code)
    return Serialization(depth=s.depth, code=code, order=order, inverse=inverse)


def pool_serialization(
    s: Serialization,
    head: Tensor,
    pooling_depth: int,
    shuffle: bool = False,
    generator: torch.Generator | None = None,
) -> Serialization:
    """Keep the codes of the ``(M,)`` head rows, dropping ``pooling_depth`` bits per axis."""
    code = s.code[:, head] >> (pooling_depth * 3)
    order, inverse = _sort_codes(code)
    depth = s.depth - pooling_depth
    if shuffle:
        return _shuffle_orders(depth, code, order, inverse, generator)
    return Serialization(depth=depth, code=code, order=order, inverse=inverse)
