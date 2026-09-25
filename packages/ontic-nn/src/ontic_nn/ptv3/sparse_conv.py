"""Submanifold sparse 3D convolution over voxelised point rows.

Rows are ``(N, C)`` features at integer ``grid_coord (N, 3)`` inside groups
``batch (N,)``; the output keeps the row set (submanifold). Kernel offsets are
enumerated row-major over ``(dx, dy, dz)`` in ``[-r, r]^3`` (spconv's order),
and the weight is ``(K**3, in, out)``: offset ``o`` reads the input row at
``grid_coord + offsets[o]`` and multiplies by ``weight[o]``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

ConvImpl = Literal["torch", "spconv"]

_COORD_BITS = 16  # 3 axes + group id fit in the 63 bits of an int64 key


def kernel_offsets(kernel_size: int) -> Tensor:
    """``(K**3, 3)`` int64 offsets, row-major over ``(dx, dy, dz)``; centre at index ``K**3 // 2``."""
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size must be odd and positive, got {kernel_size}")
    r = kernel_size // 2
    return torch.tensor(list(itertools.product(range(-r, r + 1), repeat=3)), dtype=torch.int64)


@dataclass(frozen=True)
class NeighborTable:
    """``index (K**3, N) int64``: row of the neighbour at each offset, ``-1`` where absent."""

    kernel_size: int
    index: Tensor


def _voxel_keys(grid_coord: Tensor, batch: Tensor) -> Tensor:
    grid = grid_coord.long()
    key = batch.long() << (3 * _COORD_BITS)
    return key | grid[:, 0] << (2 * _COORD_BITS) | grid[:, 1] << _COORD_BITS | grid[:, 2]


def build_neighbor_table(grid_coord: Tensor, batch: Tensor, kernel_size: int) -> NeighborTable:
    """Neighbour rows of every row for a ``kernel_size**3`` stencil.

    Duplicate voxels: the centre offset maps every row to itself; any other
    offset resolves to the lowest row index occupying that voxel.
    """
    n = grid_coord.shape[0]
    device = grid_coord.device
    if n > 0 and (grid_coord.min() < 0 or grid_coord.max() >= (1 << _COORD_BITS) - kernel_size):
        raise ValueError(f"grid coordinates must lie in [0, {(1 << _COORD_BITS) - kernel_size})")
    if n > 0 and batch.max() >= 1 << 15:
        raise ValueError("at most 2**15 groups are supported")
    offsets = kernel_offsets(kernel_size).to(device)
    keys = _voxel_keys(grid_coord, batch)
    unique, inverse = torch.unique(keys, sorted=True, return_inverse=True)
    first = torch.full((unique.shape[0],), n, dtype=torch.int64, device=device)
    first.scatter_reduce_(0, inverse, torch.arange(n, device=device), reduce="amin")

    grid = grid_coord.long()
    index = torch.empty(offsets.shape[0], n, dtype=torch.int64, device=device)
    centre = offsets.shape[0] // 2
    for o in range(offsets.shape[0]):
        if o == centre:
            index[o] = torch.arange(n, device=device)
            continue
        shifted = grid + offsets[o]
        inside = (shifted >= 0).all(-1) & (shifted < (1 << _COORD_BITS)).all(-1)
        query = _voxel_keys(shifted.clamp_min(0), batch)
        pos = torch.searchsorted(unique, query).clamp_max(max(unique.shape[0] - 1, 0))
        found = inside & (unique[pos] == query) if unique.shape[0] > 0 else inside & False
        index[o] = torch.where(found, first[pos], torch.full_like(pos, -1))
    return NeighborTable(kernel_size=kernel_size, index=index)


class SubmanifoldConv3d(nn.Module):
    """``(N, in) -> (N, out)`` submanifold convolution with ``weight (K**3, in, out)``.

    ``impl="torch"`` gathers neighbours from a :class:`NeighborTable`;
    ``impl="spconv"`` runs ``spconv.SubMConv3d`` on a cached
    ``SparseConvTensor`` (``indice_key`` shares index pairs within a stage).
    Both consume the same parameters.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool = True,
        impl: ConvImpl = "torch",
        indice_key: str | None = None,
    ):
        super().__init__()
        if impl not in ("torch", "spconv"):
            raise ValueError(f"impl must be 'torch' or 'spconv', got {impl!r}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.impl = impl
        self.indice_key = indice_key
        volume = kernel_size**3
        self.weight = nn.Parameter(torch.empty(volume, in_channels, out_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()
        self._spconv = None
        if impl == "spconv":
            self._spconv = self._make_spconv()

    def reset_parameters(self) -> None:
        bound = 1.0 / math.sqrt(self.kernel_size**3 * self.in_channels)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def _make_spconv(self):
        try:
            import spconv.pytorch as spconv
        except ImportError as e:
            raise ImportError(
                "SubmanifoldConv3d(impl='spconv') requires spconv; install ontic-nn[spconv]"
            ) from e
        conv = spconv.SubMConv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            bias=False,
            indice_key=self.indice_key,
        )
        conv.weight = None  # parameters live on this module in (K**3, in, out) layout
        return conv

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"bias={self.bias is not None}, impl={self.impl!r}"
        )

    def forward(self, feat: Tensor, table: Any) -> Tensor:
        """``table`` is a :class:`NeighborTable` (torch) or a ``SparseConvTensor`` (spconv)."""
        if self.impl == "spconv":
            return self._forward_spconv(feat, table)
        return self._forward_torch(feat, table)

    def _forward_torch(self, feat: Tensor, table: NeighborTable) -> Tensor:
        if table.kernel_size != self.kernel_size:
            raise ValueError(
                f"table built for kernel {table.kernel_size}, conv has {self.kernel_size}"
            )
        index = table.index
        centre = index.shape[0] // 2
        out = feat @ self.weight[centre]
        for o in range(index.shape[0]):
            if o == centre:
                continue
            idx = index[o]
            absent = idx < 0
            gathered = feat[idx.clamp_min(0)].masked_fill(absent.unsqueeze(-1), 0.0)
            out = out + gathered @ self.weight[o]
        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_spconv(self, feat: Tensor, sparse: Any) -> Tensor:
        k = self.kernel_size
        weight = self.weight.permute(2, 0, 1).reshape(self.out_channels, k, k, k, self.in_channels)
        x = sparse.replace_feature(feat)
        out = self._spconv._conv_forward(self.training, x, weight.contiguous(), None)
        # spconv stores the index pairs on the tensor it was given; keep them for later blocks.
        sparse.indice_dict = out.indice_dict
        result = out.features
        if self.bias is not None:
            result = result + self.bias
        return result


def make_sparse_tensor(feat: Tensor, grid_coord: Tensor, batch: Tensor, pad: int = 96) -> Any:
    """``spconv.SparseConvTensor`` over ``(batch, grid_coord)`` rows (spatial shape = max + pad)."""
    try:
        import spconv.pytorch as spconv
    except ImportError as e:
        raise ImportError("sparse tensors require spconv; install ontic-nn[spconv]") from e
    spatial_shape = (grid_coord.max(0).values + pad).tolist() if grid_coord.shape[0] else [pad] * 3
    indices = torch.cat([batch.unsqueeze(-1).int(), grid_coord.int()], dim=1).contiguous()
    batch_size = int(batch.max().item()) + 1 if batch.shape[0] else 1
    return spconv.SparseConvTensor(feat, indices, spatial_shape, batch_size)
