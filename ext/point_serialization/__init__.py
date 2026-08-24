"""CUDA fast-path for Morton (Z-order) / Hilbert serialization codes.

Vendored from https://github.com/ChristianSchott/point_serialization_cuda.

Build (in this repo's venv):

    pip install -e fwomo_3d/libs/point_serialization/
    # or, in-place build (no pip metadata):
    cd fwomo_3d/libs/point_serialization && python setup.py build_ext --inplace

Exports:
    HAS_CUDA_EXT        — bool, True if the compiled extension imported.
    morton_encode(coord)
    hilbert_encode(coord, num_bits)
    hilbert_encode_approx(coord, num_bits)

All three take a CUDA `(N, 3)` integer tensor of grid coords and return a
`(N,)` int64 tensor of codes (cast from the kernel's uint64 output). They
encode only — decoding stays in PyTorch upstream.
"""

from __future__ import annotations

import os as _os
import sys as _sys

import torch as _torch

# Auto-discover an in-place build sitting next to this file (mirrors point_rope).
_ext_dir = _os.path.dirname(_os.path.abspath(__file__))
if _ext_dir not in _sys.path:
    _sys.path.insert(0, _ext_dir)

try:
    import serialize_cuda as _ext  # type: ignore

    HAS_CUDA_EXT = True
    _IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - environment-dependent
    HAS_CUDA_EXT = False
    _IMPORT_ERROR = e
    _ext = None  # type: ignore


def _check_input(coord: _torch.Tensor) -> _torch.Tensor:
    assert coord.is_cuda, "point_serialization CUDA ext requires a CUDA tensor"
    assert coord.dim() == 2 and coord.size(-1) == 3, f"expected (N,3), got {tuple(coord.shape)}"
    return coord.contiguous()


def morton_encode(coord: _torch.Tensor) -> _torch.Tensor:
    """Z-order / Morton encode `(N,3)` int grid coords. Returns `(N,)` int64."""
    if not HAS_CUDA_EXT:
        raise ImportError(
            f"serialize_cuda not built: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}. "
            "Run `pip install -e fwomo_3d/libs/point_serialization/` to build."
        )
    coord = _check_input(coord)
    return _ext.morton_encode(coord).to(_torch.int64)


def hilbert_encode(coord: _torch.Tensor, num_bits: int) -> _torch.Tensor:
    """Exact Hilbert encode `(N,3)` int grid coords. Returns `(N,)` int64."""
    if not HAS_CUDA_EXT:
        raise ImportError(
            f"serialize_cuda not built: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}. "
            "Run `pip install -e fwomo_3d/libs/point_serialization/` to build."
        )
    coord = _check_input(coord)
    return _ext.hilbert_encode(coord, int(num_bits)).to(_torch.int64)


def hilbert_encode_approx(coord: _torch.Tensor, num_bits: int) -> _torch.Tensor:
    """Approximate (faster) Hilbert encode. Not bit-exact vs `hilbert_encode`."""
    if not HAS_CUDA_EXT:
        raise ImportError(
            f"serialize_cuda not built: {type(_IMPORT_ERROR).__name__}: {_IMPORT_ERROR}. "
            "Run `pip install -e fwomo_3d/libs/point_serialization/` to build."
        )
    coord = _check_input(coord)
    return _ext.hilbert_encode_approx(coord, int(num_bits)).to(_torch.int64)


__all__ = [
    "HAS_CUDA_EXT",
    "morton_encode",
    "hilbert_encode",
    "hilbert_encode_approx",
]
