"""Access to the CUDA extension kernels vendored under ``ext/``.

The pure-torch functions in :mod:`ontic_lib.pointops` are the reference
implementations; the kernels are separate installs
(``scripts/install_cuda_ext.sh``) that run only on CUDA tensors. Functions
with a kernel counterpart take an ``impl`` argument ("torch" | "cuda" |
"auto") resolved by :func:`resolve_impl`. Automatic routing (``"auto"``) is
enabled by default only where the kernel is measured bit-exact against the
reference (Morton/Hilbert codes); FPS defaults to ``"torch"`` because its
kernel is quality-identical but not index-stable on distance ties.
"""

from __future__ import annotations

import importlib
from functools import cache

import torch
from torch import Tensor


@cache
def _try_import(name: str):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError):
        return None


def cuda_pointops():
    """The CUDA ``pointops`` module (FPS, KNN/ball query, grouping), or None."""
    return _try_import("pointops")


def cuda_point_rope():
    """The CUDA ``point_rope_cuda`` module (rotary embeddings for point tokens), or None."""
    return _try_import("point_rope_cuda")


def cuda_serialization():
    """The CUDA ``serialize_cuda`` module (GPU Morton/Hilbert codes), or None."""
    return _try_import("serialize_cuda")


def available() -> dict[str, bool]:
    """Which of the three CUDA extensions are importable."""
    return {
        "pointops": cuda_pointops() is not None,
        "point_rope": cuda_point_rope() is not None,
        "point_serialization": cuda_serialization() is not None,
    }


def _require(module, name: str):
    if module is None:
        raise RuntimeError(
            f"CUDA extension {name!r} is not installed — build it with "
            "scripts/install_cuda_ext.sh"
        )
    return module


def resolve_impl(tensor: Tensor, impl: str, module, name: str) -> bool:
    """Resolve an ``impl`` argument to "use the CUDA kernel?" for one call.

    ``"torch"`` never routes; ``"cuda"`` demands a CUDA tensor and the
    installed extension (raising otherwise); ``"auto"`` routes exactly when
    both are available.
    """
    if impl not in ("torch", "cuda", "auto"):
        raise ValueError(f"impl must be 'torch', 'cuda', or 'auto', got {impl!r}")
    if impl == "torch":
        return False
    if impl == "cuda":
        if not tensor.is_cuda:
            raise ValueError(f"impl='cuda' requires a CUDA tensor for {name}")
        _require(module, name)
        return True
    return tensor.is_cuda and module is not None


def furthest_point_indices(points: Tensor, count: int) -> Tensor:
    """CUDA counterpart of :func:`ontic_lib.pointops.furthest_point_indices`.

    Same contract (one ``(N, 3)`` cloud in, ``(count,)`` long indices out,
    first index is 0) but computed by the ``pointops`` kernel; requires a
    contiguous CUDA tensor. Selection order can differ from the reference on
    distance ties.
    """
    ops = _require(cuda_pointops(), "pointops")
    if not points.is_cuda:
        raise ValueError("accel.furthest_point_indices requires a CUDA tensor")
    count = min(count, points.shape[0])
    offset = torch.tensor([points.shape[0]], dtype=torch.int32, device=points.device)
    new_offset = torch.tensor([count], dtype=torch.int32, device=points.device)
    return ops.farthest_point_sampling(points.contiguous(), offset, new_offset).long()


def morton_encode(grid_coordinates: Tensor) -> Tensor:
    """GPU Morton codes for non-negative integer ``(N, 3)`` grid coordinates."""
    ser = _require(cuda_serialization(), "point_serialization")
    return ser.morton_encode(grid_coordinates).to(torch.int64)


def hilbert_encode(grid_coordinates: Tensor, *, depth: int = 16) -> Tensor:
    """GPU Hilbert codes for non-negative integer ``(N, 3)`` grid coordinates.

    ``depth`` is the bits-per-axis, matching the reference
    :func:`ontic_lib.pointops.serialization.hilbert_encode` parameter. The
    kernel is measured bit-exact against the reference.
    """
    ser = _require(cuda_serialization(), "point_serialization")
    return ser.hilbert_encode(grid_coordinates, depth).to(torch.int64)
