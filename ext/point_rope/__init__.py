"""3D Rotary Position Embedding for point clouds.

Mirrors LitePT's `libs/pointrope/` layout:
  - `pointrope_torch.py`  — pure-PyTorch `Point3DRoPE` (continuous coord, autograd-traceable)
  - `pointrope_cuda.py`   — CUDA wrapper around the compiled `point_rope_cuda._C` extension
  - `kernels.cu` + `pointrope.cpp` — CUDA kernel and pybind11 bindings
  - `setup.py`            — build script (`pip install -e .` from this dir)

`Point3DRoPE` is always available (pure PyTorch). `apply_rope` is the
forward-only CUDA fast path; it's only exported when the compiled extension
was built. Construct `Point3DRoPE(..., use_cuda=True)` to opt the module's
`forward()` into the CUDA path when available; the `get_cos_sin` / `apply`
API stays pure PyTorch regardless and remains autograd-traceable through
`coord`.
"""

from .pointrope_torch import Point3DRoPE

# Auto-discover the locally-built `point_rope_cuda` package (produced by
# `setup.py build_ext --inplace` next to this file) so the CUDA fast path
# is available without a full `pip install -e .`. No-op if a properly
# installed extension is already on sys.path.
import os as _os
import sys as _sys
_ext_dir = _os.path.dirname(_os.path.abspath(__file__))
if _os.path.isdir(_os.path.join(_ext_dir, "point_rope_cuda")) and _ext_dir not in _sys.path:
    _sys.path.insert(0, _ext_dir)

try:
    from .pointrope_cuda import apply_rope  # noqa: F401

    _CUDA_AVAILABLE = True
except Exception as e:
    _CUDA_AVAILABLE = False
    _CUDA_IMPORT_ERROR = e

    def apply_rope(*args, **kwargs):  # type: ignore[no-redef]
        raise ImportError(
            f"point_rope CUDA extension not built: {type(_CUDA_IMPORT_ERROR).__name__}: "
            f"{_CUDA_IMPORT_ERROR}. Run `pip install -e fwomo_3d/libs/point_rope/` to build."
        )


__all__ = ["Point3DRoPE", "apply_rope"]
