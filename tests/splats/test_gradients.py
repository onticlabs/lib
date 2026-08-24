"""Gradient-correctness tests: rasterizer kernels vs the pure-torch replica.

Wraps scripts/check_rasterize_gradients.py (the single source of truth for
the replica) as pytest cases. These are the authoritative gradient checks —
autograd against an exact reference implementation, no finite differences.
GPU-only; the cute case additionally needs a driver the CuTeDSL supports.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts"))

from tests.splats.test_cute import _cute_executable  # noqa: E402
from tests.splats.test_rendering import _has_gsplat  # noqa: E402


@pytest.mark.skipif(
    not (torch.cuda.is_available() and _has_gsplat()), reason="needs CUDA + gsplat"
)
def test_gsplat_rasterize_gradients_match_torch_replica():
    from check_rasterize_gradients import check_kernel

    assert check_kernel("gsplat")


@pytest.mark.skipif(
    not _cute_executable(), reason="needs CUDA + gsplat + CuTeDSL with a supported driver"
)
def test_cute_rasterize_gradients_match_torch_replica():
    from check_rasterize_gradients import check_kernel

    assert check_kernel("cute")
