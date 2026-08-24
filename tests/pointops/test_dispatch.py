"""Tests for the impl="torch"|"cuda"|"auto" dispatch on pointops functions."""

import pytest
import torch

from ontic_lib.pointops import furthest_point_indices, space_filling_stride_indices
from ontic_lib.pointops.serialization import encode_grid, hilbert_encode, morton_encode


def _grid(n=64, depth=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2**depth, (n, 3), generator=g, dtype=torch.int64)


def test_invalid_impl_rejected():
    with pytest.raises(ValueError, match="impl must be"):
        morton_encode(_grid(), depth=8, impl="fast")
    with pytest.raises(ValueError, match="impl must be"):
        furthest_point_indices(torch.rand(8, 3), 4, impl="gpu")


def test_impl_cuda_requires_cuda_tensor():
    with pytest.raises(ValueError, match="CUDA tensor"):
        hilbert_encode(_grid(), depth=8, impl="cuda")
    with pytest.raises(ValueError, match="CUDA tensor"):
        furthest_point_indices(torch.rand(8, 3), 4, impl="cuda")


def test_auto_on_cpu_matches_torch():
    grid = _grid(depth=8)
    assert torch.equal(
        morton_encode(grid, depth=8, impl="auto"), morton_encode(grid, depth=8, impl="torch")
    )
    assert torch.equal(
        hilbert_encode(grid, depth=8, impl="auto"), hilbert_encode(grid, depth=8, impl="torch")
    )
    pts = torch.rand(64, 3, generator=torch.Generator().manual_seed(1))
    assert torch.equal(
        space_filling_stride_indices(pts, 4, 0.05, impl="auto"),
        space_filling_stride_indices(pts, 4, 0.05, impl="torch"),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_auto_on_gpu_bitexact_serialization():
    from ontic_lib.pointops import accel

    if accel.cuda_serialization() is None:
        pytest.skip("serialize_cuda not installed")
    grid = _grid(n=5000, depth=16).cuda()
    for fn in (morton_encode, hilbert_encode):
        assert torch.equal(
            fn(grid, depth=16, impl="auto").cpu(), fn(grid.cpu(), depth=16, impl="torch")
        )
    code = encode_grid(grid, depth=16, order="hilbert", impl="auto")
    assert torch.equal(code.cpu(), encode_grid(grid.cpu(), depth=16, order="hilbert", impl="torch"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_fps_impl_cuda_contract():
    from ontic_lib.pointops import accel

    if accel.cuda_pointops() is None:
        pytest.skip("pointops CUDA extension not installed")
    pts = torch.rand(512, 3, generator=torch.Generator().manual_seed(0)).cuda()
    idx = furthest_point_indices(pts, 64, impl="cuda")
    assert idx.shape == (64,) and idx.dtype == torch.long and idx[0].item() == 0
    assert idx.unique().numel() == 64
