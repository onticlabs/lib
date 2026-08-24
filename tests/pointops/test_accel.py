"""Tests for the opt-in CUDA kernel access in ontic_lib.pointops.accel."""

import pytest
import torch

from ontic_lib.pointops import accel, furthest_point_indices


def test_available_reports_all_three_keys():
    avail = accel.available()
    assert set(avail) == {"pointops", "point_rope", "point_serialization"}
    assert all(isinstance(v, bool) for v in avail.values())


def test_missing_extension_raises_cleanly(monkeypatch):
    monkeypatch.setattr(accel, "cuda_pointops", lambda: None)
    with pytest.raises(RuntimeError, match="install_cuda_ext"):
        accel.furthest_point_indices(torch.rand(8, 3), 4)


def test_cpu_tensor_rejected():
    if accel.cuda_pointops() is None:
        pytest.skip("pointops CUDA extension not installed")
    with pytest.raises(ValueError, match="CUDA tensor"):
        accel.furthest_point_indices(torch.rand(8, 3), 4)


needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="no CUDA device"
)


@needs_gpu
def test_cuda_fps_matches_reference_contract():
    if accel.cuda_pointops() is None:
        pytest.skip("pointops CUDA extension not installed")
    g = torch.Generator().manual_seed(0)
    points = torch.rand(500, 3, generator=g).cuda()
    idx = accel.furthest_point_indices(points, 32)
    assert idx.shape == (32,) and idx.dtype == torch.long
    assert idx[0].item() == 0
    assert idx.unique().numel() == 32
    # the CUDA selection spreads at least as well as a random subset
    ref = furthest_point_indices(points.cpu(), 32)

    def spread(i, p):
        return torch.cdist(p[i], p[i]).topk(2, largest=False).values[:, 1].mean()

    assert spread(idx.cpu(), points.cpu()) > 0.5 * spread(ref, points.cpu())


@needs_gpu
def test_cuda_serialization_orders_like_reference():
    if accel.cuda_serialization() is None:
        pytest.skip("point_serialization CUDA extension not installed")
    from ontic_lib.pointops.serialization import morton_encode as ref_morton

    g = torch.Generator().manual_seed(1)
    grid = torch.randint(0, 2**10, (400, 3), generator=g)
    cuda_codes = accel.morton_encode(grid.cuda()).cpu()
    ref_codes = ref_morton(grid)
    # same Morton definition -> identical sort order
    assert torch.equal(torch.argsort(cuda_codes), torch.argsort(ref_codes))
