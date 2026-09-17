"""Tests for ontic_nn.ppt (plain point transformer and kNN helpers)."""

import sys

import pytest
import torch

from ontic_nn.ppt import (
    MultiViewLowResAttention,
    PlainPointTransformer,
    knn_query,
    local_knn_query,
)


def _cloud(b: int, v: int, h: int, w: int, c: int):
    torch.manual_seed(0)
    n = b * v * h * w
    p = torch.randn(n, 3)
    x = torch.randn(n, c)
    offset = torch.arange(1, b + 1) * (v * h * w)
    return p, x, offset


def test_knn_query_torch_respects_groups_and_includes_self():
    torch.manual_seed(0)
    xyz = torch.randn(10, 3)
    offset = torch.tensor([4, 10])
    idx, dist = knn_query(3, xyz, offset)
    assert idx.shape == (10, 3) and idx.dtype == torch.long
    assert torch.equal(idx[:, 0], torch.arange(10))
    assert torch.all(dist[:, 0] == 0)
    assert (idx[:4] < 4).all() and (idx[4:] >= 4).all()
    d_full = torch.cdist(xyz, xyz)
    ref = d_full[4:, 4:].topk(3, largest=False).indices + 4
    assert torch.equal(idx[4:], ref)


def test_knn_query_pads_small_groups_and_rejects_bad_impl():
    xyz = torch.randn(5, 3)
    idx, _ = knn_query(4, xyz, torch.tensor([2, 5]))
    assert idx.shape == (5, 4)
    assert (idx[:2] < 2).all()
    with pytest.raises(ValueError, match="impl must be"):
        knn_query(2, xyz, torch.tensor([5]), impl="fast")


def test_knn_query_cuda_requires_pointops(monkeypatch):
    monkeypatch.setitem(sys.modules, "pointops", None)
    with pytest.raises(ImportError, match="install_cuda_ext.sh pointops"):
        knn_query(4, torch.randn(8, 3), torch.tensor([8]), impl="cuda")


def test_ppt_forward_shape_torch_impl():
    b, v, h, w, c = 2, 2, 4, 4, 16
    p, x, offset = _cloud(b, v, h, w, c)
    model = PlainPointTransformer(
        c, knn_samples=4, num_blocks=2, attn_proj_channels=8, mvattn_down_factor=2, impl="torch"
    ).eval()
    out, knn_idx = model(p, x, offset, b=b, v=v, h=h, w=w, return_knn_idx=True)
    assert out.shape == (b * v * h * w, c)
    assert knn_idx.shape == (b * v * h * w, 4)
    again = model(p, x, offset, b=b, v=v, h=h, w=w, knn_idx=knn_idx)
    assert torch.allclose(out, again)
    with pytest.raises(ValueError, match="points"):
        model(p, x, offset, b=b, v=v, h=h, w=w + 1)


def test_ppt_checkpointing_matches_plain_forward():
    b, v, h, w, c = 1, 2, 4, 4, 16
    p, x, offset = _cloud(b, v, h, w, c)
    model = PlainPointTransformer(c, knn_samples=4, num_blocks=1, mvattn_down_factor=4)
    ref = model(p, x, offset, b=b, v=v, h=h, w=w)
    model.use_checkpointing = True
    out = model(p, x.requires_grad_(), offset, b=b, v=v, h=h, w=w)
    assert torch.allclose(out, ref, atol=1e-6)
    out.sum().backward()
    assert x.grad is not None


def test_ppt_cuda_impl_without_pointops_raises(monkeypatch):
    monkeypatch.setitem(sys.modules, "pointops", None)
    b, v, h, w, c = 1, 1, 4, 4, 16
    p, x, offset = _cloud(b, v, h, w, c)
    model = PlainPointTransformer(c, knn_samples=4, num_blocks=1, impl="cuda")
    with pytest.raises(ImportError, match="install_cuda_ext.sh pointops"):
        model(p, x, offset, b=b, v=v, h=h, w=w)


def test_multiview_attention_cross_and_down8():
    torch.manual_seed(0)
    mv = MultiViewLowResAttention(16, down_factor=8).eval()
    x = torch.randn(1, 2 * 16 * 16, 16)
    assert mv(x, 2, 16, 16).shape == x.shape
    ctx = torch.randn(1, 3 * 16 * 16, 16)
    assert mv(x, 2, 16, 16, context=ctx).shape == x.shape
    with pytest.raises(ValueError, match="divisible"):
        mv(torch.randn(1, 2 * 12 * 12, 16), 2, 12, 12)


def test_local_knn_query_matches_brute_force_on_grid():
    torch.manual_seed(0)
    v, h, w = 2, 6, 6
    ys, xs = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    grid = torch.stack([xs, ys, torch.zeros_like(xs)], -1).float()
    pts = torch.cat([grid.reshape(-1, 3), grid.reshape(-1, 3) + torch.tensor([0.0, 0.0, 5.0])])
    cam_to_world = torch.eye(4).repeat(v, 1, 1)
    cam_to_world[1, 2, 3] = 5.0
    intrinsics = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]).repeat(v, 1, 1)
    idx = local_knn_query(4, pts, cam_to_world, intrinsics, v, h, w, spatial_radius=2)
    assert idx.shape == (v * h * w, 4) and idx.dtype == torch.long
    d = torch.cdist(pts, pts)
    d.fill_diagonal_(float("inf"))  # candidates exclude the point itself
    ref = d.topk(4, largest=False).values
    got = d.gather(1, idx).sort(dim=1).values
    assert torch.allclose(got, ref)
    assert (idx // (h * w) == torch.arange(v * h * w)[:, None] // (h * w)).all()
