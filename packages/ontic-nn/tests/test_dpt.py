"""Tests for ontic_nn.dpt heads on tiny inputs."""

import pytest
import torch

from ontic_nn.dpt import DPTHead, DualDPTHead
from ontic_nn.dpt.head import _interpolate

DIM, FEATS, OUT_CH, PATCH = 16, 16, (8, 8, 8, 8), 14


def _feats(b: int, s: int, h: int, w: int, extra_tokens: int = 0):
    torch.manual_seed(0)
    n = (h // PATCH) * (w // PATCH) + extra_tokens
    return [torch.randn(b, s, n, DIM) for _ in range(4)]


@pytest.mark.parametrize("chunk_size", [None, 1])
def test_dpt_head_shapes(chunk_size):
    head = DPTHead(
        DIM, patch_size=PATCH, output_dim=2, features=FEATS, out_channels=OUT_CH, use_sky_head=True
    ).eval()
    b, s, h, w = 1, 3, 28, 42
    out = head(_feats(b, s, h, w, extra_tokens=1), h, w, patch_start_idx=1, chunk_size=chunk_size)
    assert set(out) == {"depth", "depth_conf", "sky"}
    assert out["depth"].shape == (b, s, h, w)
    assert out["depth_conf"].shape == (b, s, h, w)
    assert out["sky"].shape == (b, s, h, w)
    assert (out["depth"] > 0).all() and (out["depth_conf"] > 1).all()


def test_dpt_head_single_channel_no_sky_and_down_ratio():
    head = DPTHead(
        DIM,
        patch_size=PATCH,
        output_dim=1,
        features=FEATS,
        out_channels=OUT_CH,
        use_sky_head=False,
        down_ratio=2,
        pos_embed=True,
        norm_type="layer",
    ).eval()
    b, s, h, w = 2, 1, 28, 28
    out = head(_feats(b, s, h, w), h, w, chunk_size=8)
    assert set(out) == {"depth"}
    assert out["depth"].shape == (b, s, h // 2, w // 2)


def test_dual_dpt_head_shapes():
    head = DualDPTHead(
        DIM, patch_size=PATCH, output_dim=2, features=FEATS, out_channels=OUT_CH
    ).eval()
    b, s, h, w = 1, 2, 28, 42
    out = head(_feats(b, s, h, w), h, w, chunk_size=1)
    assert set(out) == {"depth", "depth_conf", "ray", "ray_conf"}
    assert out["depth"].shape == (b, s, h, w)
    assert out["depth_conf"].shape == (b, s, h, w)
    ph, pw = h // PATCH, w // PATCH  # aux branch stays at the 8x patch grid
    assert out["ray"].shape == (b, s, 8 * ph, 8 * pw, 6)
    assert out["ray_conf"].shape == (b, s, 8 * ph, 8 * pw)


def test_dual_dpt_head_aux_levels_and_conv_num():
    head = DualDPTHead(
        DIM,
        patch_size=PATCH,
        features=FEATS,
        out_channels=OUT_CH,
        aux_pyramid_levels=2,
        aux_out1_conv_num=1,
    ).eval()
    assert len(head.scratch.output_conv1_aux) == 2
    out = head(_feats(1, 1, 28, 28), 28, 28)
    assert out["ray"].shape == (1, 1, 16, 16, 6)
    with pytest.raises(ValueError, match="aux_out1_conv_num"):
        DualDPTHead(DIM, features=FEATS, out_channels=OUT_CH, aux_out1_conv_num=2)


def test_interpolate_matches_functional():
    x = torch.randn(2, 3, 4, 5)
    a = _interpolate(x, scale_factor=2)
    b = torch.nn.functional.interpolate(x, size=(8, 10), mode="bilinear", align_corners=True)
    assert torch.equal(a, b)
    with pytest.raises(ValueError, match="size or scale_factor"):
        _interpolate(x)
