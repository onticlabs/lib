"""Unit tests for ontic_lib.metrics3d on synthetic particle clouds / frames."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ontic_lib import metrics3d as m3


def _two_cluster_cloud(n_obj=200, n_plate=800, gap=0.5, seed=0):
    """Synthetic scene: a flat plate slab at z~0 and a compact object above it."""
    g = torch.Generator().manual_seed(seed)
    plate = torch.rand(n_plate, 3, generator=g) * torch.tensor([1.0, 1.0, 0.05])
    obj = torch.rand(n_obj, 3, generator=g) * 0.2 + torch.tensor([0.4, 0.4, 0.05 + gap])
    return torch.cat([plate, obj]), n_plate


def test_otsu_zcut_separates_bimodal():
    pos, n_plate = _two_cluster_cloud()
    zcut = m3.otsu_zcut(pos[:, 2])
    # the cut separates the modes: (almost) all plate below, all object above
    assert float((pos[:n_plate, 2] <= zcut).float().mean()) > 0.9
    assert bool((pos[n_plate:, 2] > zcut).all())


def test_split_obj_plate_zgap():
    pos, n_plate = _two_cluster_cloud()
    mask = torch.ones(pos.shape[0], dtype=torch.bool)
    obj, plate = m3.split_obj_plate(pos, mask)
    assert obj is not None
    assert bool(obj[n_plate:].all()) and not bool(obj[:n_plate].any())
    assert bool(plate[:n_plate].all()) and not bool(plate[n_plate:].any())


def test_split_obj_plate_too_few_particles():
    pos = torch.rand(8, 3)
    obj, plate = m3.split_obj_plate(pos, torch.ones(8, dtype=torch.bool))
    assert obj is None and plate is None


def test_cluster_obj_plate_3d_clean_scene():
    pos, n_plate = _two_cluster_cloud(gap=0.5)
    seg = torch.ones(pos.shape[0], dtype=torch.bool)
    obj, plate, clean, info = m3.cluster_obj_plate_3d(pos, seg)
    assert clean, info
    # (almost) every particle lands in its own cluster, none in the other's
    assert float(obj[n_plate:].float().mean()) > 0.95 and not bool(obj[:n_plate].any())
    assert float(plate[:n_plate].float().mean()) > 0.95 and not bool(plate[n_plate:].any())
    assert info["vgap"] >= 0.18


def test_cluster_obj_plate_3d_touching_not_clean():
    pos, _ = _two_cluster_cloud(gap=0.02)  # object nearly touching the plate
    seg = torch.ones(pos.shape[0], dtype=torch.bool)
    _, _, clean, info = m3.cluster_obj_plate_3d(pos, seg)
    assert not clean


def test_radius_components_two_blobs():
    a = torch.rand(100, 3) * 0.1
    b = torch.rand(120, 3) * 0.1 + 2.0
    comps = m3.radius_components(torch.cat([a, b]), eps=0.15, min_size=10)
    assert len(comps) == 2
    assert comps[0].numel() == 120 and comps[1].numel() == 100  # largest first


def test_dissolution_ratio():
    base = torch.randn(50, 3)
    coherent = torch.stack([base + t * 0.01 for t in range(10)])  # rigid drift
    diffusing = torch.stack([base * (1 + 0.3 * t) for t in range(10)])  # expanding
    assert m3.dissolution(coherent) == pytest.approx(1.0, abs=0.05)
    assert m3.dissolution(diffusing) > 2.0


def test_psnr_and_u8_roundtrip():
    x = torch.rand(4, 3, 8, 8)
    u = m3.to_u8(x)
    assert u.shape == (4, 8, 8, 3) and u.dtype == np.uint8
    p_same = m3.psnr_per_step(u, u)
    assert (p_same >= 90).all()  # identical frames -> capped ~100 dB
    p_diff = m3.psnr_per_step(u, 255 - u)
    assert (p_diff < 20).all()


def test_render_dissolution_expanding_footprint():
    T, H, W = 60, 32, 32
    alpha = torch.zeros(T, H, W)
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    d2 = (yy - H / 2) ** 2 + (xx - W / 2) ** 2
    for t in range(T):
        r = 3 + 8 * t / T  # footprint grows over time
        alpha[t] = (d2 <= r * r).float()
    ratio = m3.render_dissolution(alpha, n_state=4, early_w=6, late_lo=30, late_hi=50)
    assert ratio > 1.5
    # constant footprint -> ~1
    const = alpha[:1].repeat(T, 1, 1)
    assert m3.render_dissolution(const, n_state=4) == pytest.approx(1.0, abs=0.05)
