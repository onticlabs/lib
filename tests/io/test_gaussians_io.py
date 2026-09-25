"""Tests for ontic_lib.io Gaussians serialization."""

import importlib.util
import sys

import pytest
import torch

from ontic_lib.io import load_gaussians, save_gaussians
from ontic_lib.structures import Gaussians


def _has(module):
    return importlib.util.find_spec(module) is not None


def _make(batch=(), n=6, *, extras=True, mask=True, seed=0):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*batch, n, *shape, generator=g)

    return Gaussians(
        means=rand(3),
        scales=rand(3) + 0.1,
        rotations=torch.nn.functional.normalize(rand(4) - 0.5, dim=-1),
        opacities=rand(1) * 0.98 + 0.01,
        harmonics=rand(3, 4),
        mask=(rand(1) > 0.3) if mask else None,
        extras={"feat": rand(7), "ids": torch.arange(n).expand(*batch, n)} if extras else {},
    )


def _assert_exact(a, b):
    assert a.batch_shape == b.batch_shape
    assert [k for k, _ in a.items()] == [k for k, _ in b.items()]
    for (name, x), (_, y) in zip(a.items(), b.items()):
        assert x.dtype == y.dtype, name
        assert torch.equal(x, y), name


def test_npz_round_trip_exact(tmp_path):
    gs = _make(batch=(2, 3))
    path = tmp_path / "g.npz"
    save_gaussians(path, gs, metadata={"source": "test"})
    _assert_exact(load_gaussians(path), gs)


def test_npz_round_trip_unbatched_minimal(tmp_path):
    gs = Gaussians(means=torch.randn(4, 3, dtype=torch.float64))
    save_gaussians(tmp_path / "g.npz", gs)
    _assert_exact(load_gaussians(tmp_path / "g.npz"), gs)


def test_reserved_metadata_key_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_gaussians(tmp_path / "g.npz", _make(), metadata={"batch_shape": "x"})


def test_unknown_suffix_raises(tmp_path):
    with pytest.raises(ValueError, match="unsupported suffix"):
        save_gaussians(tmp_path / "g.bin", _make())
    with pytest.raises(ValueError, match="unsupported suffix"):
        load_gaussians(tmp_path / "g.bin")


@pytest.mark.skipif(not _has("safetensors"), reason="safetensors not installed")
def test_safetensors_round_trip_exact(tmp_path):
    gs = _make(batch=(2,))
    path = tmp_path / "g.safetensors"
    save_gaussians(path, gs, metadata={"source": "test"})
    _assert_exact(load_gaussians(path), gs)


def test_safetensors_missing_gives_informative_import_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.torch", None)
    with pytest.raises(ImportError, match=r"ontic-lib\[safetensors\]"):
        save_gaussians(tmp_path / "g.safetensors", _make())


@pytest.mark.skipif(not _has("plyfile"), reason="plyfile not installed")
def test_ply_round_trip_unbatched_with_sh(tmp_path):
    gs = _make(extras=False, mask=False)
    path = tmp_path / "g.ply"
    save_gaussians(path, gs, metadata={"source": "test"})
    back = load_gaussians(path)
    assert back.batch_shape == () and back.mask is None and back.extras == {}
    for (name, x), (_, y) in zip(gs.items(), back.items()):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5, msg=name)
    # standard 3DGS vertex fields are present
    from plyfile import PlyData

    names = PlyData.read(path)["vertex"].data.dtype.names
    assert {"x", "y", "z", "f_dc_0", "f_rest_8", "opacity", "scale_2", "rot_3"} <= set(names)
    assert "f_rest_9" not in names


@pytest.mark.skipif(not _has("plyfile"), reason="plyfile not installed")
def test_ply_rejects_batched_and_extras(tmp_path):
    with pytest.raises(ValueError, match="unbatched"):
        save_gaussians(tmp_path / "g.ply", _make(batch=(2,), extras=False, mask=False))
    with pytest.raises(ValueError, match="mask"):
        save_gaussians(tmp_path / "g.ply", _make())


def test_ply_missing_gives_informative_import_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "plyfile", None)
    with pytest.raises(ImportError, match=r"ontic-lib\[ply\]"):
        save_gaussians(tmp_path / "g.ply", _make(extras=False, mask=False))
