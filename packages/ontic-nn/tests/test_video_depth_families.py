"""Temporal ordering, metric units and offline loading of the new clip adapters."""

from types import SimpleNamespace
import sys

import pytest
import torch

from ontic_nn.video_depth import DA3VideoConfig, VeloDepthConfig
from ontic_nn.video_depth import da3, velodepth
from ontic_nn.video_depth.common import research_checkout
from ontic_nn.wrappers.da3 import IMAGENET_MEAN, IMAGENET_STD


class FakeVelo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.calls = []

    @torch.no_grad()
    @torch.autocast("cpu", dtype=torch.bfloat16)
    def infer(self, frames):
        assert not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled("cpu")  # wrapper owns precision
        self.calls.append(frames.clone())
        return {"depth": frames[:, :1], "distance": frames[:, :1] + 100}


class FakeNested(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.da3_metric = torch.nn.Identity()
        self.calls = []
        self.is_metric = 1

    def forward(self, images, **kwargs):
        assert not torch.is_grad_enabled()
        assert not torch.is_autocast_enabled("cpu")
        self.calls.append(images.clone())
        # Recover the per-frame identity and simulate joint context in the output.
        values = images[:, :, 0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]
        return {"depth": values * 255 + values.mean(1, keepdim=True), "is_metric": self.is_metric}


@pytest.fixture(params=["velodepth", "da3_nested"])
def adapter(request, monkeypatch):
    if request.param == "velodepth":
        upstream = FakeVelo()
        factory = SimpleNamespace(from_pretrained=lambda path: upstream)
        monkeypatch.setattr(
            velodepth, "import_research_module", lambda *a, **k: SimpleNamespace(VeloDepth=factory)
        )
        monkeypatch.setattr(velodepth, "resolve_checkpoint", lambda *a, **k: "/local/snapshot")
        model = VeloDepthConfig(resolution_level=2).build()
        assert upstream.resolution_level == 2
    else:
        upstream = FakeNested()
        monkeypatch.setattr(da3, "load_da3", lambda *a, **k: SimpleNamespace(model=upstream))
        model = DA3VideoConfig(input_size=28).build()
    return request.param, model


def test_joint_ordered_context_isolated_per_batch_and_camera(adapter):
    name, model = adapter
    images = torch.empty(2, 3, 2, 3, 12, 20)
    for b in range(2):
        for v in range(2):
            images[b, :, v] = (20 + b * 50 + v * 10 + torch.arange(3))[:, None, None, None] / 255
    depth = model(images)
    expected = images[:, :, :, 0] * 255
    if name == "da3_nested":
        expected = expected + images[:, :, :, 0].mean(1, keepdim=True)
        assert all(call.shape == (1, 3, 3, 14, 28) for call in model.model.calls)
    else:
        assert all(call.shape == (3, 3, 12, 20) for call in model.model.calls)
    torch.testing.assert_close(depth, expected)
    assert depth.shape == (2, 3, 2, 12, 20) and depth.dtype == torch.float32
    assert depth.device.type == "cpu" and len(model.model.calls) == 4
    assert not next(model.parameters()).requires_grad and not model.model.training
    # A repeated call has fresh camera-local context.
    torch.testing.assert_close(model(images), depth)


def test_cancellation_and_invalid_input(adapter):
    _, model = adapter

    def cancel():
        if model.model.calls:
            raise InterruptedError("cancelled")

    for bad in (torch.ones(1, 3, 3, 4, 4), torch.full((1, 3, 1, 3, 4, 4), float("nan"))):
        with pytest.raises(ValueError, match="RGB"):
            model(bad)
    with pytest.raises(InterruptedError):
        model(torch.ones(1, 3, 2, 3, 4, 4), check_cancel=cancel)
    assert len(model.model.calls) == 1


def test_velodepth_offline_guard_covers_auxiliary_initializers(monkeypatch, tmp_path):
    requested = []

    def from_pretrained(path):
        torch.hub.download_url_to_file(
            "https://example.invalid/initializer.pt", str(tmp_path / "x")
        )
        return FakeVelo()

    monkeypatch.setattr(torch.hub, "download_url_to_file", lambda *a, **k: requested.append(a))
    monkeypatch.setattr(velodepth, "resolve_checkpoint", lambda *a, **k: str(tmp_path))
    monkeypatch.setattr(
        velodepth,
        "import_research_module",
        lambda *a, **k: SimpleNamespace(VeloDepth=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    with pytest.raises(RuntimeError, match="allow_download=False"):
        VeloDepthConfig().build()
    assert requested == []
    VeloDepthConfig(allow_download=True).build()
    assert len(requested) == 1


def test_da3_rejects_relative_checkpoint_and_output(monkeypatch):
    net = FakeNested()
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(model=net)

    monkeypatch.setattr(da3, "load_da3", load)
    model = DA3VideoConfig().build()
    assert calls[0][0][0] == "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    assert calls[0][1]["allow_download"] is False
    net.is_metric = 0
    with pytest.raises(ValueError, match="metric depth"):
        model(torch.ones(1, 2, 1, 3, 4, 4))
    del net.da3_metric
    with pytest.raises(ValueError, match="Nested checkpoint"):
        DA3VideoConfig().build()


@pytest.mark.parametrize(
    "package,relative,source",
    [
        ("velodepth", "velodepth/models/velodepth.py", "."),
        ("depth_anything_3", "src/depth_anything_3/api.py", "src"),
    ],
)
def test_explicit_checkout_search_root_and_conflict(
    tmp_path, monkeypatch, package, relative, source
):
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.touch()
    monkeypatch.delitem(sys.modules, package, raising=False)
    original = list(sys.path)
    with research_checkout(str(tmp_path), package, relative):
        assert sys.path[0] == str((tmp_path / source).resolve())
    assert sys.path == original
    monkeypatch.setitem(sys.modules, package, SimpleNamespace(__file__="/different/__init__.py"))
    with pytest.raises(ImportError, match="restart"):
        with research_checkout(str(tmp_path), package, relative):
            pytest.fail("conflicting source accepted")
