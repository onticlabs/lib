"""Video adapter contract without research packages or pretrained weights."""

from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

from ontic_nn.video_depth import VIDEO_DEPTH_MODELS, VideoDepthAnythingConfig
from ontic_nn.video_depth import vda


@pytest.fixture
def upstream(monkeypatch, tmp_path):
    class Model(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))
            self.kwargs, self.calls = kwargs, []

        def infer_video_depth(self, frames, fps, **kwargs):
            self.calls.append((frames.copy(), fps, kwargs))
            # Preserve time and camera identity in the returned depth.
            return frames[..., 0].astype(np.float32), fps

    monkeypatch.setattr(vda, "_import_vda", lambda path: SimpleNamespace(VideoDepthAnything=Model))
    checkpoint = tmp_path / "weights.pth"
    torch.save({"weight": torch.ones(())}, checkpoint)
    resolutions = []

    def resolve(*args, **kwargs):
        resolutions.append((args, kwargs))
        return str(checkpoint)

    monkeypatch.setattr(vda, "resolve_checkpoint", resolve)
    return resolutions


@pytest.mark.parametrize(
    "name,encoder,size",
    [
        ("vda_small", "vits", "Small"),
        ("vda_base", "vitb", "Base"),
        ("vda_large", "vitl", "Large"),
    ],
)
def test_metric_checkpoints_and_frozen_model(upstream, name, encoder, size):
    cfg = VIDEO_DEPTH_MODELS[name](allow_download=False, input_size=140)
    model = cfg.build()
    assert model.model.kwargs["encoder"] == encoder
    assert model.model.kwargs["metric"] is True
    assert not model.model.training and not next(model.parameters()).requires_grad
    args, kwargs = upstream[0]
    assert args[1] == f"depth-anything/Metric-Video-Depth-Anything-{size}"
    assert kwargs["filename"] == f"metric_video_depth_anything_{encoder}.pth"
    assert kwargs["allow_download"] is False


def test_full_videos_are_ordered_and_never_mix_cameras_or_batches(upstream):
    model = VideoDepthAnythingConfig().build()
    images = torch.empty(2, 5, 3, 3, 4, 6)
    for bi in range(2):
        for vi in range(3):
            images[bi, :, vi] = (1 + bi * 50 + vi * 10 + torch.arange(5))[:, None, None, None] / 255
    depth = model(images)
    assert depth.shape == (2, 5, 3, 4, 6) and depth.device.type == "cpu"
    assert len(model.model.calls) == 6
    torch.testing.assert_close(depth, (images[:, :, :, 0] * 255).round())
    for frames, fps, kwargs in model.model.calls:
        assert frames.shape == (5, 4, 6, 3) and frames.dtype == np.uint8
        assert fps == -1 and kwargs["fp32"] and kwargs["device"] == "cpu"


def test_cancellation_stops_before_next_camera(upstream):
    model = VideoDepthAnythingConfig().build()

    def cancel():
        if model.model.calls:
            raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        model(torch.ones(1, 3, 2, 3, 4, 6), check_cancel=cancel)
    assert len(model.model.calls) == 1


@pytest.mark.parametrize("value", [float("nan"), -0.1, 1.1])
def test_invalid_rgb_rejected(upstream, value):
    model = VideoDepthAnythingConfig().build()
    with pytest.raises(ValueError, match="RGB floats"):
        model(torch.full((1, 2, 1, 3, 4, 4), value))
    assert not model.model.calls


def test_bad_output_shape_rejected(upstream):
    model = VideoDepthAnythingConfig().build()
    model.model.infer_video_depth = lambda *a, **kw: (np.ones((1, 2, 2)), -1)
    with pytest.raises(ValueError, match="expected"):
        model(torch.ones(1, 2, 1, 3, 4, 4))


def test_config_and_missing_repo_errors(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="encoder"):
        VideoDepthAnythingConfig(encoder="bad")
    with pytest.raises(ValueError, match="multiple of 14"):
        VideoDepthAnythingConfig(input_size=15)
    with pytest.raises(ValueError, match="checkout"):
        VideoDepthAnythingConfig(repo_path=str(tmp_path)).build()
    monkeypatch.setitem(sys.modules, "video_depth_anything", None)
    with pytest.raises(ImportError, match="ontic-nn\\[video-depth\\]"):
        VideoDepthAnythingConfig().build()
