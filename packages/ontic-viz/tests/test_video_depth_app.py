"""Video depth selection feeds preview, playback and tracking geometry."""

import pytest
import torch
import viser

from ontic_viz.backbone_viewer.app import BackboneViewer
from ontic_viz.backbone_viewer.recording import save_recording
from ontic_viz.backbone_viewer.video_depth import VIDEO_METRIC


@pytest.fixture
def app(tmp_path):
    server = viser.ViserServer(port=0, verbose=False)
    app = BackboneViewer(
        server,
        device="cpu",
        roots={},
        presets_path=tmp_path / "presets.json",
        datasets=["demo"],
        backbones=["da3"],
        metric_models=["da3"],
    )
    try:
        app.load_demo()
        yield app
    finally:
        app.close()
        server.stop()


@pytest.mark.parametrize(
    "label,name",
    [
        ("Video Depth Anything Base", "vda_base"),
        ("VeloDepth", "velodepth"),
        ("DA3 Nested Giant-Large 1.1 (joint clip)", "da3_nested"),
    ],
)
def test_video_controls_preview_cache_and_recording_restore(app, tmp_path, label, name):
    panel = app.tracking
    panel.geometry.value = VIDEO_METRIC
    panel.video_model.value = label
    panel._on_video_model_change()
    panel.video_resolution_level.value = 3
    panel.length.value = 3
    panel.resolution.value = 128
    panel.update_depth_summary()
    assert panel.video_checkpoint.visible and panel.video_download.visible
    assert label in panel.depth_summary.content
    assert panel.video_resolution_level.visible == (name == "velodepth")
    assert panel.video_input_size.visible == (name != "velodepth")
    if name == "da3_nested":
        assert panel.video_input_size.value == 504
        assert panel.video_input_size.label == "DA3 long side"
    assert "temporal context" in panel.geometry_hint.content
    calls = []

    def builder(settings, device):
        calls.append(settings)
        return lambda images, **kwargs: torch.full((*images.shape[:3], *images.shape[-2:]), 2.0)

    def forbidden(*args, **kwargs):
        pytest.fail("Video preview must not load or configure an image backbone")

    app._configure_backbone = app.runner.load = forbidden
    panel.runner.video_builder = builder
    panel.start_job(background=False, track=False)
    clip = panel.preview
    assert clip is not None, panel.message.content
    assert calls[0].name == name and not calls[0].allow_download
    assert calls[0].resolution_level == 3
    assert name in panel.history_picker.value
    panel.seek_clip(2)
    assert (app._result.depth == 2).all()
    panel.start_job(background=False, track=False)
    assert len(calls) == 1  # geometry cache, no video inference on scrub/reuse
    path = save_recording(clip, tmp_path / "video.viewer.npz")
    panel.video_model.value = "Video Depth Anything Small"
    app._open_recording(path)
    assert panel.video_model.value == label
    assert panel.video_resolution_level.value == 3
    assert panel.geometry.value == VIDEO_METRIC and not panel.video_download.value


def test_video_model_switch_preserves_own_files_and_resolution(app):
    panel = app.tracking
    panel.geometry.value = VIDEO_METRIC
    panel.video_checkpoint.value = "/local/vda-small.pth"
    panel.video_repo.value = "/local/vda"
    panel.video_input_size.value = 280
    panel.video_download.value = True
    panel.video_model.value = "VeloDepth"
    panel._on_video_model_change()
    assert panel.video_checkpoint.value == panel.video_repo.value == ""
    assert not panel.video_download.value
    panel.video_checkpoint.value = "/local/velo-snapshot"
    panel.video_resolution_level.value = 4
    panel.video_model.value = "Video Depth Anything Small"
    panel._on_video_model_change()
    assert panel.video_checkpoint.value == "/local/vda-small.pth"
    assert panel.video_input_size.value == 280 and panel.video_download.value
    panel.video_model.value = "VeloDepth"
    panel._on_video_model_change()
    assert panel.video_checkpoint.value == "/local/velo-snapshot"
    assert panel.video_resolution_level.value == 4
    panel.geometry.value = "Sensor depth"
    panel.update_depth_summary()
    assert not panel.video_resolution_level.visible and not panel.video_input_size.visible


def test_video_resolution_defaults_come_from_model_config_and_can_be_restored(app, monkeypatch):
    from dataclasses import replace
    from ontic_nn.video_depth import VIDEO_DEPTH_MODELS

    config = VIDEO_DEPTH_MODELS["da3_nested"]
    monkeypatch.setitem(VIDEO_DEPTH_MODELS, "da3_nested", lambda: replace(config(), input_size=560))
    panel = app.tracking
    panel.video_model.value = "DA3 Nested Giant-Large 1.1 (joint clip)"
    panel._on_video_model_change()
    assert panel.video_input_size.value == 560
    assert panel.video_input_size.label == "DA3 long side"
    panel.video_input_size.value = 280
    panel._reset_video_resolution()
    assert panel.video_input_size.value == 560
    panel.video_model.value = "Video Depth Anything Small"
    panel._on_video_model_change()
    assert panel.video_input_size.value == VIDEO_DEPTH_MODELS["vda_small"]().input_size
    assert panel.video_input_size.label == "VDA short side"
