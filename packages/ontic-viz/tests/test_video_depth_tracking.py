"""Temporal geometry integration: calibration, caching, cancellation and playback."""

import dataclasses
from threading import Event

import pytest
import torch

from ontic_viz.backbone_viewer import video_depth

from ontic_viz.backbone_viewer.demo import DemoSource
from ontic_viz.backbone_viewer.recording import load_recording, recording_label, save_recording
from ontic_viz.backbone_viewer.tracking import ClipSpec, TrackingCancelled, TrackingRunner
from ontic_viz.backbone_viewer.video_depth import (
    VIDEO_METRIC,
    VIDEO_SENSOR_SCALE,
    VideoDepthSettings,
)


class Source(DemoSource):
    height, width = 24, 32

    def get_frame(self, traj, t, with_depth=False):
        frame = super().get_frame(traj, t, with_depth)
        # Sensor calibration changes over time; video predictions below stay constant.
        frame.depth.fill_(1 + t)
        frame.depth[1] *= 2
        if t % 2:
            return dataclasses.replace(
                frame,
                images=frame.images.flip(0),
                depth=frame.depth.flip(0),
                extrinsics=frame.extrinsics.flip(0),
                intrinsics=frame.intrinsics.flip(0),
                cam_names=frame.cam_names[::-1],
            )
        return frame


def prepare(runner, source, **kwargs):
    return runner.prepare(source, 0, ClipSpec(length=3), ("left", "right"), **kwargs)


def test_clip_scale_is_constant_across_time_and_matches_camera_names():
    calls = []

    def builder(settings, device):
        def model(images, **kwargs):
            calls.append(images)
            return torch.full((*images.shape[:3], *images.shape[-2:]), 4.0)

        return model

    source = Source()
    runner = TrackingRunner(video_builder=builder)
    clip = prepare(runner, source, geometry_source=VIDEO_SENSOR_SCALE)
    assert len(calls) == 1 and calls[0].shape[:3] == (1, 3, 2)
    assert clip.confidence is None
    assert clip.geometry.provenance["per_camera_clip_depth_scale"] == [0.5, 1.0]
    assert clip.geometry.provenance["per_frame_depth_scale"] == []
    torch.testing.assert_close(clip.geometry.depth[0, :, 0], torch.full((3, 24, 32), 2.0))
    torch.testing.assert_close(clip.geometry.depth[0, :, 1], torch.full((3, 24, 32), 4.0))
    assert prepare(runner, source, geometry_source=VIDEO_SENSOR_SCALE) is clip
    assert len(calls) == 1
    other = prepare(
        runner,
        source,
        geometry_source=VIDEO_SENSOR_SCALE,
        video_settings=VideoDepthSettings(input_size=280),
    )
    assert other is not clip and len(calls) == 2


def test_metric_video_without_sensor_depth_and_recording_roundtrip(tmp_path):
    source = Source()
    get_frame = source.get_frame
    source.get_frame = lambda *a, **kw: dataclasses.replace(get_frame(*a, **kw), depth=None)
    runner = TrackingRunner(
        video_builder=lambda *a: (
            lambda images, **kw: torch.full((*images.shape[:3], *images.shape[-2:]), 3.0)
        )
    )
    clip = prepare(runner, source, geometry_source=VIDEO_METRIC)
    assert (clip.geometry.depth == 3).all()
    assert clip.geometry.units == ("meters",)
    path = save_recording(clip, tmp_path / "video.viewer.npz")
    restored = load_recording(path)
    torch.testing.assert_close(restored.geometry.depth, clip.geometry.depth)
    assert restored.geometry.provenance == clip.geometry.provenance
    assert "vda_small" in recording_label(path) and "Sensor depth" not in recording_label(path)
    with pytest.raises(ValueError, match="requires recorded depth"):
        prepare(runner, source, geometry_source=VIDEO_SENSOR_SCALE)
    assert runner.cached_clip is clip


def test_cancelled_video_does_not_replace_cached_geometry():
    source, cancel = Source(), Event()

    def model(images, **kwargs):
        cancel.set()
        kwargs["check_cancel"]()

    runner = TrackingRunner(video_builder=lambda *a: model)
    previous = prepare(runner, source)
    with pytest.raises(TrackingCancelled):
        prepare(runner, source, geometry_source=VIDEO_METRIC, cancel=cancel)
    assert runner.cached_clip is previous


@pytest.mark.parametrize("value", [0.0, float("nan")])
def test_all_invalid_video_frame_rejected(value):
    runner = TrackingRunner(
        video_builder=lambda *a: (
            lambda images, **kw: torch.full((*images.shape[:3], *images.shape[-2:]), value)
        )
    )
    with pytest.raises(ValueError, match="no valid predictions"):
        prepare(runner, Source(), geometry_source=VIDEO_METRIC)
    assert runner.cached_clip is None


@pytest.mark.parametrize("name", ["vda_small", "velodepth", "da3_nested"])
@pytest.mark.parametrize("input_size", [None, 280])
def test_builder_routes_native_resolution_settings(monkeypatch, name, input_size):
    calls = []

    class Config:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def build(self):
            return torch.nn.Identity()

    monkeypatch.setitem(video_depth.VIDEO_DEPTH_MODELS, name, Config)
    video_depth.build_video_depth(
        VideoDepthSettings(name=name, input_size=input_size, resolution_level=3), "cpu"
    )
    assert calls[0]["allow_download"] is False
    if name == "velodepth":
        assert calls[0]["resolution_level"] == 3 and "input_size" not in calls[0]
    elif input_size is not None:
        assert calls[0]["input_size"] == 280 and "resolution_level" not in calls[0]
    else:
        assert "input_size" not in calls[0]  # the selected config owns its default
