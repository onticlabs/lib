"""Viewer recording round trips preserve geometry, query identity and optional masks."""

import dataclasses
import numpy as np
import pytest
import torch

from ontic_viz.backbone_viewer.demo import DemoSource
from ontic_viz.backbone_viewer.recording import load_recording, save_recording
from ontic_viz.backbone_viewer.tracking import ClipSpec, TrackerSettings, TrackingRunner


def test_roundtrip_tracks_and_depth_preview(tmp_path):
    source = DemoSource()
    runner = TrackingRunner()
    clip = runner.prepare(
        source, 0, ClipSpec(start=3, length=3, stride=2, image_long_side=32), ("right",)
    )
    clip.confidence = torch.rand_like(clip.geometry.depth)
    clip.confidence[0, 0, 0, 0, 0] = float("nan")
    run = runner.run(clip, TrackerSettings(name="demo"), source=source, count=32)
    for value in (clip, run):
        path = save_recording(value, tmp_path / f"{type(value).__name__}.npz")
        with np.load(path, allow_pickle=False) as data:
            for field in data.files:
                assert not data[field].dtype.hasobject
        loaded = load_recording(path)
        actual = loaded.clip if value is run else loaded
        assert actual.indices == (3, 5, 7) and actual.camera_names == ("right",)
        assert actual.frames[0].cam_names == ["left", "right"]
        torch.testing.assert_close(actual.images, clip.images)
        torch.testing.assert_close(actual.geometry.depth, clip.geometry.depth)
        torch.testing.assert_close(actual.geometry.extrinsics, clip.geometry.extrinsics)
        torch.testing.assert_close(actual.confidence, clip.confidence, equal_nan=True)
        assert actual.geometry.provenance == clip.geometry.provenance
        if value is run:
            for field in dataclasses.fields(run.output):
                before, after = getattr(run.output, field.name), getattr(loaded.output, field.name)
                if isinstance(before, torch.Tensor):
                    torch.testing.assert_close(after, before)
            torch.testing.assert_close(loaded.queries.source_uv, run.queries.source_uv)
            np.testing.assert_array_equal(loaded.colors, run.colors)


def test_unknown_recording_version_rejected(tmp_path):
    path = tmp_path / "unknown.npz"
    np.savez(path, viewer_version=2)
    with pytest.raises(ValueError, match="Unsupported"):
        load_recording(path)
