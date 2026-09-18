"""Tracking pipeline checks without model weights or GPU access."""

from __future__ import annotations

import dataclasses
import io
import json
from threading import Event

import numpy as np
import pytest
import torch

from ontic_nn.trackers.common import make_tracker_output, validate_tracker_inputs
from ontic_viz.backbone_viewer.demo import DemoSource
from ontic_viz.backbone_viewer.runner import (
    BackboneResult,
    aligned_depth_and_cameras,
)
from ontic_viz.backbone_viewer.render import PointFilters, build_point_cloud
from ontic_viz.backbone_viewer.tracking import (
    PREVIEW_ALIGNMENT,
    RIG_SCALE,
    SENSOR_SCALE,
    ClipSpec,
    TrackerSettings,
    TrackingCancelled,
    TrackingRunner,
    export_tracks,
    render_tracks,
    seed_queries,
    visible_query_mask,
    visible_track_samples,
)


class SmallSource(DemoSource):
    height, width = 24, 32

    def __init__(self):
        self.calls = []

    def get_frame(self, traj, t, with_depth=False):
        self.calls.append((traj, t))
        return super().get_frame(traj, t, with_depth)


@pytest.fixture
def source():
    return SmallSource()


def prepare(source, runner=None, **kwargs):
    return (runner or TrackingRunner()).prepare(
        source, 0, ClipSpec(length=12), ("left", "right"), **kwargs
    )


def test_clip_bounds():
    assert ClipSpec(start=2, length=3, stride=4).indices(11) == (2, 6, 10)
    for spec in [ClipSpec(start=-1), ClipSpec(length=1), ClipSpec(stride=0), ClipSpec(length=73)]:
        with pytest.raises(ValueError):
            spec.indices(72)


def test_cache_reuses_geometry_and_invalidates_on_camera_change(source):
    runner = TrackingRunner()
    clip = prepare(source, runner)
    assert len(source.calls) == 12
    assert prepare(source, runner) is clip
    runner.run(clip, TrackerSettings(name="demo"), source=source, count=32)
    runner.run(clip, TrackerSettings(name="demo"), source=source, count=64)
    assert len(source.calls) == 12
    different = runner.prepare(source, 0, ClipSpec(length=12), ("right",))
    assert different is not clip and len(source.calls) == 24
    # Cached frames retain all cameras for the camera-selection controls.
    assert different.frames[0].cam_names == ["left", "right"]
    assert different.images.shape[2] == 1


def test_camera_order_is_matched_by_name(source):
    get_frame = source.get_frame

    def reordered(traj, t, with_depth=False):
        f = get_frame(traj, t, with_depth)
        if t % 2:
            return dataclasses.replace(
                f,
                images=f.images.flip(0),
                intrinsics=f.intrinsics.flip(0),
                extrinsics=f.extrinsics.flip(0),
                depth=f.depth.flip(0),
                cam_names=f.cam_names[::-1],
            )
        return f

    source.get_frame = reordered
    clip = prepare(source)
    assert torch.all(clip.geometry.extrinsics[0, :, 0, 0, 3] == -0.3)
    assert torch.all(clip.geometry.extrinsics[0, :, 1, 0, 3] == 0.3)


class ScaledBackbone:
    def __init__(self):
        self.calls = self.releases = 0

    def set_long_side(self, value):
        self.long_side = value

    def load(self, name):
        self.name = name

    def release(self):
        self.releases += 1

    def run(self, frame, indices, **kwargs):
        scale = 2.0 + self.calls
        self.calls += 1
        poses = frame.extrinsics.clone()
        poses[:, :3, 3] *= scale
        return BackboneResult(
            frame.depth[:, 0] * scale,
            None,
            poses,
            frame.intrinsics,
            frame.extrinsics,
            frame.intrinsics,
        )


@pytest.mark.parametrize("policy", [SENSOR_SCALE, RIG_SCALE])
def test_backbone_scale_is_calibrated_before_tracking(source, policy):
    backbone = ScaledBackbone()
    clip = prepare(source, geometry_source=policy, backbone="fake", backbone_runner=backbone)
    expected = torch.stack([f.depth[:, 0] for f in clip.frames])[None]
    torch.testing.assert_close(clip.geometry.depth, expected)
    torch.testing.assert_close(clip.geometry.extrinsics[0, 0], clip.frames[0].extrinsics)
    np.testing.assert_allclose(
        clip.geometry.provenance["per_frame_depth_scale"], 1 / np.arange(2.0, 14.0)
    )
    assert backbone.releases == 1
    assert clip.geometry.units == ("meters",)


@pytest.mark.parametrize("alignment", ["sim3_points", "prescale_gt", "metric_mono"])
def test_tracking_preserves_preview_depth_cameras_and_intrinsics(source, alignment):
    backbone = ScaledBackbone()
    raw_run = backbone.run
    results = []

    def run(frame, indices, **kwargs):
        result = raw_run(frame, indices, **kwargs)
        # Camera predictions differ beyond a common scale: replacing these
        # with dataset cameras must change the cloud and fail this regression.
        result.pred_extrinsics[0, :3, :3] = torch.tensor([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
        result.pred_intrinsics = result.pred_intrinsics.clone()
        result.pred_intrinsics[:, 0, 0] *= 1.3
        result.metric_scale = 0.25
        results.append(result)
        return result

    backbone.run = run
    backbone.compute_metric_scale = lambda *a, **kw: 0.25

    runner = TrackingRunner()
    args = (source, 0, ClipSpec(length=2), ("left", "right"))
    options = dict(
        geometry_source=PREVIEW_ALIGNMENT,
        backbone="fake",
        backbone_runner=backbone,
        backbone_alignment=alignment,
    )
    clip = runner.prepare(*args, **options)
    for t, result in enumerate(results):
        depth, poses, intrinsics = aligned_depth_and_cameras(result, alignment)
        torch.testing.assert_close(clip.geometry.depth[0, t], depth)
        torch.testing.assert_close(clip.geometry.extrinsics[0, t], poses)
        torch.testing.assert_close(clip.geometry.intrinsics[0, t], intrinsics)
        expected = build_point_cloud(result, clip.images[0, t], align_mode=alignment)[0]
        actual = build_point_cloud(clip.display_result(t), clip.images[0, t])[0]
        np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert clip.geometry.provenance["backbone_alignment"] == alignment
    if alignment != "prescale_gt":
        assert not torch.allclose(clip.geometry.extrinsics[0, 0], clip.frames[0].extrinsics)
        assert not torch.allclose(clip.geometry.intrinsics[0, 0], clip.frames[0].intrinsics)
    assert runner.prepare(*args, **options) is clip
    options["backbone_alignment"] = "prescale_gt" if alignment == "sim3_points" else "sim3_points"
    assert runner.prepare(*args, **options) is not clip


def test_preview_alignment_rejects_unanchored_or_degenerate_geometry(source):
    options = dict(
        geometry_source=PREVIEW_ALIGNMENT,
        backbone="fake",
        backbone_runner=ScaledBackbone(),
    )
    with pytest.raises(ValueError, match="metric world frame"):
        prepare(source, backbone_alignment="none", **options)
    with pytest.raises(ValueError, match="two distinct camera"):
        TrackingRunner().prepare(source, 0, ClipSpec(length=2), ("left",), **options)


def test_backbone_input_size_is_independent_of_playback_and_part_of_cache_key(source):
    backbone, runner = ScaledBackbone(), TrackingRunner()
    native_run = backbone.run

    def run(frame, indices, **kwargs):
        assert frame.images.shape[-2:] == (source.height, source.width)
        return native_run(frame, indices, **kwargs)

    backbone.run = run
    args = (source, 0, ClipSpec(length=2, image_long_side=16), ("left", "right"))
    options = dict(geometry_source=SENSOR_SCALE, backbone="fake", backbone_runner=backbone)
    clip = runner.prepare(*args, **options)
    assert backbone.long_side is None  # use the model's own default, not playback's 16 px
    assert clip.images.shape[-2:] == (12, 16)
    assert runner.prepare(*args, **options) is clip
    assert backbone.calls == 2
    other = runner.prepare(*args, **options, backbone_long_side=966)
    assert other is not clip and backbone.calls == 4
    assert backbone.long_side == 966
    assert other.geometry.provenance["backbone_long_side"] == 966
    assert runner.prepare(*args, **options, backbone_long_side=966) is other


def test_sensor_depth_missing_and_degenerate_rig_are_rejected(source):
    original = source.get_frame
    source.get_frame = lambda *a, **kw: dataclasses.replace(original(*a, **kw), depth=None)
    with pytest.raises(ValueError, match="no sensor/GT depth"):
        prepare(source)
    source.get_frame = lambda *a, **kw: dataclasses.replace(
        original(*a, **kw), extrinsics=torch.eye(4).repeat(2, 1, 1)
    )
    with pytest.raises(ValueError, match="coincident"):
        prepare(
            source, geometry_source=RIG_SCALE, backbone="fake", backbone_runner=ScaledBackbone()
        )


def test_queries_and_ids_stay_on_reference_surfaces(source):
    clip = prepare(source)
    for sampling in ["Even coverage", "Farthest points"]:
        q = seed_queries(clip.geometry, 32, sampling=sampling, crop=((-2, -2, 1), (2, 2, 3)))
        validate_tracker_inputs(clip.images, q, clip.geometry)
        assert q.ids.unique().numel() == 32
        assert torch.all(q.xyz_world[..., 2] < 3)
        assert torch.all(q.time == 0)
    with pytest.raises(ValueError, match="excludes all"):
        seed_queries(clip.geometry, 32, crop=((-1, -1, -1), (0.1, 0.1, 0.1)))


@pytest.mark.parametrize(
    "name,view_count", [("mvtracker", 2), ("tapip3d", 1), ("trackcraft3r", 1), ("cotracker3", 2)]
)
def test_neural_adapter_receives_explicit_views_and_cpu_export(source, name, view_count):
    observed = []

    def build(settings, device):
        def model(images, queries, *, geometry):
            observed.append((images.shape, queries, geometry))
            xyz = queries.xyz_world[:, None].expand(-1, images.shape[1], -1, -1).clone()
            return make_tracker_output(
                queries, geometry, xyz, torch.ones(xyz.shape[:-1]), metadata={"tracker": name}
            )

        return model

    runner = TrackingRunner(builder=build)
    clip = prepare(source, runner)
    result = runner.run(clip, TrackerSettings(name=name), query_view="right", count=32)
    shape, queries, geometry = observed[0]
    assert shape[2] == view_count and geometry.depth.shape[2] == view_count
    if view_count == 1:
        assert (queries.source_view == 0).all()
        assert (geometry.extrinsics[..., 0, 3] == 0.3).all()
    archive = np.load(io.BytesIO(export_tracks(result)), allow_pickle=False)
    assert archive["tracks_world"].shape == (1, 12, 32, 3)
    np.testing.assert_array_equal(archive["ids"], queries.ids)
    assert archive["camera_to_world"].shape[2] == view_count
    metadata = json.loads(str(archive["metadata_json"]))
    assert metadata["tracking_cameras"] == (["right"] if view_count == 1 else ["left", "right"])
    assert metadata["units"] == ["meters"]
    assert metadata["dataset_frames"] == list(range(12))


def test_trails_never_bridge_invalid_or_hidden_samples(source):
    runner = TrackingRunner()
    run = runner.run(prepare(source), TrackerSettings(name="demo"), source=source, count=32)
    run.output.valid[:] = True
    run.output.visibility[:] = 1
    run.output.valid[0, 1, 0] = False
    run.output.visibility[0, 2, 1] = 0
    pts, colors, lines, _ = render_tracks(run, 3, trail_length=3, show_occluded=True)
    assert len(pts) == 32 and len(lines) == 3 * 32 - 2
    _, _, hidden_lines, _ = render_tracks(run, 3, trail_length=3, show_occluded=False)
    assert len(hidden_lines) == 3 * 32 - 4
    _, _, empty, _ = render_tracks(run, 3, trail_length=0)
    assert empty.shape == (0, 2, 3)
    np.testing.assert_array_equal(colors, run.colors)


def test_cancel_does_not_cache_partial_geometry_and_applies_to_cache(source):
    runner, cancel = TrackingRunner(), Event()

    def progress(message):
        cancel.set()

    with pytest.raises(TrackingCancelled):
        prepare(source, runner, progress=progress, cancel=cancel)
    assert runner.cached_clip is None
    cancel.clear()
    clip = prepare(source, runner)
    cancel.set()
    with pytest.raises(TrackingCancelled):
        prepare(source, runner, cancel=cancel)
    assert runner.cached_clip is clip


def test_sensor_validity_is_not_treated_as_confidence(source):
    from ontic_viz.backbone_viewer.render import build_point_cloud

    clip = prepare(source)
    # A binary validity mask with invalid pixels must not become percentile confidence.
    clip.geometry.depth[0, 0, 0, 0] = 0
    result = clip.display_result(0)
    assert result.conf is None
    points, _ = build_point_cloud(
        result, clip.images[0, 0], stride=1, conf_thresh=1.0, align_mode="none"
    )
    assert len(points) == int(clip.geometry.valid_depth[0, 0].sum())


class ConfidentBackbone(ScaledBackbone):
    def run(self, frame, indices, **kwargs):
        result = super().run(frame, indices, **kwargs)
        result.conf = torch.arange(result.depth.numel()).reshape(result.depth.shape).float()
        result.conf[0, 0, 0] = float("nan")
        return result


@pytest.mark.parametrize("sampling", ["Even coverage", "Farthest points"])
def test_queries_are_actual_displayed_samples_after_all_filters(source, sampling):
    clip = prepare(
        source, geometry_source=SENSOR_SCALE, backbone="fake", backbone_runner=ConfidentBackbone()
    )
    assert clip.confidence is not None
    torch.testing.assert_close(clip.display_result(0).conf, clip.confidence[0, 0], equal_nan=True)
    filters = PointFilters(
        stride=2,
        drop_conf_pct=30,
        crop=((-2, -2, 1), (2, 2, 3)),
        voxel_size=0.15,
        sfc_stride=2,
        max_points=35,
    )
    display = clip.display_result(0)
    points, _, indices = build_point_cloud(
        display,
        clip.images[0, 0],
        **filters.cloud_options(display.conf),
        return_source_indices=True,
    )
    run = TrackingRunner().run(
        clip,
        TrackerSettings(name="demo"),
        source=source,
        count=32,
        sampling=sampling,
        point_filters=filters,
    )
    assert 0 < run.queries.ids.numel() <= len(points) <= 35
    assert torch.isin(run.queries.ids, indices).all()
    # Voxel reduction retains real depth samples, not off-surface averages.
    nearest = torch.cdist(run.queries.xyz_world[0], torch.from_numpy(points)).min(-1).values
    assert (nearest < 1e-3).all()
    assert visible_query_mask(run, filters).all()
    assert torch.equal(run.clip.geometry.valid_depth, clip.geometry.valid_depth)


def test_confidence_filter_uses_all_displayed_views_before_mono_selection(source):
    clip = prepare(
        source, geometry_source=SENSOR_SCALE, backbone="fake", backbone_runner=ConfidentBackbone()
    )
    # The left view has only the bottom half of the global confidence range.
    with pytest.raises(ValueError, match="No visible depth"):
        TrackingRunner(builder=lambda *_: pytest.fail("No model should load")).run(
            clip,
            TrackerSettings(name="tapip3d"),
            query_view="left",
            point_filters=PointFilters(drop_conf_pct=60),
        )


def test_query_boxes_are_a_union_intersected_with_display_filters(source):
    clip = prepare(source)
    filters = PointFilters(stride=2)
    all_queries = seed_queries(
        clip.geometry, 64, source_indices=clip.visible_source_indices(filters)
    )
    centers = all_queries.xyz_world[0, [0, -1]]
    regions = tuple((tuple(p - 0.01), tuple(p + 0.01)) for p in centers)
    runner = TrackingRunner()
    run = runner.run(
        clip,
        TrackerSettings(name="demo"),
        source=source,
        count=64,
        point_filters=filters,
        regions=regions,
    )
    assert run.queries.ids.numel() >= 2
    assert visible_query_mask(run, filters, regions).all()
    assert not visible_query_mask(run, filters, (((100, 100, 100), (101, 101, 101)),)).any()
    visible = visible_query_mask(run, filters, regions[:1])
    assert visible.any() and not visible.all()
    points, _, lines, _ = render_tracks(run, 2, query_mask=visible)
    assert len(points) == int(visible.sum())
    assert len(lines) <= 2 * int(visible.sum())
    with pytest.raises(ValueError, match="excludes all"):
        runner.run(
            clip, TrackerSettings(name="demo"), source=source, point_filters=filters, regions=()
        )


def test_workspace_excludes_current_samples_and_breaks_trails_on_reentry(source):
    runner = TrackingRunner()
    run = runner.run(prepare(source), TrackerSettings(name="demo"), source=source, count=1)
    run.output.tracks_world[:] = torch.tensor([0.0, 0.0, 2.0])
    run.output.tracks_world[0, 1, 0, 0] = 3.0
    run.output.valid[:] = True
    original_valid = run.output.valid.clone()
    filters = PointFilters(crop=((-1, -1, 1), (1, 1, 3)))
    mask = visible_track_samples(run, filters)
    assert mask[:3, 0].tolist() == [True, False, True]
    points, _, trails, _ = render_tracks(run, 1, sample_mask=mask, show_occluded=True)
    assert len(points) == len(trails) == 0
    points, _, trails, _ = render_tracks(run, 2, sample_mask=mask, show_occluded=True)
    assert len(points) == 1 and len(trails) == 0
    assert len(render_tracks(run, 3, sample_mask=mask)[2]) == 1
    assert visible_track_samples(run, PointFilters()).all()
    torch.testing.assert_close(run.output.valid, original_valid)


def test_current_confidence_filters_follow_positions_and_camera_scope(source):
    runner = TrackingRunner()
    clip = prepare(source)
    run = runner.run(clip, TrackerSettings(name="demo"), source=source, count=1)
    clip.geometry.extrinsics[:] = torch.eye(4)
    clip.geometry.intrinsics[:] = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    clip.geometry.depth[:] = 2.0
    clip.confidence = torch.ones_like(clip.geometry.depth)
    run.output.tracks_world[:] = torch.tensor([0.125, 0.125, 2.0])
    run.output.valid[:] = True
    run.queries.source_view[:] = 0
    # Projection is (0.5625, 0.5625): pixel x=18, y=13 on this 32x24 grid.
    clip.confidence[0, 1, :, 13, 18] = 0
    clip.confidence[0, 2, 0, 13, 18] = 0  # right camera still supports this sample
    run.output.tracks_world[0, 3, 0, 0] = 100.0
    run.output.tracks_world[0, 4, 0, 2] = -2.0
    filters = PointFilters(drop_conf_pct=50)
    mask = visible_track_samples(run, filters)
    assert mask[:5, 0].tolist() == [True, False, True, False, False]
    assert len(render_tracks(run, 2, sample_mask=mask)[2]) == 0
    run.output.visibility_scope = "query_view"
    assert not visible_track_samples(run, filters)[2, 0]
    assert len(render_tracks(run, 1, sample_mask=mask, show_occluded=True)[0]) == 0
    assert visible_track_samples(run, PointFilters(drop_conf_pct=0))[1, 0]
