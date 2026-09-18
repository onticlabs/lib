"""Viewer integration: cached scrubbing, camera selection, and failure recovery."""

from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
import torch
import viser

from ontic_viz.backbone_viewer.app import BackboneViewer


@pytest.fixture
def app(tmp_path):
    server = viser.ViserServer(port=0, verbose=False)
    viewer = BackboneViewer(
        server,
        device="cpu",
        roots={},
        presets_path=tmp_path / "presets.json",
        datasets=["demo"],
        backbones=["da3"],
        metric_models=["da3"],
    )
    try:
        viewer.load_demo()
        yield viewer
    finally:
        viewer.close()
        server.stop()


def test_demo_scrub_preserves_ids_and_camera_selection_without_loading(app):
    panel = app.tracking
    result = panel.result
    assert result is not None and not result.output.metadata["neural_inference"]
    app._toggle_cam(1)
    assert app._selected_ix() == [0]

    def unexpected(*args, **kwargs):
        raise AssertionError("Cached scrubbing must not load data or infer geometry")

    app._source.get_frame = app.runner.compute_metric_scale = unexpected
    app.dd_align.value = "metric_mono"
    app.sl_time.value = 8
    app._load_frame()
    assert app._selected_ix() == [0]
    assert app.tracking.result is result
    assert app._tracking_frame
    torch.testing.assert_close(app._result.depth, result.clip.geometry.depth[0, 8])
    assert "9 / 48" in panel.timeline.content
    assert not panel.play.disabled
    assert not app._operation_lock.locked()


def test_rerun_failure_keeps_previous_result_and_restores_controls(app):
    panel = app.tracking
    previous = panel.result
    panel.dd_tracker.value = "MVTracker"

    def unavailable(*a, **kw):
        raise ImportError("test upstream dependency unavailable")

    panel.runner.builder = unavailable
    panel.start_job(background=False)
    assert "upstream dependency unavailable" in panel.message.content
    assert panel.result is previous
    assert not panel.run_button.disabled and not app.btn_loadds.disabled
    assert panel.cancel_button.disabled and not panel.play.disabled
    assert not app._busy and not app._operation_lock.locked()


def test_automatic_frame_updates_keep_playing_but_user_browsing_pauses(app):
    panel = app.tracking
    panel._playing.set()
    # Viser can dispatch this callback after the play loop resets _suspend.
    app._on_traj_or_time(SimpleNamespace(client=None))
    assert panel._playing.is_set()
    app._on_traj_or_time(SimpleNamespace(client=object()))
    assert not panel._playing.is_set()
    assert panel.play.label == "Play clip"


def test_invalid_clip_is_reported_before_work_and_context_reset_clears(app):
    panel = app.tracking
    previous = panel.result
    panel.start.value = 70
    panel.start_job(background=False)
    assert "Clip ends" in panel.message.content
    assert panel.result is previous and not app._busy
    panel.set_context(app._source, 1)
    assert panel.result is None and panel.runner.cached_clip is None
    assert panel.play.disabled and panel.download_button.disabled


def test_outside_clip_hides_tracks_and_camera_selection_survives(app):
    app._toggle_cam(1)
    app.sl_time.value = 60
    app._load_frame()
    assert not app._tracking_frame and not app.tracking._handles
    assert "outside" in app.tracking.timeline.content
    assert app._selected_ix() == [0]
    app.sl_time.value = 0
    app._load_frame()
    assert app._tracking_frame and app.tracking._handles
    assert app._selected_ix() == [0]


def test_cached_depth_visibility_follows_cameras_through_playback(app):
    previous = app.tracking.result
    assert len(app._cloud_pts) > 0
    app._set_all(False)
    assert len(app._cloud_pts) == 0

    def unexpected(*args, **kwargs):
        raise AssertionError("Changing cached view visibility must not reload or infer")

    app._source.get_frame = app.runner.run = unexpected
    app.sl_time.value = 8
    app._load_frame()
    assert len(app._cloud_pts) == 0
    app._toggle_cam(1)
    assert len(app._cloud_pts) > 0
    assert app.tracking.result is previous


def test_pending_sampling_does_not_affect_playback_or_track_visibility(app):
    panel = app.tracking
    panel.seek_clip(8)
    original = app._cloud_pts.copy()
    visible = panel._visible_queries().clone()
    samples = panel._visible_samples().clone()
    app.sl_stride.value = 8
    app.sl_conf.value = 90
    app.sl_sfc.value = 4
    panel.seek_clip(9)
    panel.seek_clip(8)
    torch.testing.assert_close(torch.from_numpy(app._cloud_pts), torch.from_numpy(original))
    torch.testing.assert_close(panel._visible_queries(), visible)
    torch.testing.assert_close(panel._visible_samples(), samples)
    app._apply_point_sampling()
    assert len(app._cloud_pts) < len(original)
    assert int(panel._visible_queries().sum()) < int(visible.sum())


def test_3d_camera_click_survives_playback_and_filters_cached_depth(app, monkeypatch):
    panel = app.tracking
    cameras = list(app._cam_scene)
    checks = list(app._cam_checks)
    previous = panel.result

    def forbidden(*args, **kwargs):
        raise AssertionError("Camera selection must reuse the cached depth")

    app._source.get_frame = app.runner.run = forbidden
    panel.seek_clip(8)
    assert app._cam_scene == cameras and app._cam_checks == checks
    full_cloud = app._cloud_pts.copy()
    paused = Event()
    original_pause = panel.pause

    def pause():
        original_pause()
        paused.set()

    monkeypatch.setattr(panel, "pause", pause)
    callback = cameras[1]._impl.click_cb[0].callback
    # A click arriving during an in-flight playback frame must not be lost.
    with ThreadPoolExecutor(max_workers=1) as executor:
        with app._operation_lock:
            app._busy = True
            panel._playing.set()
            pending = executor.submit(callback, SimpleNamespace(target=cameras[1]))
            try:
                assert paused.wait(timeout=5)
            finally:
                app._busy = False
        pending.result(timeout=5)
    assert not panel._playing.is_set()
    assert app._selected_ix() == [0]
    assert len(app._cloud_pts) < len(full_cloud)
    callback(SimpleNamespace(target=cameras[1]))
    assert app._selected_ix() == [0, 1]
    torch.testing.assert_close(torch.from_numpy(app._cloud_pts), torch.from_numpy(full_cloud))
    assert panel.result is previous


def test_depth_preview_needs_no_tracker_and_history_restores_tracks(app):
    panel = app.tracking
    previous = panel.result
    previous_label = panel.history_picker.value
    panel.dd_tracker.value = "MVTracker"
    panel.length.value = 2  # too short for MVTracker, valid for a depth-only preview

    def forbidden(*args, **kwargs):
        raise AssertionError("Depth preview must not build a tracker")

    panel.runner.builder = forbidden
    panel.start_job(background=False, track=False)
    assert panel.result is None and panel.preview is not None
    assert len(panel.preview.indices) == 2
    assert not panel.play.disabled and panel.download_button.disabled
    assert "video depth" in panel.timeline.content
    panel.history_picker.value = previous_label
    panel._select_history()
    assert panel.result is previous and panel.preview is None
    assert not panel.download_button.disabled


def test_backbone_preview_is_reused_with_its_alignment_for_tracking(app, monkeypatch):
    from ontic_nn.trackers.common import make_tracker_output
    from ontic_viz.backbone_viewer.runner import BackboneResult, aligned_depth_and_cameras
    from ontic_viz.backbone_viewer.tracking import PREVIEW_ALIGNMENT

    panel = app.tracking
    calls, tracked = [], []
    app.cb_crop.value = False
    app.sl_conf.value = 0
    app.dd_align.value = "sim3_points"
    panel.length.value = 7
    panel.resolution.value = 128
    panel.dd_tracker.value = "MVTracker"

    def predict(frame, indices, **kwargs):
        assert frame.images.shape[-2:] == (app._source.height, app._source.width)
        poses = frame.extrinsics[indices].clone()
        poses[:, :3, 3] *= 2
        poses[0, :3, :3] = torch.tensor([[0.8, 0, 0.6], [0, 1, 0], [-0.6, 0, 0.8]])
        intrinsics = frame.intrinsics[indices].clone()
        intrinsics[:, 0, 0] *= 1.3
        result = BackboneResult(
            frame.depth[indices, 0] * (2 + len(calls)),
            None,
            poses,
            intrinsics,
            frame.extrinsics[indices],
            frame.intrinsics[indices],
        )
        calls.append(result)
        return result

    def tracker(images, queries, *, geometry):
        tracked.append(geometry)
        xyz = queries.xyz_world[:, None].expand(-1, images.shape[1], -1, -1).clone()
        return make_tracker_output(queries, geometry, xyz, torch.ones(xyz.shape[:-1]))

    monkeypatch.setattr(app.runner, "load", lambda *a: None)
    monkeypatch.setattr(app.runner, "release", lambda: None)
    monkeypatch.setattr(app.runner, "run", predict)
    panel.runner.builder = lambda *a: tracker
    app._run()
    assert panel.geometry.value == PREVIEW_ALIGNMENT
    expected = aligned_depth_and_cameras(calls[0], "sim3_points")
    panel.start_job(background=False)
    assert len(calls) == 7  # first frame reused; only six further predictions
    assert len(tracked) == 1, panel.message.content
    for actual, reference in zip(
        (tracked[0].depth[0, 0], tracked[0].extrinsics[0, 0], tracked[0].intrinsics[0, 0]),
        expected,
    ):
        torch.testing.assert_close(actual, reference)
    assert panel.result.clip.geometry.provenance["backbone_alignment"] == "sim3_points"
    assert not app.dd_align.disabled
    assert not app.dd_metric.disabled

    # New alignment invalidates the clip cache, but still uses the same raw preview.
    app.dd_align.value = "prescale_gt"
    panel.start_job(background=False, track=False)
    assert len(calls) == 13
    torch.testing.assert_close(panel.active_clip.geometry.extrinsics[0, 0], calls[0].gt_extrinsics)
    app._run()
    latest = calls[-1]
    panel.start_job(background=False, track=False)
    assert len(calls) == 20  # explicit rerun must supersede the older cached clip
    expected_depth, _, _ = aligned_depth_and_cameras(latest, "prescale_gt")
    torch.testing.assert_close(panel.active_clip.geometry.depth[0, 0], expected_depth)


def test_clip_scrubber_maps_sampled_frames_and_pauses_without_inference(app):
    panel = app.tracking
    panel.start.value = 4
    panel.length.value = 3
    panel.stride.value = 5
    panel.start_job(background=False, track=False)
    assert panel.active_clip.indices == (4, 9, 14)
    assert panel.clip_frame.max == 3
    assert "dataset 4–14 (step 5)" in panel.playback_hint.content
    app._toggle_cam(1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Clip scrubbing must use cached frames, without inference")

    app._source.get_frame = app.runner.run = forbidden
    panel._playing.set()
    panel.clip_frame.value = 2  # server updates must not pause or seek again
    assert panel._playing.is_set() and app.sl_time.value == 4
    panel._scrub_clip(SimpleNamespace(client=object(), target=panel.clip_frame))
    assert not panel._playing.is_set()
    assert app.sl_time.value == 9 and app._selected_ix() == [0]
    assert "2 / 3" in panel.timeline.content
    panel.seek_clip(0)
    assert app.sl_time.value == 4 and panel.clip_frame.value == 1
    assert app._selected_ix() == [0]


def test_play_at_clip_end_restarts_and_frame_updates_keep_playing(app, monkeypatch):
    panel = app.tracking
    monkeypatch.setattr(panel, "_play_loop", lambda: None)
    panel.loop.value = False
    panel.seek_clip(len(panel.active_clip.indices) - 1)
    panel.toggle_playback()
    assert app.sl_time.value == panel.active_clip.indices[0]
    assert panel._playing.is_set() and panel.play.label == "Pause clip"
    app._suspend = True
    app.sl_time.value = panel.active_clip.indices[1]
    app._suspend = False
    panel.show_frame(int(app.sl_time.value))
    assert panel.clip_frame.value == 2 and panel._playing.is_set()
    # Browsing the full dataset takes control away from automatic playback.
    app.sl_time.value = 60
    app._on_traj_or_time(SimpleNamespace(client=object()))
    assert not panel._playing.is_set()
    assert "outside" in panel.timeline.content
    panel.seek_clip(0)
    assert app._tracking_frame and panel.clip_frame.value == 1


def test_playback_empty_state_and_history_switch_update_clip_range(app):
    panel = app.tracking
    original_label = panel.history_picker.value
    assert not panel.history_picker.visible
    panel.length.value = 2
    panel.start_job(background=False, track=False)
    assert panel.history_picker.visible and panel.clip_frame.max == 2
    assert "Video depth" in panel.playback_hint.content
    panel.history_picker.value = original_label
    assert panel.clip_frame.max == 48
    assert "Tracks + depth" in panel.playback_hint.content
    panel.clear(clear_cache=True)
    assert panel.play.disabled and panel.restart.disabled and panel.clip_frame.disabled
    assert not panel.history_picker.visible
    assert "Run video depth" in panel.playback_hint.content
    assert "Run tracking" in panel.playback_hint.content


def test_recording_reopens_matching_dataset_context(app, tmp_path):
    from ontic_viz.backbone_viewer.recording import save_recording

    path = save_recording(app.tracking.result, tmp_path / "demo.viewer.npz")
    app.tracking.clear(clear_cache=True)
    app._open_recording(path)
    assert app.tracking.result is not None
    assert app.tracking.active_clip.camera_names == ("left", "right")
    assert "Sensor depth" in app.tracking.history_picker.value
    assert not app.tracking.play.disabled


def test_single_frame_preview_keeps_the_only_prepared_clip_accessible(app):
    panel = app.tracking
    previous, label = panel.result, panel.history_picker.value
    panel.clear()  # A single-frame depth preview clears the active clip, retaining history.
    assert panel.active_clip is None and panel.play.disabled
    assert panel.history_picker.visible and not panel.history_picker.disabled
    assert panel.history_picker.value not in panel.history
    panel.history_picker.value = label
    panel._select_history()
    assert panel.result is previous and not panel.play.disabled
    assert list(panel.history_picker.options) == [label]


def test_recording_load_disables_playback_and_recovers_on_failure(app, monkeypatch):
    from ontic_viz.backbone_viewer import recording

    previous = app.tracking.result

    def fail(path):
        assert app.tracking.play.disabled
        raise FileNotFoundError(path)

    monkeypatch.setattr(recording, "load_recording", fail)
    with pytest.raises(FileNotFoundError):
        app._open_recording("missing.viewer.npz")
    assert app.tracking.result is previous
    assert not app.tracking.play.disabled


def test_display_crop_hides_cached_seeds_and_blocks_new_hidden_seeds(app):
    panel = app.tracking
    previous = panel.result
    app._suspend = True
    app.cb_crop.value = True
    app.vec_wmin.value = (-1, -1, -1)
    app.vec_wmax.value = (0, 0, 0)
    app._suspend = False
    app._cheap_update()
    assert len(app._cloud_pts) == 0
    assert not panel._visible_queries().any()
    assert "0 /" in panel.timeline.content
    panel.start_job(background=False)
    assert "No visible depth" in panel.message.content
    assert panel.result is previous


@pytest.mark.parametrize("has_cloud", [True, False])
def test_query_boxes_can_be_added_resized_dragged_and_removed(app, has_cloud):
    if not has_cloud:
        app._cloud_pts = None
    regions = app.tracking.regions
    regions.add_box()
    assert regions.enabled.value and len(regions.bounds()) == 1
    first = regions.selected.value
    regions.add_box()
    assert len(regions.bounds()) == 2
    regions.size.value = (0.2, 0.3, 0.4)
    regions._edit()
    regions._gizmo.position = (0.1, 0.2, 2.5)
    regions._drag(None)
    assert regions.center.value == (0.1, 0.2, 2.5)
    lo, hi = regions.bounds()[-1]
    torch.testing.assert_close(
        torch.tensor(hi) - torch.tensor(lo), torch.tensor((0.2, 0.3, 0.4), dtype=torch.float64)
    )
    regions.include.value = False
    regions._edit()
    assert len(regions.bounds()) == 1
    regions.remove_box()
    assert regions.selected.value == first
    regions.remove_box()
    assert regions.bounds() == ()
    app.tracking.start_job(background=False)
    assert "Add or include a query box" in app.tracking.message.content
    regions.enabled.value = False
    assert regions.bounds() is None


def test_add_box_uses_selected_depth_and_first_clip_frame(app):
    from ontic_viz.backbone_viewer.tracking import region_mask
    from ontic_viz.backbone_viewer.video_depth import VIDEO_METRIC

    panel = app.tracking
    app.cb_crop.value = False
    previous = panel.result
    panel.seek_clip(8)
    assert app.sl_time.value == 8
    panel.length.value = 3
    panel.geometry.value = VIDEO_METRIC
    calls = []

    def builder(settings, device):
        calls.append(settings)
        return lambda images, **kwargs: torch.full((*images.shape[:3], *images.shape[-2:]), 6.0)

    panel.runner.video_builder = builder
    panel.show_query_frame(on_ready=panel.regions.add_box, background=False)
    assert panel.result is None and panel.preview is not None, panel.message.content
    assert app.sl_time.value == 0 and panel.clip_frame.value == 1
    assert (app._result.depth == 6).all()
    assert any(item is previous for item in panel.history.values())
    points = torch.from_numpy(app._cloud_pts)
    assert region_mask(points, panel.regions.bounds()).any()
    assert "Query reference: dataset frame 0" in panel.message.content
    panel.start_job(background=False)
    assert panel.result is not None, panel.message.content
    assert len(calls) == 1  # Selecting and tracking use exactly the same cached geometry.
    assert region_mask(panel.result.queries.xyz_world[0], panel.regions.bounds()).all()
    distances = torch.cdist(panel.result.queries.xyz_world[0].double(), points.double())
    assert distances.min(-1).values.max() < 1e-5


def test_empty_box_shows_the_new_reference_depth_instead_of_old_result(app):
    from ontic_viz.backbone_viewer.video_depth import VIDEO_METRIC

    panel = app.tracking
    app.cb_crop.value = False
    previous = panel.result
    previous_points = torch.from_numpy(app._cloud_pts).clone()
    panel.seek_clip(8)
    panel.regions.add_box()
    # This box covers the old displayed geometry, but the new depth is far away.
    lo, hi = previous_points.amin(0) - 0.1, previous_points.amax(0) + 0.1
    panel.regions.center.value = tuple(((lo + hi) / 2).tolist())
    panel.regions.size.value = tuple((hi - lo).tolist())
    panel.regions._edit()
    panel.length.value = 3
    panel.geometry.value = VIDEO_METRIC
    panel.runner.video_builder = lambda *a: (
        lambda images, **kwargs: torch.full((*images.shape[:3], *images.shape[-2:]), 20.0)
    )
    panel.start_job(background=False)
    assert "No query points in the boxes on dataset frame 0" in panel.message.content
    assert "vda_small" in panel.message.content
    assert panel.result is None and panel.preview is panel.runner.cached_clip
    assert app.sl_time.value == 0 and panel.clip_frame.value == 1
    assert (app._result.depth == 20).all()
    assert any(item is previous for item in panel.history.values())
    assert not panel.regions.reference.disabled and not panel.run_button.disabled
    assert not app._busy and not app._operation_lock.locked()


def test_new_query_box_starts_on_a_visible_point_despite_depth_outliers(app):
    import numpy as np
    from ontic_viz.backbone_viewer.tracking import region_mask

    points = app._cloud_pts.copy()
    app._cloud_pts = np.concatenate([points, np.full((1, 3), 1e6, dtype=np.float32)])
    app._cloud_bounds = (app._cloud_pts.min(0), app._cloud_pts.max(0))
    app.tracking.regions.add_box()
    assert region_mask(torch.from_numpy(points), app.tracking.regions.bounds()).any()
    assert max(app.tracking.regions.size.value) < 10


def test_playback_excludes_tracks_after_they_leave_the_display_workspace(app):
    panel = app.tracking
    run = panel.result
    selected = int(torch.where(panel._visible_queries())[0][0])
    run.output.tracks_world[0, 1, selected] = torch.tensor([10.0, 10.0, 10.0])
    panel._sample_mask_cache = None
    app._suspend = True
    app.sl_time.value = run.clip.indices[1]
    app._suspend = False
    panel.show_frame(run.clip.indices[1])
    assert panel._visible_queries()[selected]  # the reference seed remains eligible
    assert not panel._visible_samples()[1, selected]
    points = torch.from_numpy(panel._handles[0].points)
    lo, hi = torch.tensor(app.vec_wmin.value), torch.tensor(app.vec_wmax.value)
    assert ((points >= lo) & (points <= hi)).all()


def test_tracker_switch_preserves_separate_model_files(app):
    panel = app.tracking

    def select(label):
        panel.dd_tracker.value = label
        panel._on_tracker_change()

    select("MVTracker")
    panel.repo.value = "/research/mvtracker"
    panel.checkpoint.value = "/models/mvtracker.pth"
    select("TAPIP3D")
    assert panel.repo.value == panel.checkpoint.value == ""
    panel.repo.value = "/research/tapip3d"
    panel.checkpoint.value = "/models/tapip3d.pth"
    select("CoTracker3")
    assert panel.repo.value == panel.checkpoint.value == ""
    assert not panel.view.visible
    assert "independently" in panel.model_hint.content
    panel.repo.value = "/research/co-tracker"
    panel.checkpoint.value = "/models/scaled_online.pth"
    select("TrackCraft3R")
    assert panel.repo.value == panel.checkpoint.value == ""
    panel.wan_cache.value = "/models/wan"
    panel.download.value = True
    select("MVTracker")
    assert panel.repo.value == "/research/mvtracker"
    assert panel.checkpoint.value == "/models/mvtracker.pth"
    assert panel.wan_cache.value == "" and not panel.download.value
    select("TAPIP3D")
    assert panel.repo.value == "/research/tapip3d"
    assert panel.checkpoint.value == "/models/tapip3d.pth"
    select("CoTracker3")
    assert panel.repo.value == "/research/co-tracker"
    assert panel.checkpoint.value == "/models/scaled_online.pth"
    select("TrackCraft3R")
    assert panel.wan_cache.value == "/models/wan" and panel.download.value


@pytest.mark.parametrize("legacy", [False, True])
def test_model_config_assigns_paths_to_named_trackers(app, legacy):
    from ontic_viz.backbone_viewer.cli import apply_model_config

    mv = {"repo_path": "/research/mvtracker", "checkpoint_path": "/models/mv.pth"}
    tap = {"repo_path": "/research/tapip3d", "checkpoint_path": "/models/tap.pth"}
    config = {"tracker": mv} if legacy else {"trackers": {"mvtracker": mv, "tapip3d": tap}}
    panel = app.tracking
    panel.dd_tracker.value = "TAPIP3D"
    panel._on_tracker_change()
    apply_model_config(app, config)
    assert panel.repo.value == ("" if legacy else tap["repo_path"])
    assert panel.checkpoint.value == ("" if legacy else tap["checkpoint_path"])
    panel.dd_tracker.value = "MVTracker"
    panel._on_tracker_change()
    assert panel.repo.value == mv["repo_path"]
    assert panel.checkpoint.value == mv["checkpoint_path"]


def test_model_config_video_paths_survive_model_switches(app, tmp_path, monkeypatch):
    import torch
    from ontic_viz.backbone_viewer.cli import apply_model_config
    from ontic_viz.backbone_viewer.tracking_panel import VIDEO_MODEL_LABELS

    hub_dirs = []
    monkeypatch.setattr(torch.hub, "set_dir", hub_dirs.append)
    checkpoints = {name: f"/models/{name}" for name in VIDEO_MODEL_LABELS.values()}
    panel = app.tracking
    panel.video_input_size.value = 756
    apply_model_config(app, {"video_checkpoints": checkpoints, "torch_hub_dir": str(tmp_path)})
    assert hub_dirs == [str(tmp_path)]
    assert panel.video_input_size.value == 756
    for label, name in VIDEO_MODEL_LABELS.items():
        panel.video_model.value = label
        panel._on_video_model_change()
        assert panel.video_checkpoint.value == checkpoints[name]
        assert panel.video_download.value is False
    with pytest.raises(ValueError, match="Unknown video models"):
        panel.configure_video_checkpoints({"typo": "/model"})
