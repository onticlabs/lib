"""Viewer integration: cached scrubbing, camera selection, and failure recovery."""

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
    assert "depth preview" in panel.timeline.content
    panel.history_picker.value = previous_label
    panel._select_history()
    assert panel.result is previous and panel.preview is None
    assert not panel.download_button.disabled


def test_recording_reopens_matching_dataset_context(app, tmp_path):
    from ontic_viz.backbone_viewer.recording import save_recording

    path = save_recording(app.tracking.result, tmp_path / "demo.viewer.npz")
    app.tracking.clear(clear_cache=True)
    app._open_recording(path)
    assert app.tracking.result is not None
    assert app.tracking.active_clip.camera_names == ("left", "right")
    assert "Sensor depth" in app.tracking.history_picker.value
    assert not app.tracking.play.disabled


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


def test_query_boxes_can_be_added_resized_dragged_and_removed(app):
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
