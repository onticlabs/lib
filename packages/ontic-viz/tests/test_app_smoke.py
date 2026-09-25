"""Smoke tests for the viser app.

The headless part builds a real viser server on an ephemeral port and drives the GUI
with fakes (no models, no data). The end-to-end part additionally needs CUDA, the
``ontic_nn`` / ``ontic_data`` registries, cached DINOv2 weights and a readable dataset
root with GT depth; it runs the ``gtdepth`` backbone on one frame.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import viser

from ontic_viz.backbone_viewer.app import _INPUT_COLOR, _OFF_COLOR, _PRED_COLOR, BackboneViewer
from ontic_viz.backbone_viewer.data_source import DEFAULT_ROOTS, Frame, dataset_has_gt_depth
from ontic_viz.backbone_viewer.runner import BackboneResult

BACKBONES = ["da3", "ma", "vggt", "pi3x", "dvlt", "moge3", "gtdepth"]
DATASETS = ["genesis", "hocap", "taco", "dextris", "physinone", "synthrobot"]


@pytest.fixture
def viewer(tmp_path):
    server = viser.ViserServer(port=0, verbose=False)
    try:
        yield BackboneViewer(
            server,
            device="cpu",
            roots={},
            presets_path=tmp_path / "presets.json",
            backbones=BACKBONES,
            datasets=DATASETS,
            metric_models=["da3"],
        )
    finally:
        server.stop()


def _intr(v):
    k = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    return k.unsqueeze(0).repeat(v, 1, 1)


def _frame(v=3, hw=28, hands=None):
    return Frame(
        images=torch.rand(v, 3, hw, hw),
        intrinsics=_intr(v),
        extrinsics=torch.eye(4).expand(v, 4, 4).clone(),
        cam_names=[f"cam/{i}" for i in range(v)],
        hands=hands,
    )


def _result(frame):
    v = frame.images.shape[0]
    return BackboneResult(
        depth=torch.full((v, 14, 14), 2.0),
        conf=torch.full((v, 14, 14), 3.0),
        pred_extrinsics=frame.extrinsics,
        pred_intrinsics=frame.intrinsics,
        gt_extrinsics=frame.extrinsics,
        gt_intrinsics=frame.intrinsics,
    )


def _wire(viewer, frame):
    viewer._frame = frame
    viewer._rebuild_cameras(frame)
    viewer._run_images = frame.images
    viewer._run_ix = list(range(frame.images.shape[0]))
    viewer._result = _result(frame)


def _click_camera(handle):
    callbacks = handle._impl.click_cb
    assert len(callbacks) == 1
    callbacks[0].callback(SimpleNamespace(target=handle))


def test_backbone_size_defaults_and_overrides_follow_the_selected_model(viewer):
    from ontic_nn.wrappers import BACKBONES

    assert viewer.sl_lside.value == BACKBONES[viewer.dd_backbone.value]().long_side
    viewer.sl_lside.value = 280
    for name in ("moge3", "vggt", "ma", "dvlt", "pi3x"):
        viewer.dd_backbone.value = name
        viewer._on_backbone_change()
        assert viewer.sl_lside.value == BACKBONES[name]().long_side
    viewer.dd_backbone.value = "da3"
    viewer._on_backbone_change()
    assert viewer.sl_lside.value == 280


def test_cameras_checkboxes_frustums_and_toggle(viewer):
    frame = _frame(v=3)
    viewer._frame = frame
    viewer._rebuild_cameras(frame)
    assert len(viewer._cam_checks) == 3 and len(viewer._cam_scene) == 3
    assert viewer._selected_ix() == [0, 1, 2]
    _click_camera(viewer._cam_scene[2])
    assert viewer._selected_ix() == [0, 1]
    assert tuple(viewer._cam_scene[2].color) == _OFF_COLOR
    assert tuple(viewer._cam_scene[0].color) == _INPUT_COLOR
    _click_camera(viewer._cam_scene[2])
    assert viewer._selected_ix() == [0, 1, 2]
    viewer._set_all(False)
    assert viewer._selected_ix() == []
    old_camera = viewer._cam_scene[0]
    viewer._rebuild_cameras(_frame(v=2))
    assert len(viewer._cam_checks) == 2
    _click_camera(old_camera)
    assert viewer._selected_ix() == []  # ignore delayed events from removed cameras


def test_depth_inference_only_receives_cameras_selected_in_3d(viewer, monkeypatch):
    frame = _frame(v=3)
    viewer._frame = frame
    viewer._rebuild_cameras(frame)
    _click_camera(viewer._cam_scene[1])
    selected = []

    def run(frame, indices, **kwargs):
        selected.append(list(indices))
        return BackboneResult(
            depth=torch.full((len(indices), 14, 14), 2.0),
            conf=None,
            pred_extrinsics=None,
            pred_intrinsics=None,
            gt_extrinsics=frame.extrinsics[indices],
            gt_intrinsics=frame.intrinsics[indices],
        )

    monkeypatch.setattr(viewer, "_configure_backbone", lambda: None)
    monkeypatch.setattr(viewer.runner, "load", lambda name: None)
    monkeypatch.setattr(viewer.runner, "run", run)
    viewer._run()
    assert selected == [[0, 2]]
    assert viewer._run_ix == [0, 2]
    _click_camera(viewer._cam_scene[1])
    assert selected == [[0, 2]]  # selection changes never trigger inference
    assert "Run depth again" in viewer.status.content


def test_render_cloud_all_color_modes_and_fps_on_demand(viewer):
    _wire(viewer, _frame(v=2))
    for mode in ("RGB", "confidence", "per-view"):
        viewer.dd_color.value = mode
        viewer._render_cloud()
        assert "points" in viewer.status.content
    assert "fps" not in viewer.status.content
    viewer.sl_fps.value = 5
    viewer._apply_point_sampling()
    assert "fps 5" in viewer.status.content


def test_point_cloud_edits_wait_for_apply_even_across_other_redraws(viewer, monkeypatch):
    _wire(viewer, _frame(v=2))
    viewer._render_cloud()
    original = viewer._cloud_pts.copy()
    original_filters = viewer._point_filters()
    redraws = []
    render_cloud = viewer._render_cloud

    def render():
        redraws.append(True)
        render_cloud()

    monkeypatch.setattr(viewer, "_render_cloud", render)
    for control, value in (
        (viewer.sl_stride, 4),
        (viewer.sl_conf, 50),
        (viewer.sl_voxel, 0.01),
        (viewer.sl_sfc, 2),
        (viewer.sl_fps, 5),
    ):
        control.value = value
        # Dispatch the same callbacks as a client GUI edit, synchronously.
        for callback in control._impl.update_cb:
            callback(SimpleNamespace(client=object(), target=control))
    assert not redraws
    assert viewer._point_filters() == original_filters

    # Appearance and camera updates must not commit the pending sampling values.
    viewer.dd_color.value = "per-view"
    viewer._cheap_update()
    _click_camera(viewer._cam_scene[0])
    _click_camera(viewer._cam_scene[0])
    np.testing.assert_allclose(viewer._cloud_pts, original)
    assert "stride 2, drop 0%" in viewer.status.content

    viewer._apply_point_sampling()
    assert len(viewer._cloud_pts) == 5
    assert "stride 4, drop 50%" in viewer.status.content
    assert "voxel 0.010m, sfc/2, fps 5" in viewer.status.content


def test_sampling_presets_and_edits_before_depth_stay_pending(viewer):
    from ontic_viz.backbone_viewer.config import ViewConfig

    viewer._apply_config(ViewConfig(stride=4, fps_max_points=5))
    assert viewer._point_filters().stride == 2
    # Apply must work even before the first depth result exists.
    viewer._apply_point_sampling()
    _wire(viewer, _frame(v=2))
    viewer._render_cloud()
    assert len(viewer._cloud_pts) == 5
    viewer._apply_config(ViewConfig(stride=1, fps_max_points=0))
    assert len(viewer._cloud_pts) == 5
    viewer._apply_point_sampling()
    assert len(viewer._cloud_pts) == 2 * 14 * 14


def test_camera_selection_filters_cached_moge_depth_without_inference(viewer):
    frame = _frame(v=4)
    frame.extrinsics[:, 0, 3] = torch.arange(4) * 10
    _wire(viewer, frame)
    # The run contains a noncontiguous subset of dataset cameras.
    subset = [1, 3]
    result = _result(frame)
    result.pred_extrinsics = result.pred_intrinsics = None
    result.depth = result.depth[subset]
    result.conf = result.conf[subset]
    result.gt_extrinsics = frame.extrinsics[subset]
    result.gt_intrinsics = frame.intrinsics[subset]
    viewer._result = result
    viewer._run_ix = subset
    viewer._run_images = frame.images[subset]
    viewer._suspend = True
    viewer.dd_backbone.value = "moge3"
    viewer.cb_crop.value = False
    viewer.sl_voxel.value = 0
    viewer.sl_sfc.value = 1
    viewer._suspend = False
    viewer._render_cloud()
    original = viewer._cloud_pts.copy()
    assert len(original) > 0

    def forbidden(*args, **kwargs):
        raise AssertionError("Camera visibility must not rerun inference")

    viewer.runner.run = viewer.runner.compute_metric_scale = forbidden
    _click_camera(viewer._cam_scene[1])
    assert len(viewer._cloud_pts) == len(original) // 2
    assert (viewer._cloud_pts[:, 0] > 20).all()  # dataset camera 3
    assert viewer._result is result
    assert (result.depth > 0).all()  # cached depth remains intact
    viewer._set_all(False)
    assert len(viewer._cloud_pts) == 0
    viewer._toggle_occupancy()
    assert viewer._occupancy_on
    viewer._toggle_occupancy()
    viewer._toggle_cam(0)  # no prediction for this camera in the cached run
    assert len(viewer._cloud_pts) == 0
    viewer._set_all(True)
    np.testing.assert_allclose(viewer._cloud_pts, original)


def test_metric_mono_scale_fit_cached_and_invalidated(viewer):
    _wire(viewer, _frame(v=2))
    viewer.runner.compute_metric_scale = lambda result, images, name, **kw: 2.5
    viewer.dd_align.value = "metric_mono"
    viewer._render_cloud()
    assert viewer._result.metric_scale == 2.5 and "metric_mono" in viewer.status.content
    viewer._on_metric_model_change()
    assert viewer._result.metric_scale == 2.5  # recomputed by the cheap update
    viewer.runner.compute_metric_scale = lambda *a, **k: 9.9
    viewer.sl_conf.value = 50
    viewer._cheap_update()
    assert viewer._result.metric_scale == 2.5
    viewer._apply_point_sampling()
    assert viewer._result.metric_scale == 9.9

    def _boom(*a, **k):
        raise RuntimeError("metric model not installed")

    viewer.runner.compute_metric_scale = _boom
    viewer._result.metric_scale = None
    viewer._render_cloud()
    assert "metric_mono" in viewer.status.content


def test_condition_toggle_and_gt_depth_gating(viewer):
    viewer.dd_backbone.value = "vggt"
    viewer._on_backbone_change()
    assert viewer.cb_condition.disabled is True
    viewer.dd_backbone.value = "da3"
    viewer._on_backbone_change()
    assert viewer.cb_condition.disabled is False
    viewer._frame = _frame()
    viewer.dd_backbone.value = "gtdepth"
    viewer.dd_dataset.value = "taco"
    viewer._apply_gt_depth_gating(available=False)
    assert viewer.btn_run.disabled and "gtdepth" in viewer.status.content
    msg = viewer.status.content
    viewer.dd_dataset.value = "synthrobot"
    viewer._apply_gt_depth_gating(available=False)
    assert viewer.status.content != msg
    viewer.dd_backbone.value = "da3"
    viewer._on_backbone_change()
    assert not viewer.btn_run.disabled


def test_occupancy_overlays_and_workspace_crop(viewer):
    frame = _frame(v=2)
    _wire(viewer, frame)
    viewer.dd_align.value = "sim3_points"
    plain = np.array(viewer._cam_scene[0].image, copy=True)
    viewer._toggle_occupancy()
    assert viewer._occupancy_on
    assert not np.array_equal(viewer._cam_scene[0].image, plain)
    assert len(viewer._pred_cam_scene) == 2
    viewer._toggle_occupancy()
    assert not viewer._occupancy_on and viewer._pred_cam_scene == []
    assert np.array_equal(viewer._cam_scene[0].image, plain)
    # a tiny crop box around the origin excludes the cloud (z=2): frustums stay plain.
    viewer.vec_wmin.value = (-0.01, -0.01, -0.01)
    viewer.vec_wmax.value = (0.01, 0.01, 0.01)
    viewer.cb_crop.value = True
    viewer._show_occupancy()
    assert np.array_equal(viewer._cam_scene[0].image, plain)
    # monocular (no predicted poses): GT half still painted, no pred frustums.
    viewer.cb_crop.value = False
    viewer._result.pred_extrinsics = None
    viewer._show_occupancy()
    assert not np.array_equal(viewer._cam_scene[0].image, plain)
    assert viewer._pred_cam_scene == []


def test_pred_camera_overlay(viewer):
    _wire(viewer, _frame(v=2))
    viewer.dd_align.value = "none"
    viewer._render_cloud()
    assert viewer._pred_cam_scene == []
    viewer.cb_pred_cams.value = True
    viewer._render_pred_cameras()
    assert len(viewer._pred_cam_scene) == 2
    assert all(tuple(f.color) == _PRED_COLOR for f in viewer._pred_cam_scene)
    _click_camera(viewer._pred_cam_scene[0])
    assert len(viewer._pred_cam_scene) == 1
    assert not viewer._cam_checks[0].value
    _click_camera(viewer._cam_scene[0])
    assert len(viewer._pred_cam_scene) == 2

    # metric_mono without a fitted scale draws nothing; the GUI callback must not be
    # able to fit one here (it would load a real metric model).
    def _no_metric(*a, **k):
        raise RuntimeError("no metric model in this test")

    viewer.runner.compute_metric_scale = _no_metric
    viewer.dd_align.value = "metric_mono"
    viewer._result.metric_scale = None
    viewer._render_pred_cameras()
    assert viewer._pred_cam_scene == []


def test_ruler_and_click_pick(viewer):
    from types import SimpleNamespace

    viewer._cloud_pts = np.array([[0, 0, 5], [1, 0, 5], [0, 0, -5]], dtype=np.float32)
    viewer.cb_ruler.value = True
    viewer._toggle_ruler()
    assert viewer._tc_a is not None and viewer._ruler_line is not None
    ev = SimpleNamespace(ray_origin=(0.0, 0, 0), ray_direction=(0.0, 0, 1))
    viewer._on_ruler_click(ev)
    np.testing.assert_allclose(np.asarray(viewer._tc_a.position), [0, 0, 5], atol=1e-4)
    viewer.cb_ruler.value = False
    viewer._toggle_ruler()
    assert viewer._tc_a is None and viewer._ruler_picking is False


def test_hands_workspace_and_presets(viewer):
    viewer._frame = _frame(v=2, hands=torch.zeros(1, 21, 3) + torch.tensor([2.0, 3.0, 4.0]))
    viewer.cb_hands.value = True
    viewer._draw_hands()
    assert len(viewer._hand_scene) == 2
    viewer.cb_hands.value = False
    viewer._draw_hands()
    assert viewer._hand_scene == []
    viewer._set_workspace_from_frame()
    lo, hi = np.array(viewer.vec_wmin.value), np.array(viewer.vec_wmax.value)
    assert (lo <= [2.0, 3.0, 4.0]).all() and (hi >= [2.0, 3.0, 4.0]).all()
    viewer._draw_workspace_box()
    assert viewer._workspace_box is not None
    # no hands -> box from the camera rig, which sits far from the origin.
    far = _frame(v=2)
    far.extrinsics[:, :3, 3] = torch.tensor([10.0, 30.0, 1.0])
    viewer._frame = far
    viewer._set_workspace_from_frame()
    assert (np.array(viewer.vec_wmin.value) <= [10.0, 30.0, 1.0]).all()
    # presets: apply + GUI round trip + save.
    from ontic_viz.backbone_viewer.config import ViewConfig, load_presets

    viewer._apply_config(ViewConfig(stride=5, color_mode="per-view", align_mode="prescale_gt"))
    assert viewer.sl_stride.value == 5 and viewer.dd_align.value == "prescale_gt"
    assert viewer._current_config().stride == 5
    viewer.tb_preset.value = "mine"
    viewer._save_preset()
    assert load_presets(viewer.presets_path)["mine"].stride == 5
    assert "genesis_default" in viewer._configs


def test_robot_overlay_hides_frames_without_joint_angles(viewer):
    viewer._frame = _frame()
    viewer.cb_robot.value = True
    viewer._draw_robot()
    assert viewer.cb_robot.value is True and viewer._robot_scene == {}
    assert "No robot pose" in viewer.md_robot.content
    viewer.cb_action_pts.value = True
    viewer._draw_action_points()
    assert viewer.cb_action_pts.value is False and viewer._action_pts_scene == []


def test_estimated_robot_pose_is_removed_when_scrubbing_and_restored_on_return(viewer):
    from types import SimpleNamespace

    geom = SimpleNamespace(
        name="arm",
        vertices=np.eye(3, dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        color=(1.0, 1.0, 1.0),
    )
    viewer._robot_model = lambda: SimpleNamespace(
        link_geoms=[geom],
        geom_world_poses=lambda *args: {"arm": (np.zeros(3), np.array([1.0, 0, 0, 0]))},
    )
    fitted = _frame()
    fitted.robot = {
        "qpos": {},
        "base_pose": [0, 0, 0, 1, 0, 0, 0],
        "source": "image_fit",
        "frame_index": 0,
    }
    viewer._frame = fitted
    viewer.cb_robot.value = True
    viewer._draw_robot()
    assert len(viewer._robot_scene) == 1
    assert "estimated pose" in viewer.md_robot.content
    viewer._frame = _frame()
    viewer._draw_robot()
    assert viewer._robot_scene == {} and viewer.cb_robot.value
    viewer._frame = fitted
    viewer._draw_robot()
    assert len(viewer._robot_scene) == 1


# --------------------------------------------------------------------------- #
# End to end: gtdepth backbone on one real frame (frontier venv, CUDA, data root)
# --------------------------------------------------------------------------- #


def _has(name):
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def _dinov2_cached() -> bool:
    hub = (
        Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
    )
    return any(hub.glob("dinov2_vitb14*")) if hub.exists() else False


def _first_depth_dataset_root():
    for name, root in DEFAULT_ROOTS.items():
        if dataset_has_gt_depth(name) and os.access(root, os.R_OK):
            return name, root
    return None, None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.skipif(not _has("fwomo_3d"), reason="frontier venv only")
@pytest.mark.skipif(
    not (_has("ontic_nn.wrappers") and _has("ontic_data")),
    reason="needs ontic_nn.wrappers + ontic_data",
)
@pytest.mark.skipif(not _dinov2_cached(), reason="DINOv2 vitb14 weights not cached")
def test_end_to_end_gtdepth_on_one_frame(tmp_path):
    name, root = _first_depth_dataset_root()
    if name is None:
        pytest.skip("no readable dataset root with GT depth")
    server = viser.ViserServer(port=0, verbose=False)
    try:
        v = BackboneViewer(
            server, device="cuda", roots={name: root}, presets_path=tmp_path / "p.json"
        )
        v.dd_dataset.value = name
        v.dd_backbone.value = "gtdepth"
        v._load_dataset()
        assert v._frame is not None and v._frame.depth is not None, v.status.content
        ix = v._selected_ix()[:4]
        v._set_all(False)
        for i in ix:
            v._cam_checks[i].value = True
        v._run()
        assert v._result is not None, v.status.content
        assert v._cloud_pts is not None and v._cloud_pts.shape[0] > 0, v.status.content
        assert "points" in v.status.content
    finally:
        server.stop()
