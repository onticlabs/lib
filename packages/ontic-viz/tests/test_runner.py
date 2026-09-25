"""Runner contract on synthetic backbone / metric-model outputs (no weights, no GPU)."""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass, field

import pytest
import torch
from torch import Tensor

from ontic_viz.backbone_viewer.data_source import Frame
from ontic_viz.backbone_viewer.runner import (
    ALIGN_MODES,
    AlignMode,
    BackboneResult,
    BackboneRunner,
    backbone_accepts_gt_cameras,
    backbone_is_monocular,
    backbone_needs_gt_depth,
    effective_align_mode,
    metric_unproject,
)


@dataclass
class _Output:
    """Stand-in for ``ontic_nn.wrappers.BackboneOutput`` (the runner reads ``.data``)."""

    data: dict = field(default_factory=dict)


@dataclass
class _MetricOutput:
    depth: Tensor
    conf: Tensor | None = None
    intrinsics: Tensor | None = None


class DummyBackbone(torch.nn.Module):
    accepts_gt_cameras = True

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.calls: list[tuple] = []
        self.cfg = type("Cfg", (), {"long_side": 504})()

    def forward(self, images, extrinsics=None, intrinsics=None, depth=None):
        assert torch.is_inference_mode_enabled()
        self.calls.append((extrinsics, intrinsics))
        b, v, _, h, w = images.shape
        hd, wd = h // 2, w // 2
        return _Output(
            {
                "depth": torch.ones(b, v, hd, wd),
                "depth_conf": torch.full((b, v, hd, wd), 2.0),
                "extrinsics_pred": torch.eye(4).expand(b, v, 4, 4).clone(),
                "intrinsics_pred": torch.eye(3).expand(b, v, 3, 3).clone(),
            }
        )


class PoseFreeDummy(DummyBackbone):
    accepts_gt_cameras = False


class DummyMetric(torch.nn.Module):
    requires_intrinsics = False

    def __init__(self, value=6.0):
        super().__init__()
        self.value = value
        self.intr_seen = "unset"

    def forward(self, images, intrinsics=None):
        self.intr_seen = intrinsics
        b, v, _, h, w = images.shape
        return _MetricOutput(depth=torch.full((b, v, h // 2, w // 2), self.value))


def _frame(v=2, hw=28):
    return Frame(
        images=torch.rand(v, 3, hw, hw),
        intrinsics=torch.eye(3).expand(v, 3, 3).clone(),
        extrinsics=torch.eye(4).expand(v, 4, 4).clone(),
        cam_names=[f"c{i}" for i in range(v)],
    )


def _runner(backbone=DummyBackbone, **kw):
    return BackboneRunner(device="cpu", builder=lambda name: backbone(), **kw)


def test_align_modes_and_hints():
    assert ALIGN_MODES == ["none", "sim3_points", "prescale_gt", "metric_mono"]
    assert backbone_accepts_gt_cameras("da3") and not backbone_accepts_gt_cameras("vggt")
    assert backbone_needs_gt_depth("gtdepth") and not backbone_needs_gt_depth("da3")
    assert backbone_is_monocular("moge3") and not backbone_is_monocular("da3")
    assert backbone_accepts_gt_cameras("unknown") is False


class TestRunnerContract:
    def test_default_passes_no_cameras(self):
        r = _runner()
        r.load("da3")
        r.run(_frame(), [0, 1])
        assert r._backbone.calls[-1] == (None, None)

    def test_gt_cameras_fed_only_when_requested_and_accepted(self):
        r = _runner()
        r.load("da3")
        res = r.run(_frame(), [0, 1], condition_on_gt_cameras=True)
        ext, intr = r._backbone.calls[-1]
        assert ext.shape == (1, 2, 4, 4) and intr.shape == (1, 2, 3, 3) and res.conditioned
        r2 = _runner(PoseFreeDummy)
        r2.load("vggt")
        res2 = r2.run(_frame(), [0, 1], condition_on_gt_cameras=True)
        assert r2._backbone.calls[-1] == (None, None) and not res2.conditioned

    def test_result_shapes_cpu_and_selected_cameras(self):
        r = _runner()
        r.load("da3")
        res = r.run(_frame(v=4), [1, 3])
        assert isinstance(res, BackboneResult)
        assert res.depth.shape[0] == 2 and res.depth.device.type == "cpu"
        assert not res.depth.requires_grad
        assert res.pred_extrinsics.shape == (2, 4, 4) and res.gt_intrinsics.shape == (2, 3, 3)
        torch.testing.assert_close(res.depth, torch.ones_like(res.depth))  # raw depth

    def test_long_side_override_survives_switch_and_restores(self):
        r = _runner(long_side=280)
        r.load("da3")
        assert r._backbone.cfg.long_side == 280
        r.load("ma")
        assert r._backbone.cfg.long_side == 280
        r.set_long_side(None)
        assert r._backbone.cfg.long_side == 504

    def test_reload_is_noop_and_switch_frees(self, monkeypatch):
        calls = {"n": 0}
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            torch.cuda, "empty_cache", lambda: calls.__setitem__("n", calls["n"] + 1)
        )
        r = _runner()
        r.load("da3")
        inst = r._backbone
        r.load("da3")
        assert r._backbone is inst
        ref = weakref.ref(inst)
        del inst
        r.load("ma")
        gc.collect()
        assert ref() is None and calls["n"] >= 1

    def test_gtdepth_refuses_missing_depth(self):
        r = _runner()
        r.load("gtdepth")
        with pytest.raises(ValueError, match="(?i)gt depth"):
            r.run(_frame(), [0, 1], depth=None)

    def test_run_without_load_raises(self):
        with pytest.raises(RuntimeError):
            _runner().run(_frame(), [0])


def _aligned_result(v=3, hd=4, wd=4, depth_val=2.0, scale=1.0):
    """Predicted cameras are ``scale`` x the GT cameras (spread along x)."""
    intr = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]).expand(v, 3, 3).clone()
    centers = torch.zeros(v, 3)
    centers[:, 0] = torch.arange(v).float()
    gt = torch.eye(4).expand(v, 4, 4).clone()
    gt[:, :3, 3] = centers
    pred = gt.clone()
    pred[:, :3, 3] = scale * centers
    return BackboneResult(
        depth=torch.full((v, hd, wd), depth_val),
        conf=torch.full((v, hd, wd), 3.0),
        pred_extrinsics=pred,
        pred_intrinsics=intr.clone(),
        gt_extrinsics=gt,
        gt_intrinsics=intr.clone(),
    )


class TestMetricUnproject:
    def test_none_and_string_mode(self):
        res = _aligned_result(scale=0.5)
        rgb = torch.zeros(3, 3, 4, 4)
        assert metric_unproject(res, rgb, "none").points.shape == (48, 3)
        assert metric_unproject(res, rgb, AlignMode.NONE).colors.shape == (48, 3)

    def test_sim3_and_prescale_rescale_the_cloud(self):
        res = _aligned_result(scale=0.5, depth_val=2.0)
        rgb = torch.zeros(3, 3, 4, 4)
        none = metric_unproject(res, rgb, AlignMode.NONE).points
        sim3 = metric_unproject(res, rgb, AlignMode.SIM3_POINTS).points
        pre = metric_unproject(res, rgb, AlignMode.PRESCALE_GT).points
        assert not torch.allclose(none, sim3)
        assert pre[:, 2].max() > 1.9 * none[:, 2].max()

    def test_metric_mono_requires_scale_then_is_a_similarity(self):
        res = _aligned_result(depth_val=2.0)
        rgb = torch.zeros(3, 3, 4, 4)
        with pytest.raises(RuntimeError):
            metric_unproject(res, rgb, AlignMode.METRIC_MONO)
        none = metric_unproject(res, rgb, AlignMode.NONE).points
        res.metric_scale = 3.0
        before = res.pred_extrinsics.clone()
        mono = metric_unproject(res, rgb, AlignMode.METRIC_MONO).points
        i, j = 0, none.shape[0] // 2
        torch.testing.assert_close((mono[i] - mono[j]).norm(), 3.0 * (none[i] - none[j]).norm())
        torch.testing.assert_close(res.pred_extrinsics, before)  # not mutated

    def test_metric_mono_with_true_scale_matches_prescale_gt(self):
        res = _aligned_result(depth_val=2.0, scale=0.5)
        rgb = torch.zeros(3, 3, 4, 4)
        res.metric_scale = 2.0
        mono = metric_unproject(res, rgb, AlignMode.METRIC_MONO).points
        pre = metric_unproject(res, rgb, AlignMode.PRESCALE_GT).points
        torch.testing.assert_close(mono, pre, atol=1e-4, rtol=1e-4)

    def test_umeyama_modes_degrade_to_none_without_predicted_poses(self):
        res = _aligned_result()
        res.pred_extrinsics = None
        assert effective_align_mode(res, "sim3_points") is AlignMode.NONE
        assert effective_align_mode(res, "prescale_gt") is AlignMode.NONE
        assert effective_align_mode(res, "metric_mono") is AlignMode.METRIC_MONO
        pc = metric_unproject(res, torch.zeros(3, 3, 4, 4), "sim3_points")  # lifts with GT cams
        assert pc.points.shape == (48, 3)

    def test_stride_and_conf_threshold(self):
        res = _aligned_result(v=1, hd=8, wd=8)
        res.conf[0, :4] = 0.0
        pc = metric_unproject(res, torch.zeros(1, 3, 8, 8), "none", stride=2, conf_thresh=1.0)
        assert pc.points.shape[0] == 8


class TestComputeMetricScale:
    def test_fits_scale_and_forwards_gt_intrinsics(self):
        metric = DummyMetric(6.0)
        r = _runner(metric_builder=lambda n: metric)
        r.load("vggt")
        frame = _frame(v=2, hw=28)
        res = r.run(frame, [0, 1])  # raw depth 1.0
        s = r.compute_metric_scale(res, frame.images[torch.tensor([0, 1])], "da3")
        assert abs(s - 6.0) < 1e-4
        assert metric.intr_seen.shape == (1, 2, 3, 3)

    def test_fit_respects_confidence_mask(self):
        r = _runner(metric_builder=lambda n: DummyMetric(6.0))
        eye_e = torch.eye(4).expand(1, 4, 4).clone()
        eye_i = torch.eye(3).expand(1, 3, 3).clone()
        res = BackboneResult(
            depth=torch.tensor([[[1.0, 1.0], [10.0, 10.0]]]),
            conf=torch.tensor([[[5.0, 5.0], [1.0, 1.0]]]),
            pred_extrinsics=eye_e,
            pred_intrinsics=eye_i,
            gt_extrinsics=eye_e,
            gt_intrinsics=eye_i,
        )
        images = torch.rand(1, 3, 8, 8)
        s_all = r.compute_metric_scale(res, images, "da3")
        s_conf = r.compute_metric_scale(res, images, "da3", conf_thresh=2.0)
        assert abs(s_conf - 6.0) < 1e-4 and s_all < s_conf - 1.0

    def test_switching_metric_model_rebuilds_only_on_change(self):
        built = []
        r = _runner(metric_builder=lambda n: built.append(n) or DummyMetric())
        r._load_metric("da3")
        r._load_metric("da3")
        r._load_metric("unidepth")
        assert built == ["da3", "unidepth"]


def test_checkpoint_change_rebuilds_model_and_changes_geometry_cache_key():
    builds = []

    def builder(name, **options):
        builds.append((name, options))
        return DummyBackbone()

    runner = BackboneRunner(device="cpu", builder=builder)
    runner.configure("da3", checkpoint_path="/models/first.pth")
    key = runner.model_key("da3")
    runner.load("da3")
    first = runner._backbone
    runner.configure("da3", checkpoint_path="/models/first.pth")
    runner.load("da3")
    assert runner._backbone is first and len(builds) == 1
    runner.configure("da3", checkpoint_path="/models/second.pth")
    assert runner.model_key("da3") != key
    runner.load("da3")
    assert runner._backbone is not first and len(builds) == 2
    assert builds[-1][1] == {"checkpoint_path": "/models/second.pth", "allow_download": False}
