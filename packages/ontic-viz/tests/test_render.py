"""Render helpers: colour / point assembly shapes, frustum math, predicted-camera frames."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ontic_viz.backbone_viewer.render import (
    COLOR_MODES,
    build_point_cloud,
    conf_to_rgb,
    frustum_params,
    mat_to_wxyz,
    pred_cameras_in_display_frame,
    view_palette_image,
)
from ontic_viz.backbone_viewer.runner import BackboneResult


def _intr(v=1):
    k = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    return k.unsqueeze(0).repeat(v, 1, 1)


def _result(v=1, hd=4, wd=4, depth_val=2.0):
    eye = torch.eye(4).expand(v, 4, 4).clone()
    return BackboneResult(
        depth=torch.full((v, hd, wd), depth_val),
        conf=torch.full((v, hd, wd), 3.0),
        pred_extrinsics=eye.clone(),
        pred_intrinsics=_intr(v),
        gt_extrinsics=eye.clone(),
        gt_intrinsics=_intr(v),
    )


def test_mat_to_wxyz_identity_and_rotation():
    np.testing.assert_allclose(mat_to_wxyz(np.eye(3)), [1.0, 0, 0, 0], atol=1e-6)
    r = np.array([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])  # 90 deg about z
    w, x, y, z = mat_to_wxyz(r)
    assert abs(w - np.cos(np.pi / 4)) < 1e-6 and abs(z - np.sin(np.pi / 4)) < 1e-6
    assert abs(x) < 1e-6 and abs(y) < 1e-6


def test_frustum_params_square_sensor():
    fov, aspect = frustum_params(_intr()[0])
    assert abs(aspect - 1.0) < 1e-6 and abs(fov - 2 * np.arctan(0.5)) < 1e-6


def test_color_images():
    rgb = conf_to_rgb(torch.rand(3, 4, 5) * 10 + 1)
    assert rgb.shape == (3, 3, 4, 5) and rgb.min() >= 0 and rgb.max() <= 1
    img = view_palette_image(2, (4, 4))
    assert img.shape == (2, 3, 4, 4)
    assert not torch.allclose(img[0, :, 0, 0], img[1, :, 0, 0])
    assert torch.allclose(img[0], img[0, :, 0, 0].view(3, 1, 1).expand_as(img[0]))


@pytest.mark.parametrize("mode", COLOR_MODES)
def test_build_point_cloud_shapes_and_dtypes(mode):
    res = _result(v=2)
    images = torch.zeros(2, 3, 4, 4)
    images[:, 0] = 1.0
    pts, cols = build_point_cloud(res, images, color_mode=mode)
    assert pts.shape == (32, 3) and pts.dtype == np.float32
    assert cols.shape == (32, 3) and cols.dtype == np.uint8
    if mode == "RGB":
        assert (cols[:, 0] > 200).all() and (cols[:, 1] < 50).all()
    if mode == "per-view":
        assert not np.array_equal(cols[0], cols[16])


def test_point_count_controls():
    res = _result(v=1, hd=8, wd=8)
    images = torch.zeros(1, 3, 8, 8)
    kw = dict(color_mode="RGB", stride=1, conf_thresh=0.0)
    full, _ = build_point_cloud(res, images, **kw)
    assert full.shape == (64, 3)
    assert build_point_cloud(res, images, color_mode="RGB", stride=2)[0].shape == (16, 3)
    pooled, cols = build_point_cloud(res, images, voxel_size=1.0, **kw)
    assert 1 <= pooled.shape[0] < 64 and cols.shape[0] == pooled.shape[0]
    strided, _ = build_point_cloud(res, images, sfc_stride=4, **kw)
    assert strided.shape[0] == 16
    fps, cols = build_point_cloud(res, images, max_points=20, **kw)
    assert fps.shape[0] == 20 and cols.shape[0] == 20
    cropped, _ = build_point_cloud(
        res,
        images,
        crop_lo=torch.tensor([-0.2, -0.2, 0.0]),
        crop_hi=torch.tensor([0.2, 0.2, 3.0]),
        **kw,
    )
    assert 0 < cropped.shape[0] < 64


def test_alignment_modes_through_build_point_cloud():
    centers = torch.tensor([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    pred = torch.eye(4).expand(4, 4, 4).clone()
    pred[:, :3, 3] = centers
    gt = pred.clone()
    gt[:, 0, 3] += 5.0
    res = _result(v=4)
    res.pred_extrinsics, res.gt_extrinsics = pred, gt
    images = torch.zeros(4, 3, 4, 4)
    pts_no, _ = build_point_cloud(res, images, align_mode="none")
    pts_al, _ = build_point_cloud(res, images, align_mode="sim3_points")
    assert np.abs((pts_al - pts_no) - np.array([5.0, 0.0, 0.0])).max() < 1e-3
    with pytest.raises(RuntimeError):
        build_point_cloud(res, images, align_mode="metric_mono")
    res.metric_scale = 3.0
    mono, _ = build_point_cloud(res, images, align_mode="metric_mono")
    assert mono.shape == pts_no.shape


def _result_pred_offset(v=3, dx=4.0):
    gt = torch.eye(4).expand(v, 4, 4).clone()
    gt[:, 0, 3] = torch.arange(v).float()
    pred = gt.clone()
    pred[:, 0, 3] += dx
    return BackboneResult(
        depth=torch.full((v, 4, 4), 2.0),
        conf=torch.full((v, 4, 4), 3.0),
        pred_extrinsics=pred,
        pred_intrinsics=_intr(v),
        gt_extrinsics=gt,
        gt_intrinsics=_intr(v),
    )


class TestPredCamerasInDisplayFrame:
    def test_none_returns_predicted_poses(self):
        res = _result_pred_offset()
        ext, intr = pred_cameras_in_display_frame(res, "none")
        torch.testing.assert_close(ext, res.pred_extrinsics)
        assert intr.shape == (3, 3, 3)

    def test_sim3_and_metric_mono_land_on_gt(self):
        res = _result_pred_offset()
        ext, _ = pred_cameras_in_display_frame(res, "sim3_points")
        torch.testing.assert_close(ext, res.gt_extrinsics, atol=1e-3, rtol=1e-3)
        assert pred_cameras_in_display_frame(res, "metric_mono") is None
        res.metric_scale = 1.0
        ext, _ = pred_cameras_in_display_frame(res, "metric_mono")
        torch.testing.assert_close(ext, res.gt_extrinsics, atol=1e-4, rtol=1e-4)

    def test_metric_mono_poses_move_with_scale(self):
        res = _result_pred_offset()
        res.gt_extrinsics[:, :3, 3] *= 2.0
        res.metric_scale = 1.0
        ext1, _ = pred_cameras_in_display_frame(res, "metric_mono")
        res.metric_scale = 2.0
        ext2, _ = pred_cameras_in_display_frame(res, "metric_mono")
        assert not torch.allclose(ext1, ext2)

    def test_none_without_predicted_poses(self):
        res = _result_pred_offset()
        res.pred_extrinsics = None
        assert pred_cameras_in_display_frame(res, "sim3_points") is None
