"""Unit tests for ontic_lib.camera.occupancy (+ frontier parity in the frontier venv)."""

import importlib

import pytest
import torch

from ontic_lib.camera import overlay_masks_on_images, points_with_radius, project_occupancy


def _has(name):
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def _intrinsics(views=1, fx=1.0, fy=1.0):
    k = torch.eye(3)
    k[0, 0], k[1, 1], k[0, 2], k[1, 2] = fx, fy, 0.5, 0.5
    return k.unsqueeze(0).repeat(views, 1, 1)


def _inputs(views=1, height=4, width=4, depth_value=2.0):
    return (
        torch.full((views, height, width), depth_value),
        torch.eye(4).expand(views, 4, 4).clone(),
        _intrinsics(views),
        torch.full((views, height, width), 3.0),
    )


def test_points_with_radius_from_source_depth_and_focal():
    # fx_px = fy_px = 4 -> radius = depth * 0.5 * (1/4 + 1/4) = depth / 4.
    points, radius = points_with_radius(*_inputs(depth_value=2.0))
    assert points.shape == (16, 3) and radius.shape == (16,)
    torch.testing.assert_close(radius, torch.full((16,), 0.5))


def test_points_with_radius_stride_and_confidence():
    depth, c2w, k, conf = _inputs(height=8, width=8)
    conf[0, :4] = 0.0  # top half dropped
    points, radius = points_with_radius(depth, c2w, k, conf, stride=2, confidence_threshold=1.0)
    assert points.shape[0] == 8 and radius.shape[0] == 8


def test_points_with_radius_excludes_holes_and_hidden_views():
    from ontic_lib.structures import pointcloud_from_depth_views

    depth, cameras, intrinsics, confidence = _inputs(views=2)
    depth[0] = 0  # hidden camera
    depth[1, 0, :3] = torch.tensor([float("nan"), -1, 0])
    points, radius = points_with_radius(depth, cameras, intrinsics, confidence)
    cloud = pointcloud_from_depth_views(
        depth, cameras, intrinsics, confidence=confidence, minimum_depth=0
    )
    torch.testing.assert_close(points, cloud.points)
    assert points.shape == (13, 3) and torch.isfinite(radius).all() and (radius > 0).all()


def test_project_occupancy_marks_centre_and_drops_points_behind():
    point = torch.tensor([[0.0, 0.0, 2.0]])
    masks = project_occupancy(
        point, torch.tensor([0.1]), torch.eye(4)[None], _intrinsics(), (20, 20)
    )
    assert masks.shape == (1, 20, 20) and masks.dtype == torch.bool
    assert masks[0, 10, 10]
    behind = project_occupancy(
        -point, torch.tensor([0.1]), torch.eye(4)[None], _intrinsics(), (20, 20)
    )
    assert behind.sum() == 0


def test_project_occupancy_splat_scale_and_max_radius():
    point = torch.tensor([[0.0, 0.0, 2.0]])
    radius = torch.tensor([0.1])
    args = (point, radius, torch.eye(4)[None], _intrinsics(), (40, 40))
    small = project_occupancy(*args, splat_scale=1.0)
    big = project_occupancy(*args, splat_scale=4.0)
    assert big.sum() > small.sum()
    clamped = project_occupancy(*args, splat_scale=100.0, max_radius=3)
    assert clamped.sum() <= 3.15 * (3 + 1) ** 2


def test_overlay_tints_masked_pixels_only():
    images = torch.zeros(2, 3, 8, 8)
    masks = torch.zeros(2, 8, 8, dtype=torch.bool)
    masks[0, 2, 3] = True
    out = overlay_masks_on_images(images, masks, color=(0.0, 1.0, 0.0), alpha=0.5)
    assert out.shape == (2, 3, 8, 8)
    assert out[0, 1, 2, 3] == pytest.approx(0.5)
    assert float(out[0, :, 0, 0].sum()) == 0.0 and float(out[1].sum()) == 0.0


@pytest.mark.skipif(not _has("fwomo_3d.viz_lib.backbone_viewer.occupancy"), reason="needs fwomo_3d")
class TestFrontierParity:
    def test_matches_frontier_occupancy(self):
        import numpy as np
        from fwomo_3d.viz_lib.backbone_viewer import occupancy as F

        g = torch.Generator().manual_seed(0)
        views, height, width = 3, 12, 16
        depth = torch.rand(views, height, width, generator=g) * 3 + 0.5
        conf = torch.rand(views, height, width, generator=g)
        c2w = torch.eye(4).expand(views, 4, 4).clone()
        c2w[:, :3, 3] = torch.randn(views, 3, generator=g) * 0.2
        k = _intrinsics(views, fx=0.9, fy=1.1)

        ours = points_with_radius(depth, c2w, k, conf, stride=2, confidence_threshold=0.3)
        theirs = F.points_with_radius(depth, c2w, k, conf, stride=2, conf_thresh=0.3)
        torch.testing.assert_close(ours[0], theirs[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(ours[1], theirs[1], atol=1e-6, rtol=1e-6)

        for scale, max_radius in ((1.0, 6), (3.0, 4), (0.2, 6)):
            mask_ours = project_occupancy(
                *ours, c2w, k, (height, width), splat_scale=scale, max_radius=max_radius
            )
            mask_theirs = F.project_occupancy(
                *theirs, c2w, k, (height, width), splat_scale=scale, max_radius=max_radius
            )
            assert np.array_equal(mask_ours.numpy(), mask_theirs)

        images = torch.rand(views, 3, height, width, generator=g)
        torch.testing.assert_close(
            overlay_masks_on_images(images, mask_ours),
            F.overlay_masks_on_images(images, mask_theirs),
        )
