"""Unit tests for ontic_lib.structures.pointcloud."""

import pytest
import torch

from ontic_lib.structures import (
    PointCloud,
    aabb_mask,
    crop_to_aabb,
    nearest_point_to_ray,
    pointcloud_from_depth_views,
)


def test_pointcloud_defaults_colors_none():
    pts = torch.zeros(3, 3)
    pc = PointCloud(points=pts)
    assert pc.colors is None
    assert torch.equal(pc.points, pts)


def test_aabb_mask_handbuilt():
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],  # inside
            [0.5, 0.5, 0.5],  # inside (on boundary handled below)
            [2.0, 0.0, 0.0],  # outside (x too big)
            [-1.0, 0.0, 0.0],  # outside (x too small)
            [1.0, 1.0, 1.0],  # on the inclusive upper corner -> inside
        ]
    )
    lo = torch.tensor([0.0, 0.0, 0.0])
    hi = torch.tensor([1.0, 1.0, 1.0])
    mask = aabb_mask(points, lo, hi)
    assert mask.shape == (5, 1)
    assert mask.squeeze(-1).tolist() == [True, True, False, False, True]


def test_crop_to_aabb_filters_points_and_features():
    points = torch.tensor([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.2, 0.2, 0.2]])
    feats = torch.tensor([[10.0], [20.0], [30.0]])
    lo = torch.tensor([0.0, 0.0, 0.0])
    hi = torch.tensor([1.0, 1.0, 1.0])
    kept_pts, kept_feats = crop_to_aabb(points, feats, lo, hi)
    assert torch.equal(kept_pts, points[[0, 2]])
    assert torch.equal(kept_feats, feats[[0, 2]])


def test_crop_to_aabb_noop_when_bounds_none():
    points = torch.randn(4, 3)
    out_pts, out_feats = crop_to_aabb(points, None, None, None)
    assert out_pts is points and out_feats is None


def test_nearest_point_to_ray_handbuilt():
    points = torch.tensor(
        [
            [1.0, 0.05, 0.0],  # near the +x ray, in front -> should win
            [1.0, 2.0, 0.0],  # far off the ray line
            [-1.0, 0.0, 0.0],  # behind the origin -> excluded
        ]
    )
    origin = [0.0, 0.0, 0.0]
    direction = [1.0, 0.0, 0.0]
    hit = nearest_point_to_ray(points, origin, direction)
    assert hit is not None
    assert torch.equal(hit, points[0])


def test_nearest_point_to_ray_maxdist_rejects():
    points = torch.tensor([[1.0, 0.5, 0.0]])
    hit = nearest_point_to_ray(points, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], maximum_distance=0.1)
    assert hit is None


def test_nearest_point_to_ray_empty_cloud():
    assert nearest_point_to_ray(torch.zeros(0, 3), [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]) is None


def _tiny_camera(v=1, h=4, w=4):
    c2w = torch.eye(4).unsqueeze(0).repeat(v, 1, 1)
    intrinsics = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    intrinsics = intrinsics.unsqueeze(0).repeat(v, 1, 1)
    return c2w, intrinsics


def test_pointcloud_from_depth_views_count_and_colors():
    c2w, intrinsics = _tiny_camera()
    depth = torch.ones(1, 4, 4)
    rgb = torch.rand(1, 3, 4, 4)  # channels-first, per the documented convention
    pc = pointcloud_from_depth_views(depth, c2w, intrinsics, rgb=rgb)
    assert pc.points.shape == (16, 3)
    assert pc.colors.shape == (16, 3)
    assert torch.isfinite(pc.points).all()


def test_pointcloud_from_depth_views_stride():
    c2w, intrinsics = _tiny_camera()
    depth = torch.ones(1, 4, 4)
    pc = pointcloud_from_depth_views(depth, c2w, intrinsics, stride=2)
    assert pc.points.shape == (4, 3)
    assert pc.colors is None


def test_pointcloud_from_depth_views_confidence_threshold():
    c2w, intrinsics = _tiny_camera()
    depth = torch.ones(1, 4, 4)
    confidence = torch.zeros(1, 4, 4)
    confidence[0, :2, :] = 0.9  # top half passes
    pc = pointcloud_from_depth_views(
        depth, c2w, intrinsics, confidence=confidence, confidence_threshold=0.5
    )
    assert pc.points.shape[0] == int((confidence > 0.5).sum())


def test_pointcloud_from_depth_views_minimum_depth():
    c2w, intrinsics = _tiny_camera()
    depth = torch.linspace(0.1, 1.6, 16).reshape(1, 4, 4)
    pc = pointcloud_from_depth_views(depth, c2w, intrinsics, minimum_depth=0.5)
    assert pc.points.shape[0] == int((depth > 0.5).sum())


def test_pointcloud_from_depth_views_channels_last_rgb_raises():
    # Documented quirk: rgb is expected channels-FIRST (V, 3, H, W); a channels-last
    # (V, H, W, 3) tensor is malformed for the interpolate/permute/reshape chain and
    # is rejected rather than silently producing wrong colors.
    c2w, intrinsics = _tiny_camera()
    depth = torch.ones(1, 4, 4)
    rgb = torch.rand(1, 4, 4, 3)
    with pytest.raises((IndexError, RuntimeError)):
        pointcloud_from_depth_views(depth, c2w, intrinsics, rgb=rgb)


def test_padded_aabb_spans_points_with_margin():
    from ontic_lib.structures import padded_aabb

    points = torch.tensor([[[0.0, 1.0, 2.0], [3.0, -1.0, 0.5]]])  # (1, 2, 3) leading dims
    lo, hi = padded_aabb(points, margin=0.15)
    torch.testing.assert_close(lo, torch.tensor([-0.15, -1.15, 0.35]))
    torch.testing.assert_close(hi, torch.tensor([3.15, 1.15, 2.15]))
    lo0, hi0 = padded_aabb(points)
    assert torch.equal(lo0, points.reshape(-1, 3).amin(0)) and torch.equal(
        hi0, points.reshape(-1, 3).amax(0)
    )


def test_padded_aabb_matches_frontier_hand_bbox():
    import importlib

    try:
        frontier = importlib.import_module("fwomo_3d.utils.pointcloud")
    except Exception:
        pytest.skip("needs fwomo_3d")
    from ontic_lib.structures import padded_aabb

    hands = torch.rand(2, 21, 3, generator=torch.Generator().manual_seed(0))
    ours, theirs = padded_aabb(hands, margin=0.15), frontier.hand_bbox(hands, margin=0.15)
    assert torch.equal(ours[0], theirs[0]) and torch.equal(ours[1], theirs[1])
