"""Unit tests for ontic_lib.pointops.pointcloud."""

import pytest
import torch

from ontic_lib.pointops import (
    PointCloud,
    aabb_mask,
    crop_to_aabb,
    furthest_point_indices,
    furthest_point_sample,
    nearest_point_to_ray,
    pointcloud_from_depth_views,
    space_filling_stride,
    voxel_pool,
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


def test_voxel_pool_reduces_duplicates_within_voxel():
    # Two clusters that fall in two distinct voxels; each collapses to its mean.
    points = torch.tensor(
        [
            [0.1, 0.1, 0.1],
            [0.2, 0.2, 0.2],
            [5.1, 5.1, 5.1],
            [5.3, 5.3, 5.3],
        ]
    )
    feats = torch.tensor([[1.0], [3.0], [10.0], [20.0]])
    pooled, pooled_feats = voxel_pool(points, feats, voxel_size=1.0)
    assert pooled.shape[0] == 2
    # rows are sorted by voxel id (unique) -> first voxel near origin.
    means = pooled.tolist()
    assert pytest.approx(means[0], abs=1e-6) == [0.15, 0.15, 0.15]
    assert pytest.approx(pooled_feats[0].tolist(), abs=1e-6) == [2.0]
    assert pytest.approx(pooled_feats[1].tolist(), abs=1e-6) == [15.0]


def test_voxel_pool_noop_when_size_nonpositive():
    points = torch.randn(5, 3)
    out_pts, out_feats = voxel_pool(points, None, voxel_size=0.0)
    assert out_pts is points and out_feats is None


def test_fps_count_first_point_and_spread():
    g = torch.Generator().manual_seed(0)
    points = torch.rand(200, 3, generator=g)
    k = 16
    idx = furthest_point_indices(points, k)
    assert idx.shape == (k,)
    # First point convention: index 0 is always selected first.
    assert int(idx[0]) == 0

    fps_pts, _ = furthest_point_sample(points, None, k)
    assert fps_pts.shape == (k, 3)

    # FPS spread (min pairwise distance) should beat a random subsample's spread.
    def min_pairwise(p):
        d = torch.cdist(p, p)
        d.fill_diagonal_(float("inf"))
        return d.min()

    rand_idx = torch.randperm(points.shape[0], generator=torch.Generator().manual_seed(1))[:k]
    assert min_pairwise(fps_pts) > min_pairwise(points[rand_idx])


def test_fps_rejects_negative_count():
    with pytest.raises(ValueError):
        furthest_point_indices(torch.randn(4, 3), -1)


def test_fps_clamps_to_available_points():
    points = torch.randn(3, 3)
    idx = furthest_point_indices(points, 10)
    assert idx.shape == (3,)


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
    pc = pointcloud_from_depth_views(depth, c2w, intrinsics, confidence=confidence,
                                     confidence_threshold=0.5)
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


def test_space_filling_stride_thins_cloud():
    g = torch.Generator().manual_seed(0)
    points = torch.rand(100, 3, generator=g)
    kept, _ = space_filling_stride(points, None, stride=4, grid_size=0.05)
    assert kept.shape[0] == -(-100 // 4)  # ceil(100 / 4)
    # stride <= 1 is a no-op passthrough.
    same, _ = space_filling_stride(points, None, stride=1, grid_size=0.05)
    assert torch.equal(same, points)
