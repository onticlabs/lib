"""Unit tests for ontic_lib.pointops.sampling."""

import pytest
import torch

from ontic_lib.pointops import (
    furthest_point_indices,
    furthest_point_sample,
    space_filling_stride,
    voxel_pool,
)


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


def test_space_filling_stride_thins_cloud():
    g = torch.Generator().manual_seed(0)
    points = torch.rand(100, 3, generator=g)
    kept, _ = space_filling_stride(points, None, stride=4, grid_size=0.05)
    assert kept.shape[0] == -(-100 // 4)  # ceil(100 / 4)
    # stride <= 1 is a no-op passthrough.
    same, _ = space_filling_stride(points, None, stride=1, grid_size=0.05)
    assert torch.equal(same, points)
