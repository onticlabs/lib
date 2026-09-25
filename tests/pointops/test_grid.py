"""Unit tests for ontic_lib.pointops.grid."""

import pytest
import torch

from ontic_lib.pointops.grid import cluster_reduce, code_clusters, grid_clusters, voxel_coords


def test_voxel_coords_pins_non_finite_and_clamps():
    coord = torch.tensor(
        [
            [0.5, 0.5, 0.5],
            [float("nan"), 1.0, 1.0],
            [2.0, float("inf"), 3.0],
            [1000.0, 0.5, 0.5],
        ]
    )
    grid = voxel_coords(coord, 0.5, max_index=100)
    assert grid.dtype == torch.int32
    assert torch.equal(grid[0], torch.tensor([0, 0, 0], dtype=torch.int32))
    assert torch.equal(grid[1], torch.tensor([0, 1, 1], dtype=torch.int32))
    assert torch.equal(grid[2], torch.tensor([3, 0, 5], dtype=torch.int32))
    assert torch.equal(grid[3], torch.tensor([100, 0, 0], dtype=torch.int32))
    assert torch.isnan(coord[1, 0])  # input untouched


def test_voxel_coords_default_origin_and_explicit_origin():
    g = torch.Generator().manual_seed(0)
    coord = torch.rand(64, 3, generator=g) * 2 - 1
    grid = voxel_coords(coord, 0.1)
    expected = torch.div(coord - coord.min(0).values, 0.1, rounding_mode="trunc").int()
    assert torch.equal(grid, expected)
    assert bool((grid >= 0).all())
    with_origin = voxel_coords(coord, 0.1, origin=torch.tensor([-1.0, -1.0, -1.0]))
    expected = torch.div(coord + 1.0, 0.1, rounding_mode="trunc").int()
    assert torch.equal(with_origin, expected)
    assert voxel_coords(torch.zeros(0, 3), 0.1).shape == (0, 3)


@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min", "any"])
def test_cluster_reduce_matches_loop(reduce):
    g = torch.Generator().manual_seed(1)
    values = torch.randn(40, 3, generator=g)
    cluster = torch.randint(0, 6, (40,), generator=g)
    cluster[cluster == 4] = 0  # cluster 4 empty
    if reduce == "any":
        values = values > 0.5
    out = cluster_reduce(values, cluster, 6, reduce)
    for c in range(6):
        members = values[cluster == c]
        if members.shape[0] == 0:
            expected = torch.zeros(3, dtype=out.dtype)
        elif reduce == "sum":
            expected = members.sum(0)
        elif reduce == "mean":
            expected = members.mean(0)
        elif reduce == "max":
            expected = members.max(0).values
        elif reduce == "min":
            expected = members.min(0).values
        else:
            expected = members.any(0)
        assert torch.allclose(out[c].float(), expected.float(), atol=1e-6), (reduce, c)


def test_cluster_reduce_backward():
    values = torch.randn(10, 2, requires_grad=True)
    cluster = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2, 0])
    cluster_reduce(values, cluster, 3, "max").sum().backward()
    assert values.grad is not None
    assert torch.equal(values.grad.sum(0), torch.full((2,), 3.0))


def test_grid_clusters_matches_unique_reference():
    g = torch.Generator().manual_seed(2)
    grid = torch.randint(0, 8, (100, 3), generator=g, dtype=torch.int32)
    batch = torch.sort(torch.randint(0, 3, (100,), generator=g)).values
    pooled, clusters = grid_clusters(grid, batch, stride=2)

    ref_key = torch.cat([batch[:, None], torch.div(grid, 2, rounding_mode="trunc").long()], dim=1)
    ref_unique, ref_inverse, ref_counts = torch.unique(
        ref_key, dim=0, return_inverse=True, return_counts=True
    )
    assert torch.equal(clusters.cluster, ref_inverse)
    assert torch.equal(clusters.counts, ref_counts)
    assert torch.equal(pooled.long(), ref_unique[:, 1:])
    assert pooled.dtype == grid.dtype
    # head is the lowest row of each cluster; ptr/sorted_index partition the rows.
    for c in range(clusters.num_clusters):
        rows = (clusters.cluster == c).nonzero().flatten()
        assert int(clusters.head[c]) == int(rows[0])
        segment = clusters.sorted_index[clusters.ptr[c] : clusters.ptr[c + 1]]
        assert torch.equal(segment, rows)


def test_code_clusters():
    code = torch.tensor([5, 3, 5, 9, 3, 3])
    clusters = code_clusters(code)
    assert torch.equal(clusters.cluster, torch.tensor([1, 0, 1, 2, 0, 0]))
    assert torch.equal(clusters.counts, torch.tensor([3, 2, 1]))
    assert torch.equal(clusters.head, torch.tensor([1, 0, 3]))
