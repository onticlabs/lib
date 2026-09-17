"""Unit tests for ontic_lib.pointops.serialization."""

import pytest
import torch

from ontic_lib.pointops.serialization import (
    encode_grid,
    hilbert_decode,
    hilbert_encode,
    morton_decode,
    morton_encode,
)


def _random_grid(n, depth, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 2**depth, (n, 3), generator=g)


@pytest.mark.parametrize("depth", [2, 4, 8, 12, 16])
def test_morton_roundtrip_identity(depth):
    grid = _random_grid(256, depth, seed=depth)
    code = morton_encode(grid, depth=depth)
    back = morton_decode(code, depth=depth)
    assert torch.equal(back, grid)


@pytest.mark.parametrize("depth", [2, 4, 8, 12, 16])
def test_hilbert_roundtrip_identity(depth):
    grid = _random_grid(256, depth, seed=100 + depth)
    code = hilbert_encode(grid, depth=depth)
    back = hilbert_decode(code, depth=depth)
    assert torch.equal(back, grid)


def test_morton_encode_rejects_bad_shape():
    with pytest.raises(ValueError):
        morton_encode(torch.zeros(4, 2, dtype=torch.long))


def test_morton_encode_rejects_bad_depth():
    with pytest.raises(ValueError):
        morton_encode(torch.zeros(4, 3, dtype=torch.long), depth=17)


def test_hilbert_encode_rejects_bad_depth():
    with pytest.raises(ValueError):
        hilbert_encode(torch.zeros(4, 3, dtype=torch.long), depth=22)


@pytest.mark.parametrize("order", ["z", "hilbert"])
def test_encode_grid_matches_direct_encoder(order):
    grid = _random_grid(128, 8, seed=7)
    code = encode_grid(grid, depth=8, order=order)
    direct = (morton_encode if order == "z" else hilbert_encode)(grid, depth=8)
    assert torch.equal(code, direct)


def test_encode_grid_rejects_unknown_order():
    with pytest.raises(ValueError):
        encode_grid(torch.zeros(4, 3, dtype=torch.long), order="spiral")


def test_encode_grid_batch_offsetting():
    # Batch index is packed into the high bits (batch << (depth * 3 + 10)), so a
    # nonzero batch strictly increases the code, and the low bits still match the
    # unbatched code.
    depth = 8
    grid = _random_grid(64, depth, seed=3)
    base = encode_grid(grid, depth=depth, order="z")
    batch = torch.ones(grid.shape[0], dtype=torch.long)
    batched = encode_grid(grid, batch=batch, depth=depth, order="z")
    shift = depth * 3 + 10
    assert torch.equal(batched, (batch << shift) | base)
    assert torch.all(batched > base)
    # Different batch rows never collide with batch-0 rows.
    assert torch.all(batched >= (1 << shift))


def test_encode_grid_trans_swaps_xy():
    grid = _random_grid(64, 8, seed=11)
    swapped = grid[:, [1, 0, 2]]
    assert torch.equal(
        encode_grid(grid, depth=8, order="z-trans"),
        encode_grid(swapped, depth=8, order="z"),
    )


@pytest.mark.parametrize("encode", [morton_encode, hilbert_encode])
def test_ordering_locality_sanity(encode):
    # A space-filling curve keeps consecutive codes spatially close: walking the
    # grid in code order should give small average step sizes relative to a random
    # traversal of the same points.
    depth = 5
    coords = torch.arange(2**depth)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    grid = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    code = encode(grid, depth=depth)
    order = torch.argsort(code)
    ordered = grid[order].float()
    curve_step = (ordered[1:] - ordered[:-1]).norm(dim=-1).mean()

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(grid.shape[0], generator=g)
    shuffled = grid[perm].float()
    random_step = (shuffled[1:] - shuffled[:-1]).norm(dim=-1).mean()

    assert curve_step < random_step


# --- Serialization container ---------------------------------------------------------


def _grid_and_batch(n=200, seed=5):
    g = torch.Generator().manual_seed(seed)
    grid = torch.randint(0, 40, (n, 3), generator=g, dtype=torch.int32)
    batch = torch.sort(torch.randint(0, 3, (n,), generator=g)).values
    return grid, batch


def test_serialize_depth_rule_and_layout():
    from ontic_lib.pointops.serialization import serialize

    grid, batch = _grid_and_batch()
    s = serialize(grid, batch, orders=["z", "hilbert-trans"])
    assert s.depth == int(grid.max()).bit_length() + 1
    assert s.code.shape == (2, grid.shape[0])
    for k, order in enumerate(["z", "hilbert-trans"]):
        assert torch.equal(s.code[k], encode_grid(grid, batch=batch, depth=s.depth, order=order))
        assert torch.equal(s.code[k][s.order[k]], s.code[k].sort().values)
        assert torch.equal(s.order[k][s.inverse[k]], torch.arange(grid.shape[0]))
    with pytest.raises(ValueError):
        serialize(grid, batch, orders=["z"], depth=17)
    with pytest.raises(ValueError):
        serialize(grid, torch.full_like(batch, 2**15), orders=["z"], depth=16)  # 48 + 16 > 63
    with pytest.raises(ValueError):
        serialize(grid, batch, orders=[])


def test_serialize_shuffle_is_reproducible():
    from ontic_lib.pointops.serialization import serialize

    grid, batch = _grid_and_batch()
    orders = ["z", "z-trans", "hilbert", "hilbert-trans"]
    plain = serialize(grid, batch, orders=orders)
    a = serialize(
        grid, batch, orders=orders, shuffle=True, generator=torch.Generator().manual_seed(3)
    )
    b = serialize(
        grid, batch, orders=orders, shuffle=True, generator=torch.Generator().manual_seed(3)
    )
    assert torch.equal(a.code, b.code) and torch.equal(a.order, b.order)
    # Shuffling permutes rows of the unshuffled result.
    perm = torch.randperm(4, generator=torch.Generator().manual_seed(3))
    assert torch.equal(a.code, plain.code[perm])
    assert torch.equal(a.inverse, plain.inverse[perm])


def test_reserialize_keeps_spatial_bits():
    from ontic_lib.pointops.serialization import reserialize, serialize

    grid, batch = _grid_and_batch()
    s = serialize(grid, batch, orders=["z", "hilbert"])
    new_batch = batch // 2
    r = reserialize(s, new_batch)
    shift = s.depth * 3 + 10
    assert r.depth == s.depth
    assert torch.equal(r.code & ((1 << shift) - 1), s.code & ((1 << shift) - 1))
    assert torch.equal(r.code >> shift, new_batch.expand(2, -1))
    direct = serialize(grid, new_batch, orders=["z", "hilbert"], depth=s.depth)
    assert torch.equal(r.code, direct.code)
    assert torch.equal(r.order, direct.order)


def test_pool_serialization_matches_frontier_arithmetic():
    from ontic_lib.pointops.serialization import pool_serialization, serialize

    grid, batch = _grid_and_batch()
    s = serialize(grid, batch, orders=["z", "hilbert"])
    pooling_depth = 1
    # Reference: frontier SerializedPooling.serialized_forward, first six lines.
    code = s.code >> pooling_depth * 3
    _, cluster, counts = torch.unique(code[0], sorted=True, return_inverse=True, return_counts=True)
    _, indices = torch.sort(cluster, stable=True)
    idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
    head = indices[idx_ptr[:-1]]
    ref_code = code[:, head]
    ref_order = torch.argsort(ref_code, stable=True)
    ref_inverse = torch.zeros_like(ref_order).scatter_(
        1, ref_order, torch.arange(ref_code.shape[1]).repeat(2, 1)
    )

    p = pool_serialization(s, head, pooling_depth)
    assert p.depth == s.depth - pooling_depth
    assert torch.equal(p.code, ref_code)
    assert torch.equal(p.order, ref_order)
    assert torch.equal(p.inverse, ref_inverse)
