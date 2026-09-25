"""Unit tests for ontic_lib.structures.PointBatch."""

import pytest
import torch

from ontic_lib.structures import PointBatch


def _padded(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    coord = torch.randn(*shape, 3, generator=g)
    feat = torch.randn(*shape, 4, generator=g)
    mask = torch.rand(shape, generator=g) > 0.3
    flag = torch.rand(shape, generator=g) > 0.5
    return coord, feat, mask, flag


def test_validation_errors():
    coord = torch.zeros(4, 3)
    feat = torch.zeros(4, 2)
    batch = torch.tensor([0, 0, 1, 1])
    PointBatch(coord, feat, batch)
    with pytest.raises(ValueError):
        PointBatch(coord[:, :2], feat, batch)
    with pytest.raises(ValueError):
        PointBatch(coord, feat[:3], batch)
    with pytest.raises(ValueError):
        PointBatch(coord, feat, batch.int())
    with pytest.raises(ValueError):
        PointBatch(coord, feat, torch.tensor([1, 0, 1, 1]))
    with pytest.raises(ValueError):
        PointBatch(coord, feat, batch, time=torch.zeros(3, dtype=torch.int64))
    with pytest.raises(ValueError):
        PointBatch(coord, feat, batch, extras={"x": torch.zeros(5)})


def test_from_padded_to_padded_round_trip_3d():
    coord, feat, mask, flag = _padded((3, 6))
    mask[1] = False  # empty group
    pb = PointBatch.from_padded(coord, feat, mask, extras={"flag": flag})
    assert len(pb) == int(mask.sum())
    assert pb.time is None
    assert pb.num_groups == 3
    assert torch.equal(pb.counts, mask.sum(-1))
    assert torch.equal(pb.offset, mask.sum(-1).cumsum(0))
    fields, out_mask = pb.to_padded(pad_value=-1.0)
    assert fields["coord"].shape[0] == 3
    for b in range(3):
        rows = coord[b][mask[b]]
        assert torch.equal(fields["coord"][b, : rows.shape[0]], rows)
        assert torch.equal(fields["flag"][b, : rows.shape[0]], flag[b][mask[b]])
        assert int(out_mask[b].sum()) == rows.shape[0]
        assert bool((fields["coord"][b, rows.shape[0] :] == -1.0).all())


def test_from_padded_to_padded_round_trip_4d():
    coord, feat, mask, _ = _padded((2, 3, 5), seed=1)
    pb = PointBatch.from_padded(coord, feat, mask)
    assert pb.time is not None
    # Ordered by b, then t, then k.
    b_idx, t_idx, _ = mask.nonzero(as_tuple=True)
    assert torch.equal(pb.batch, b_idx)
    assert torch.equal(pb.time, t_idx)
    fields, out_mask = pb.to_padded()
    assert fields["coord"].shape[:2] == (2, 3)
    assert fields["feat"].shape[-1] == 4
    assert out_mask.shape[:2] == (2, 3)
    for b in range(2):
        for t in range(3):
            rows = coord[b, t][mask[b, t]]
            assert torch.equal(fields["coord"][b, t, : rows.shape[0]], rows)
            assert int(out_mask[b, t].sum()) == rows.shape[0]
    # Explicit time overrides the arange.
    time = torch.full((2, 3, 5), 2, dtype=torch.int64)
    pb2 = PointBatch.from_padded(coord, feat, mask, time=time)
    assert bool((pb2.time == 2).all())


def test_select_and_cat():
    coord, feat, mask, _ = _padded((2, 4), seed=2)
    pb = PointBatch.from_padded(coord, feat, mask)
    keep = pb.batch == 1
    sub = pb.select(keep)
    assert len(sub) == int(keep.sum())
    assert torch.equal(sub.coord, pb.coord[keep])
    both = pb.cat(sub, batch_offset=2)
    assert both.num_groups == 4
    assert torch.equal(both.batch, both.batch.sort().values)
    assert torch.equal(both.coord[both.batch == 3], sub.coord)
    same = pb.cat(sub)  # appended rows re-sorted into group 1, after the existing ones
    assert torch.equal(same.coord[same.batch == 1], torch.cat([pb.coord[keep], sub.coord]))


def test_replace_to_detach_clone():
    coord, feat, mask, flag = _padded((2, 4), seed=3)
    pb = PointBatch.from_padded(coord, feat, mask, extras={"flag": flag})
    new = pb.replace(feat=pb.feat * 2, score=torch.ones(len(pb)))
    assert torch.equal(new.feat, pb.feat * 2)
    assert "score" in new.extras and "flag" in new.extras
    half = pb.to(torch.float16)
    assert half.coord.dtype == torch.float16 and half.batch.dtype == torch.int64
    assert pb.clone().detach().coord.shape == pb.coord.shape
