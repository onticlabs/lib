"""Unit tests for ontic_lib.pointops.packing."""

import pytest
import torch

from ontic_lib.pointops.packing import (
    batch_to_offset,
    offset_to_batch,
    offset_to_counts,
    pack_padded,
    scatter_to_padded,
    unpack_to_padded,
)


def _mask(shape, seed, empty_group=None):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(shape, generator=g) > 0.4
    if empty_group is not None:
        mask.reshape(-1, shape[-1])[empty_group] = False
    return mask


def test_offset_batch_conversions():
    offset = torch.tensor([3, 3, 7])  # second group empty
    assert torch.equal(offset_to_counts(offset), torch.tensor([3, 0, 4]))
    batch = offset_to_batch(offset)
    assert torch.equal(batch, torch.tensor([0, 0, 0, 2, 2, 2, 2]))
    assert torch.equal(batch_to_offset(batch), offset)
    # Trailing empty groups only exist when num_groups is given.
    assert torch.equal(batch_to_offset(torch.tensor([0, 0]), num_groups=3), torch.tensor([2, 2, 2]))
    with pytest.raises(ValueError):
        batch_to_offset(torch.tensor([0, 4]), num_groups=2)


@pytest.mark.parametrize("shape", [(3, 5), (2, 3, 4)])
def test_pack_unpack_round_trip(shape):
    mask = _mask(shape, seed=len(shape), empty_group=1)
    g = torch.Generator().manual_seed(0)
    coord = torch.randn(*shape, 3, generator=g)
    flag = torch.rand(shape, generator=g) > 0.5
    batch, (coord_p, flag_p) = pack_padded(mask, coord, flag)

    assert batch.shape == (int(mask.sum()),)
    assert torch.equal(batch, batch.sort().values)
    assert torch.equal(coord_p, coord[mask])
    assert torch.equal(flag_p, flag[mask])
    assert not bool((batch == 1).any())  # the emptied group contributes nothing

    num_groups = mask.reshape(-1, shape[-1]).shape[0]
    padded, out_mask = unpack_to_padded(coord_p, batch, num_groups=num_groups)
    assert padded.shape == (num_groups, int(mask.reshape(-1, shape[-1]).sum(-1).max()), 3)
    # Left-compacted: the round trip matches after compacting the input mask.
    for gi in range(num_groups):
        rows = coord.reshape(num_groups, shape[-1], 3)[gi][mask.reshape(num_groups, -1)[gi]]
        assert torch.equal(padded[gi, : rows.shape[0]], rows)
        assert int(out_mask[gi].sum()) == rows.shape[0]

    assert torch.equal(scatter_to_padded(coord_p, mask), coord.masked_fill(~mask[..., None], 0.0))


def test_unpack_respects_max_count_and_unsorted_batch():
    packed = torch.arange(6.0).unsqueeze(-1)
    batch = torch.tensor([1, 0, 1, 0, 1, 0])
    padded, mask = unpack_to_padded(packed, batch, max_count=2, pad_value=-1.0)
    assert padded.shape == (2, 2, 1)
    assert torch.equal(padded[0, :, 0], torch.tensor([1.0, 3.0]))
    assert torch.equal(padded[1, :, 0], torch.tensor([0.0, 2.0]))
    assert bool(mask.all())


def test_pack_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pack_padded(torch.ones(2, 3), torch.zeros(2, 3, 1))
    with pytest.raises(ValueError):
        pack_padded(torch.ones(2, 3, dtype=torch.bool), torch.zeros(2, 4, 1))
    with pytest.raises(ValueError):
        scatter_to_padded(torch.zeros(2, 1), torch.ones(3, dtype=torch.bool))
