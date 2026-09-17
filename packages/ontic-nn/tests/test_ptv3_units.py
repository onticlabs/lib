"""Unit tests for the PTv3 building blocks: sparse conv, windowing, sdpa attention."""

import torch

from ontic_lib.pointops import batch_to_offset, serialize
from ontic_lib.structures import PointBatch
from ontic_nn.ptv3.attention import SerializedAttention
from ontic_nn.ptv3.config import AttentionCfg
from ontic_nn.ptv3.sparse_conv import SubmanifoldConv3d, build_neighbor_table, kernel_offsets
from ontic_nn.ptv3.state import StageState
from ontic_nn.ptv3.windows import window_padding


def _dense_reference(feat, grid, batch, weight, bias, k):
    """Loop over rows and kernel offsets; neighbour = lowest row at that voxel."""
    n = feat.shape[0]
    offsets = kernel_offsets(k)
    out = torch.zeros(n, weight.shape[-1])
    for i in range(n):
        for o in range(offsets.shape[0]):
            target = grid[i] + offsets[o]
            if bool((offsets[o] == 0).all()):
                src = i
            else:
                hits = [
                    j for j in range(n) if batch[j] == batch[i] and bool((grid[j] == target).all())
                ]
                if not hits:
                    continue
                src = hits[0]
            out[i] += feat[src] @ weight[o]
    return out + bias if bias is not None else out


def test_submanifold_conv_torch_matches_dense_reference():
    torch.manual_seed(0)
    grid = torch.tensor(
        [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 1], [0, 0, 0], [1, 1, 1], [1, 1, 2]],
        dtype=torch.int32,
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1])  # row 3 duplicates row 0; row 5 is in group 1
    feat = torch.randn(7, 4)
    conv = SubmanifoldConv3d(4, 5, kernel_size=3, bias=True)
    table = build_neighbor_table(grid, batch, 3)
    centre = table.index.shape[0] // 2
    assert torch.equal(table.index[centre], torch.arange(7))
    assert int(table.index[centre + 9][3]) == 1  # (+1, 0, 0) neighbour of the duplicate row 0/3
    assert int(table.index[centre + 9][5]) == -1  # group 1 has no such neighbour
    out = conv(feat, table)
    ref = _dense_reference(feat, grid.long(), batch, conv.weight.detach(), conv.bias.detach(), 3)
    assert torch.allclose(out.detach(), ref, atol=1e-5)
    out.sum().backward()
    assert conv.weight.grad is not None


def _frontier_padding(offset, patch_size):
    """Verbatim port of fwomo SerializedMixin.get_padding_and_inverse."""
    bincount = torch.diff(offset, prepend=torch.tensor([0]))
    bincount_pad = (
        torch.div(bincount + patch_size - 1, patch_size, rounding_mode="trunc") * patch_size
    )
    bincount_pad = torch.where(bincount > patch_size, bincount_pad, bincount)
    _offset = torch.nn.functional.pad(offset, (1, 0))
    _offset_pad = torch.nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
    pad = torch.arange(_offset_pad[-1])
    unpad = torch.arange(_offset[-1])
    cu_seqlens = []
    for i in range(len(offset)):
        total_pads_added_before_batch_i = _offset_pad[i] - _offset[i]
        if bincount[i] != bincount_pad[i]:
            pad_added_at_batch_i = bincount_pad[i] - bincount[i]
            end_added_pad = _offset_pad[i + 1]
            start_added_pad = end_added_pad - pad_added_at_batch_i
            end_borrowed_pad = end_added_pad - patch_size
            start_borrowed_pad = end_borrowed_pad - pad_added_at_batch_i
            pad[start_added_pad:end_added_pad] = pad[start_borrowed_pad:end_borrowed_pad]
        unpad[_offset[i] : _offset[i + 1]] += total_pads_added_before_batch_i
        pad[_offset_pad[i] : _offset_pad[i + 1]] -= total_pads_added_before_batch_i
        cu_seqlens.append(
            torch.arange(_offset_pad[i], _offset_pad[i + 1], step=patch_size, dtype=torch.int32)
        )
    return (
        pad,
        unpad,
        torch.nn.functional.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]),
    )


def test_window_padding_matches_frontier_port():
    g = torch.Generator().manual_seed(0)
    for _ in range(5):
        counts = torch.randint(1, 40, (4,), generator=g)
        counts[1] = 3  # a short group
        offset = counts.cumsum(0)
        for patch_size in (4, 8):
            w = window_padding(offset, patch_size)
            pad, unpad, cu = _frontier_padding(offset, patch_size)
            assert torch.equal(w.pad, pad) and torch.equal(w.unpad, unpad)
            assert torch.equal(w.cu_seqlens, cu)
            # Every stream slot lands in exactly one window position and back.
            flat = torch.full((w.num_windows * patch_size,), -1, dtype=torch.long)
            flat[w.slot_index] = torch.arange(pad.shape[0])
            gathered = w.window_index[w.key_mask]
            assert torch.equal(flat[flat >= 0], gathered)


def test_sdpa_attention_matches_manual_softmax_with_short_group():
    torch.manual_seed(1)
    n_per = [12, 5, 9]  # middle group shorter than the patch size
    batch = torch.repeat_interleave(torch.arange(3), torch.tensor(n_per))
    n = int(batch.shape[0])
    coord = torch.rand(n, 3)
    feat = torch.randn(n, 8)
    points = PointBatch(coord, feat, batch)
    state = StageState.from_points(points, 0.1, orders=("z",))
    attn = SerializedAttention(8, 2, patch_size=4, order_index=0, cfg=AttentionCfg()).eval()
    out = attn(feat, state)

    # Manual: windows of serialized order within each group, borrowed tail, softmax per window.
    s = state.serialization
    w = state.window(4)
    order = s.order[0][w.pad]
    qkv = attn.qkv(feat)[order].view(-1, 3, 2, 4)
    expected_stream = torch.zeros(order.shape[0], 8)
    for i in range(w.num_windows):
        a, b = int(w.cu_seqlens[i]), int(w.cu_seqlens[i + 1])
        q, k, v = qkv[a:b].unbind(1)  # (L, H, D)
        scores = torch.einsum("ihd,jhd->hij", q, k) * attn.scale
        expected_stream[a:b] = torch.einsum("hij,jhd->ihd", scores.softmax(-1), v).reshape(b - a, 8)
    inverse = w.unpad[s.inverse[0]]
    expected = attn.proj(expected_stream[inverse])
    assert torch.allclose(out, expected, atol=1e-5)


def test_serialize_and_offsets_consistent_with_state():
    batch = torch.tensor([0, 0, 1, 1, 1])
    grid = torch.randint(0, 5, (5, 3), generator=torch.Generator().manual_seed(2))
    s = serialize(grid, batch, orders=("hilbert",))
    assert torch.equal(batch_to_offset(batch), torch.tensor([2, 5]))
    assert torch.equal(s.code[0] >> (s.depth * 3 + 10), batch)
