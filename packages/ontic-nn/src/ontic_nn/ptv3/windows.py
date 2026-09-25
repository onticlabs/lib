"""Fixed-size attention windows over serialized, packed point groups.

Every group of ``count`` points (in serialized order) is cut into windows of
``patch_size``. A group larger than ``patch_size`` is padded up to a multiple
by *borrowing* the tail of its second-to-last window (no dummy tokens); a
group with ``count <= patch_size`` forms one short window.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ontic_lib.pointops.packing import offset_to_counts


@dataclass(frozen=True)
class WindowPadding:
    """Index tensors for one ``(offset, patch_size)`` pair.

    ``pad (N',)``: original row of every slot of the padded stream (sorted
    order applied afterwards). ``unpad (N,)``: slot of every original row.
    ``cu_seqlens (W+1,) int32``: window boundaries in the stream.
    ``window_index (W, P)``: stream slot per window position (short windows
    repeat their own slots), ``key_mask (W, P)`` bool marks real positions and
    ``slot_index (N',)`` maps stream slots to ``window * P + position``.
    """

    patch_size: int
    pad: Tensor
    unpad: Tensor
    cu_seqlens: Tensor
    window_index: Tensor
    key_mask: Tensor
    slot_index: Tensor

    @property
    def num_windows(self) -> int:
        return int(self.cu_seqlens.shape[0]) - 1


@torch.no_grad()
def window_padding(offset: Tensor, patch_size: int) -> WindowPadding:
    """Windowing of groups with cumulative counts ``offset (B,)`` (no leading zero)."""
    if patch_size < 1:
        raise ValueError(f"patch_size must be >= 1, got {patch_size}")
    device = offset.device
    counts = offset_to_counts(offset)
    counts_pad = torch.div(counts + patch_size - 1, patch_size, rounding_mode="trunc") * patch_size
    counts_pad = torch.where(counts > patch_size, counts_pad, counts)

    offset_ = torch.nn.functional.pad(offset, (1, 0))
    offset_pad = torch.nn.functional.pad(torch.cumsum(counts_pad, dim=0), (1, 0))
    pad = torch.arange(int(offset_pad[-1]), device=device)
    unpad = torch.arange(int(offset_[-1]), device=device)
    starts = []
    for i in range(offset.shape[0]):
        pads_before = offset_pad[i] - offset_[i]
        if counts[i] != counts_pad[i]:
            added = counts_pad[i] - counts[i]
            end_added = offset_pad[i + 1]
            start_added = end_added - added
            end_borrowed = end_added - patch_size
            start_borrowed = end_borrowed - added
            pad[start_added:end_added] = pad[start_borrowed:end_borrowed]
        unpad[offset_[i] : offset_[i + 1]] += pads_before
        pad[offset_pad[i] : offset_pad[i + 1]] -= pads_before
        starts.append(
            torch.arange(
                int(offset_pad[i]),
                int(offset_pad[i + 1]),
                step=patch_size,
                dtype=torch.int32,
                device=device,
            )
        )
    cu_seqlens = torch.nn.functional.pad(torch.cat(starts), (0, 1), value=int(offset_pad[-1]))

    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    window_start = cu_seqlens[:-1].long()
    position = torch.arange(patch_size, device=device)
    key_mask = position.unsqueeze(0) < lengths.unsqueeze(1)
    wrapped = position.unsqueeze(0) % lengths.clamp_min(1).unsqueeze(1)
    window_index = window_start.unsqueeze(1) + wrapped
    window_of_slot = torch.repeat_interleave(torch.arange(lengths.shape[0], device=device), lengths)
    slot = torch.arange(pad.shape[0], device=device)
    slot_index = window_of_slot * patch_size + slot - window_start[window_of_slot]
    return WindowPadding(
        patch_size=patch_size,
        pad=pad,
        unpad=unpad,
        cu_seqlens=cu_seqlens,
        window_index=window_index,
        key_mask=key_mask,
        slot_index=slot_index,
    )
