# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Strided-convolution patch embedding."""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

from torch import Tensor, nn


def make_2tuple(x: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    if isinstance(x, tuple):
        if len(x) != 2:
            raise ValueError(f"expected a 2-tuple, got {x!r}")
        return x
    return (x, x)


class PatchEmbed(nn.Module):
    """Image to patch tokens: ``(B, C, H, W) -> (B, N, D)`` (or ``(B, H', W', D)``).

    ``H`` and ``W`` must be multiples of ``patch_size``; ``img_size`` only fixes
    ``num_patches`` for the positional-embedding table.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_hw = make_2tuple(img_size)
        patch_hw = make_2tuple(patch_size)
        patch_grid_size = (image_hw[0] // patch_hw[0], image_hw[1] // patch_hw[1])

        self.img_size = image_hw
        self.patch_size = patch_hw
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_hw, stride=patch_hw)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape
        patch_h, patch_w = self.patch_size
        if h % patch_h != 0 or w % patch_w != 0:
            raise ValueError(
                f"image size ({h}, {w}) is not a multiple of patch size ({patch_h}, {patch_w})"
            )

        x = self.proj(x)  # B C H' W'
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # B N C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, h, w, self.embed_dim)  # B H' W' C
        return x
