# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Per-channel learnable residual scaling (CaiT-style LayerScale)."""

from __future__ import annotations

from typing import Union

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    """Multiply the last dim by a learnable ``gamma (dim,)`` initialised to ``init_values``."""

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
