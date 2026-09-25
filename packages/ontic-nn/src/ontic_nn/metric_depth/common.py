"""Base types for metric monocular depth models.

A metric model only has to produce **metric** z-depth (meters) per view — no features, no
poses. Intrinsics crossing this boundary, in and out, are normalised ``[0, 1]``
``(fx/W, fy/H, cx/W, cy/H)``, so they survive each model's internal resize unchanged.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch.nn as nn
from torch import Tensor

from ontic_nn.wrappers.common import offline_guard, resize_to_long_side

__all__ = [
    "MetricDepthConfig",
    "MetricDepthModel",
    "MetricDepthOutput",
    "offline_guard",
    "resize_to_long_side",
]


@dataclass
class MetricDepthOutput:
    """``depth (B, V, Hd, Wd)`` metric z-depth at the model's ``long_side``-resized grid;
    optional ``conf (B, V, Hd, Wd)`` (model-specific scale) and ``intrinsics (B, V, 3, 3)``
    normalised, predicted by self-calibrating models (``None`` otherwise)."""

    depth: Tensor
    conf: Optional[Tensor] = None
    intrinsics: Optional[Tensor] = None


@dataclass(kw_only=True)
class MetricDepthConfig:
    """Base config; the model owns its input resolution via ``long_side``.

    ``checkpoint_path`` (a pre-downloaded file / snapshot directory) is used first; else the
    model's default hub source is fetched into ``cache_dir`` (``None`` → the HF / torch hub
    cache); ``allow_download=False`` serves from the cache only.
    """

    long_side: int = 504
    checkpoint_path: Optional[str] = None
    cache_dir: Optional[str] = None
    allow_download: bool = True

    def build(self) -> MetricDepthModel:
        raise NotImplementedError


class MetricDepthModel(nn.Module):
    """Frozen, eval-only ``forward(images (B, V, 3, H, W) in [0, 1], intrinsics=None)``.

    Models that self-estimate the camera ignore the input ``intrinsics`` and return their own
    estimate; ``REQUIRES_INTRINSICS`` models (Metric3D) need them and raise without.
    """

    PATCH_SIZE: int = 14
    REQUIRES_INTRINSICS: bool = False

    if TYPE_CHECKING:

        def __call__(self, *args, **kwargs) -> MetricDepthOutput: ...

    @abstractmethod
    def forward(self, images: Tensor, intrinsics: Optional[Tensor] = None) -> MetricDepthOutput: ...

    @property
    def requires_intrinsics(self) -> bool:
        return self.REQUIRES_INTRINSICS
