"""Configuration dataclasses and size presets for PointTransformerV3."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

from ontic_lib.pointops import SpaceFillingOrder

AttentionBackend = Literal["sdpa", "flash"]
RpeVersion = Literal["none", "v1", "v2"]
RopeCoordSource = Literal["coord", "grid_coord"]
RopeImpl = Literal["torch", "cuda"]
PoolingKind = Literal["serialized", "grid"]
PoolNorm = Literal["ln", "bn"]
ConvImpl = Literal["torch", "spconv"]


@dataclass
class StageCfg:
    """One encoder/decoder stage: ``depth`` blocks of ``channels`` with ``heads``."""

    depth: int
    channels: int
    heads: int
    patch_size: int = 1024
    conv: bool = True
    attn: bool = True


@dataclass
class AttentionCfg:
    """Serialized attention options shared by every block."""

    backend: AttentionBackend = "sdpa"
    qkv_bias: bool = True
    qk_scale: float | None = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    upcast_attention: bool = False
    upcast_softmax: bool = False
    rpe: RpeVersion = "none"
    rope: bool = False
    rope_base: float = 100.0
    rope_f0: float = 1.0
    rope_coord_source: RopeCoordSource = "coord"
    rope_stage_rescale: bool = True
    rope_impl: RopeImpl = "torch"


@dataclass
class TemporalCfg:
    """Time merging after each pooling: ``merge_window[s]`` steps fold into one group."""

    merge_window: tuple[int, ...] = (1, 2, 2, 2)
    time_embedding: bool = True


@dataclass
class PoolingCfg:
    """Down-sampling between stages."""

    kind: PoolingKind = "serialized"
    strides: tuple[int, ...] = (2, 2, 2, 2)
    reduce: Literal["sum", "mean", "max", "min"] = "max"
    norm: PoolNorm = "ln"
    neighbor_attender: bool = False


@dataclass
class PointTransformerV3Cfg:
    in_channels: int = 6
    out_channels: int = 3
    grid_size: float = 0.02
    orders: tuple[SpaceFillingOrder, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")
    shuffle_orders: bool = True
    encoder: tuple[StageCfg, ...] = ()
    decoder: tuple[StageCfg, ...] = ()
    attention: AttentionCfg = field(default_factory=AttentionCfg)
    pooling: PoolingCfg = field(default_factory=PoolingCfg)
    temporal: TemporalCfg | None = field(default_factory=TemporalCfg)
    conv_impl: ConvImpl = "torch"
    mlp_ratio: int = 4
    drop_path: float = 0.3
    pre_norm: bool = True
    cls_mode: bool = False
    feature_pyramid: bool = False
    gradient_checkpointing: bool = False

    @property
    def num_stages(self) -> int:
        return len(self.encoder)

    def validate(self) -> None:
        """Raise ``ValueError`` on inconsistent stage/pooling/attention settings."""
        n = self.num_stages
        if n < 1:
            raise ValueError("at least one encoder stage is required")
        if len(self.pooling.strides) != n - 1:
            raise ValueError(
                f"pooling.strides must have {n - 1} entries, got {self.pooling.strides}"
            )
        for stride in self.pooling.strides:
            if stride < 1 or stride & (stride - 1):
                raise ValueError(f"pooling strides must be powers of two, got {stride}")
        if self.temporal is not None:
            if len(self.temporal.merge_window) != n - 1:
                raise ValueError(
                    f"temporal.merge_window must have {n - 1} entries, "
                    f"got {self.temporal.merge_window}"
                )
            for window in self.temporal.merge_window:
                if window < 1 or window & (window - 1):
                    raise ValueError(f"merge windows must be powers of two, got {window}")
        if not self.cls_mode and len(self.decoder) != n - 1:
            raise ValueError(f"decoder must have {n - 1} stages, got {len(self.decoder)}")
        for stage in (*self.encoder, *self.decoder):
            if stage.channels % stage.heads:
                raise ValueError(f"channels {stage.channels} not divisible by heads {stage.heads}")
            if self.attention.rope and (stage.channels // stage.heads) % 6:
                raise ValueError("RoPE requires channels // heads to be a multiple of 6")
        attn = self.attention
        if attn.backend == "flash":
            if attn.rpe != "none":
                raise ValueError("flash attention does not support RPE")
            if attn.upcast_attention or attn.upcast_softmax:
                raise ValueError("flash attention does not support upcast_attention/upcast_softmax")
        if len(self.orders) == 0:
            raise ValueError("at least one serialization order is required")


def _stages(depths, channels, heads, patch_size=1024) -> tuple[StageCfg, ...]:
    return tuple(
        StageCfg(depth=d, channels=c, heads=h, patch_size=patch_size)
        for d, c, h in zip(depths, channels, heads)
    )


def base(**overrides: Any) -> PointTransformerV3Cfg:
    """PTv3 base: enc (2,2,2,6,2) x (32..512), dec (2,2,2,2) x (64,64,128,256)."""
    cfg = PointTransformerV3Cfg(
        encoder=_stages((2, 2, 2, 6, 2), (32, 64, 128, 256, 512), (2, 4, 8, 16, 32)),
        decoder=_stages((2, 2, 2, 2), (64, 64, 128, 256), (4, 4, 8, 16)),
    )
    return dataclasses.replace(cfg, **overrides)


def medium(**overrides: Any) -> PointTransformerV3Cfg:
    """PTv3 medium: enc (3,3,3,6,3) x (48..512), dec (3,3,3,3) x (64,96,192,384)."""
    cfg = PointTransformerV3Cfg(
        encoder=_stages((3, 3, 3, 6, 3), (48, 96, 192, 384, 512), (3, 6, 12, 24, 32)),
        decoder=_stages((3, 3, 3, 3), (64, 96, 192, 384), (4, 6, 12, 24)),
    )
    return dataclasses.replace(cfg, **overrides)


def large(**overrides: Any) -> PointTransformerV3Cfg:
    """PTv3 large: enc (3,3,3,12,3) x (48..512), dec (2,2,2,2) x (64,96,192,384)."""
    cfg = PointTransformerV3Cfg(
        encoder=_stages((3, 3, 3, 12, 3), (48, 96, 192, 384, 512), (3, 6, 12, 24, 32)),
        decoder=_stages((2, 2, 2, 2), (64, 96, 192, 384), (4, 6, 12, 24)),
    )
    return dataclasses.replace(cfg, **overrides)


def fwomo_legacy(**overrides: Any) -> PointTransformerV3Cfg:
    """:func:`base` with the accelerators fwomo ran on: spconv CPE and flash attention."""
    cfg = base(conv_impl="spconv", attention=AttentionCfg(backend="flash"))
    return dataclasses.replace(cfg, **overrides)
