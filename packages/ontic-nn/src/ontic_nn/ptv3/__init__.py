"""Point Transformer V3 over :class:`ontic_lib.structures.PointBatch`.

Pure torch by default (``conv_impl="torch"``, ``attention.backend="sdpa"``);
spconv, flash-attn and the RoPE CUDA kernel are opt-in accelerators imported
lazily at construction time.
"""

from . import presets
from .compat import load_fwomo_state_dict
from .config import AttentionCfg, PointTransformerV3Cfg, PoolingCfg, StageCfg, TemporalCfg
from .model import PointTransformerV3
from .state import MergeRecord, PoolRecord, StageState

__all__ = [
    "AttentionCfg",
    "MergeRecord",
    "PointTransformerV3",
    "PointTransformerV3Cfg",
    "PoolRecord",
    "PoolingCfg",
    "StageCfg",
    "StageState",
    "TemporalCfg",
    "load_fwomo_state_dict",
    "presets",
]
