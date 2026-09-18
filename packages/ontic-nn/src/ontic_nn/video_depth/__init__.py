"""Optional temporal depth models; importing this registry never loads weights."""

from .vda import VDABaseConfig, VDALargeConfig, VideoDepthAnythingConfig, VideoDepthAnythingModel
from .velodepth import VeloDepthConfig, VeloDepthModel
from .da3 import DA3VideoConfig, DA3VideoModel

VIDEO_DEPTH_MODELS = {
    "vda_small": VideoDepthAnythingConfig,
    "vda_base": VDABaseConfig,
    "vda_large": VDALargeConfig,
    "velodepth": VeloDepthConfig,
    "da3_nested": DA3VideoConfig,
}

__all__ = [
    "VIDEO_DEPTH_MODELS",
    "DA3VideoConfig",
    "DA3VideoModel",
    "VeloDepthConfig",
    "VeloDepthModel",
    "VDABaseConfig",
    "VDALargeConfig",
    "VideoDepthAnythingConfig",
    "VideoDepthAnythingModel",
]
