"""Metric monocular depth models with a common ``MetricDepthOutput`` contract.

Importing this package registers every model config in :data:`METRIC_MODELS` (keys ``da3``,
``depthpro``, ``metric3d``, ``unidepth``). The research packages are imported lazily by
``build()``.
"""

from .common import MetricDepthConfig, MetricDepthModel, MetricDepthOutput, offline_guard
from .da3 import DA3MetricConfig, DA3MetricModel
from .depthpro import DepthProMetricConfig, DepthProMetricModel
from .metric3d import Metric3DMetricConfig, Metric3DMetricModel
from .registry import METRIC_MODELS, register_metric_model
from .unidepth import UniDepthMetricConfig, UniDepthMetricModel

__all__ = [
    "DA3MetricConfig",
    "DA3MetricModel",
    "DepthProMetricConfig",
    "DepthProMetricModel",
    "METRIC_MODELS",
    "Metric3DMetricConfig",
    "Metric3DMetricModel",
    "MetricDepthConfig",
    "MetricDepthModel",
    "MetricDepthOutput",
    "UniDepthMetricConfig",
    "UniDepthMetricModel",
    "offline_guard",
    "register_metric_model",
]
