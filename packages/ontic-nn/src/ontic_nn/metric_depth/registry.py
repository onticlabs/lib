"""Name → config-class registry of the metric depth models (populated by ``ontic_nn.metric_depth``)."""

from __future__ import annotations

from typing import Dict, Type

from .common import MetricDepthConfig

METRIC_MODELS: Dict[str, Type[MetricDepthConfig]] = {}


def register_metric_model(name: str):
    """Class decorator adding a config class to :data:`METRIC_MODELS` under ``name``."""

    def deco(cls: Type[MetricDepthConfig]) -> Type[MetricDepthConfig]:
        METRIC_MODELS[name] = cls
        return cls

    return deco
