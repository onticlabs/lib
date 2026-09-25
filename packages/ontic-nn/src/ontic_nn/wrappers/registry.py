"""Name → config-class registry of the backbone wrappers (populated by ``ontic_nn.wrappers``)."""

from __future__ import annotations

from typing import Dict, Type

from .common import BackboneConfig

BACKBONES: Dict[str, Type[BackboneConfig]] = {}


def register_backbone(name: str):
    """Class decorator adding a config class to :data:`BACKBONES` under ``name``."""

    def deco(cls: Type[BackboneConfig]) -> Type[BackboneConfig]:
        BACKBONES[name] = cls
        return cls

    return deco
