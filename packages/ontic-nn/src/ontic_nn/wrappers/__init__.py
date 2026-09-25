"""Pretrained multi-view backbone wrappers with a common ``BackboneOutput`` contract.

Importing this package registers every wrapper config in :data:`BACKBONES` (keys ``da3``,
``ma``, ``vggt``, ``pi3x``, ``dvlt``, ``moge3``, ``gtdepth``). The research packages behind
them are imported lazily by ``build()`` / ``forward``.
"""

from .common import (
    BackboneBase,
    BackboneConfig,
    BackboneOutput,
    amp_dtype,
    buffers_to_params,
    convert_to_buffer,
    extract_weights,
    import_research_module,
    offline_guard,
    resize_to_long_side,
    resolve_checkpoint,
    set_frozen,
)
from .da3 import DA3Backbone, DA3BackboneConfig, load_da3
from .dvlt import DVLTBackbone, DVLTBackboneConfig
from .gtdepth import GTDepthBackbone, GTDepthBackboneConfig
from .ma import MABackbone, MABackboneConfig
from .moge3 import MoGe3Backbone, MoGe3BackboneConfig
from .pi3x import Pi3XBackbone, Pi3XBackboneConfig
from .registry import BACKBONES, register_backbone
from .vggt import VGGTBackbone, VGGTBackboneConfig

__all__ = [
    "BACKBONES",
    "BackboneBase",
    "BackboneConfig",
    "BackboneOutput",
    "DA3Backbone",
    "DA3BackboneConfig",
    "DVLTBackbone",
    "DVLTBackboneConfig",
    "GTDepthBackbone",
    "GTDepthBackboneConfig",
    "MABackbone",
    "MABackboneConfig",
    "MoGe3Backbone",
    "MoGe3BackboneConfig",
    "Pi3XBackbone",
    "Pi3XBackboneConfig",
    "VGGTBackbone",
    "VGGTBackboneConfig",
    "amp_dtype",
    "buffers_to_params",
    "convert_to_buffer",
    "extract_weights",
    "import_research_module",
    "load_da3",
    "offline_guard",
    "register_backbone",
    "resize_to_long_side",
    "resolve_checkpoint",
    "set_frozen",
]
