"""Load fwomo (frontier) PointTransformerV3 checkpoints into :class:`PointTransformerV3`.

Key differences handled: spconv weights ``(out, k, k, k, in)`` become
``(K**3, in, out)``; ``PointSequential`` numeric children (``cpe.0/1/2``,
``norm1.0``, ``mlp.0``, ``down.norm.0``, ``up.proj.0/1``) get their attribute
names; the unnamed final MLP ``dec.<n>`` becomes ``dec.head``.
"""

from __future__ import annotations

import re
from typing import Any

from torch import Tensor, nn

_RENAMES = [
    (re.compile(r"\.cpe\.0\."), ".cpe.conv."),
    (re.compile(r"\.cpe\.1\."), ".cpe.linear."),
    (re.compile(r"\.cpe\.2\."), ".cpe.norm."),
    (re.compile(r"\.norm0\.0\."), ".norm0."),
    (re.compile(r"\.norm1\.0\."), ".norm1."),
    (re.compile(r"\.norm2\.0\."), ".norm2."),
    (re.compile(r"\.mlp\.0\."), ".mlp."),
    (re.compile(r"\.down\.norm\.0\."), ".down.norm."),
    (re.compile(r"\.up\.(proj|proj_skip)\.0\."), r".up.\1.linear."),
    (re.compile(r"\.up\.(proj|proj_skip)\.1\."), r".up.\1.norm."),
    (re.compile(r"^dec\.\d+\."), "dec.head."),
]
_SPCONV_WEIGHT = re.compile(r"(\.cpe\.0\.weight|embedding\.stem\.conv\.weight)$")


def spconv_to_canonical(weight: Tensor) -> Tensor:
    """``(out, k, k, k, in)`` spconv layout -> ``(K**3, in, out)``."""
    out_channels, k = weight.shape[0], weight.shape[1]
    return weight.reshape(out_channels, k**3, -1).permute(1, 2, 0).contiguous()


def canonical_to_spconv(weight: Tensor) -> Tensor:
    """``(K**3, in, out)`` -> spconv ``(out, k, k, k, in)``."""
    volume, in_channels, out_channels = weight.shape
    k = round(volume ** (1 / 3))
    return weight.permute(2, 0, 1).reshape(out_channels, k, k, k, in_channels).contiguous()


def convert_fwomo_state_dict(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    """Rename and re-layout a fwomo ``PointTransformerV3.state_dict()``."""
    converted = {}
    for key, value in state_dict.items():
        if _SPCONV_WEIGHT.search(key) and value.ndim == 5:
            value = spconv_to_canonical(value)
        new_key = key
        for pattern, replacement in _RENAMES:
            new_key = pattern.sub(replacement, new_key)
        converted[new_key] = value
    return converted


def load_fwomo_state_dict(
    model: nn.Module, state_dict: dict[str, Tensor], strict: bool = True
) -> Any:
    """Load a fwomo checkpoint into ``model``; returns ``load_state_dict``'s result."""
    return model.load_state_dict(convert_fwomo_state_dict(state_dict), strict=strict)
