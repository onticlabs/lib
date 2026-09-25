"""Shared flat encoding: ``Gaussians`` <-> ``{name: tensor}`` + string metadata."""

from __future__ import annotations

import json
from typing import Any

from torch import Tensor

from ..structures.gaussians import Gaussians

EXTRAS_PREFIX = "extras."
BATCH_SHAPE_KEY = "batch_shape"


def to_tensors(g: Gaussians, metadata: dict[str, str]) -> tuple[dict[str, Tensor], dict[str, str]]:
    """Flatten fields to a name->tensor dict; extras become ``extras.<name>``."""
    tensors: dict[str, Tensor] = {}
    for name, value in g.items():
        key = EXTRAS_PREFIX + name if name in g.extras else name
        tensors[key] = value.detach().cpu().contiguous()
    if BATCH_SHAPE_KEY in metadata:
        raise ValueError(f"metadata key {BATCH_SHAPE_KEY!r} is reserved")
    meta = {BATCH_SHAPE_KEY: json.dumps(list(g.batch_shape))}
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("metadata must map str -> str")
        meta[key] = value
    return tensors, meta


def from_tensors(tensors: dict[str, Tensor], metadata: dict[str, str], device: Any) -> Gaussians:
    fields: dict[str, Any] = {}
    extras: dict[str, Tensor] = {}
    for key, value in tensors.items():
        value = value.to(device) if device is not None else value
        if key.startswith(EXTRAS_PREFIX):
            extras[key[len(EXTRAS_PREFIX) :]] = value
        else:
            fields[key] = value
    g = Gaussians(**fields, extras=extras)
    stored = metadata.get(BATCH_SHAPE_KEY)
    if stored is not None and tuple(json.loads(stored)) != g.batch_shape:
        raise ValueError(f"stored batch shape {stored} does not match tensors {g.batch_shape}")
    return g
