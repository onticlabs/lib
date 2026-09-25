"""``.npz`` backend: one array per field, metadata as a JSON string array."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from ..structures.gaussians import Gaussians
from ._codec import from_tensors, to_tensors

_METADATA_KEY = "__metadata__"


def save(path: str, g: Gaussians, *, metadata: dict[str, str]) -> None:
    tensors, meta = to_tensors(g, metadata)
    arrays = {key: value.numpy() for key, value in tensors.items()}
    arrays[_METADATA_KEY] = np.array(json.dumps(meta))
    np.savez(path, **arrays)


def load(path: str, *, device: Any) -> Gaussians:
    with np.load(path) as data:
        meta = json.loads(str(data[_METADATA_KEY])) if _METADATA_KEY in data else {}
        tensors = {
            key: torch.from_numpy(np.ascontiguousarray(data[key]))
            for key in data.files
            if key != _METADATA_KEY
        }
    return from_tensors(tensors, meta, device)
