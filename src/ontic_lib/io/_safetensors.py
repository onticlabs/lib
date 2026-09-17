"""``.safetensors`` backend (``ontic-lib[safetensors]``)."""

from __future__ import annotations

from typing import Any

from ..deps import missing_dependency
from ..structures.gaussians import Gaussians
from ._codec import from_tensors, to_tensors


def _safetensors() -> Any:
    try:
        import safetensors
        import safetensors.torch
    except ImportError as e:
        raise missing_dependency(
            "safetensors Gaussians files",
            package="ontic-lib",
            extra="safetensors",
            needs="the `safetensors` package",
        ) from e
    return safetensors


def save(path: str, g: Gaussians, *, metadata: dict[str, str]) -> None:
    st = _safetensors()
    tensors, meta = to_tensors(g, metadata)
    st.torch.save_file(tensors, path, metadata=meta)


def load(path: str, *, device: Any) -> Gaussians:
    st = _safetensors()
    target = "cpu" if device is None else device
    with st.safe_open(path, framework="pt", device=str(target)) as f:
        meta = f.metadata() or {}
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    return from_tensors(tensors, meta, None)
