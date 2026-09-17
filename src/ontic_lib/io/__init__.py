"""Save and load :class:`~ontic_lib.structures.Gaussians`; format chosen by file suffix.

``.npz`` (numpy, core), ``.safetensors`` (``ontic-lib[safetensors]``) and ``.ply``
(``ontic-lib[ply]``, standard 3DGS vertex layout, unbatched only). Round trips through
``.npz`` and ``.safetensors`` are exact; ``.ply`` stores float32 log-scales and
logit-opacities and no mask/extras/covariances.
"""

from __future__ import annotations

import os
from typing import Any

from ..structures.gaussians import Gaussians

_FORMATS = {".npz": "_npz", ".safetensors": "_safetensors", ".ply": "_ply"}


def _backend(path: str | os.PathLike[str]) -> Any:
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    module = _FORMATS.get(suffix)
    if module is None:
        raise ValueError(
            f"unsupported suffix {suffix!r} for {os.fspath(path)!r}; "
            f"expected one of {sorted(_FORMATS)}"
        )
    import importlib

    return importlib.import_module(f".{module}", __name__)


def save_gaussians(
    path: str | os.PathLike[str],
    gaussians: Gaussians,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write ``gaussians`` to ``path``; ``metadata`` is stored as string key/value pairs."""
    _backend(path).save(os.fspath(path), gaussians, metadata=dict(metadata or {}))


def load_gaussians(path: str | os.PathLike[str], *, device: Any = None) -> Gaussians:
    """Read ``Gaussians`` from ``path``, moving tensors to ``device`` (default: CPU)."""
    return _backend(path).load(os.fspath(path), device=device)


__all__ = ["load_gaussians", "save_gaussians"]
