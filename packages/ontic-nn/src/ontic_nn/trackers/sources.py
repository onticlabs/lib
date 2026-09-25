"""Locate research checkouts registered by the unified workspace installer."""

from __future__ import annotations

import json
from pathlib import Path
import sys


def installed_repo(name: str) -> Path | None:
    """Read the invoking environment's source index only when a model is built.

    Generic upstream packages such as TAPIP3D's ``models`` must not be added to
    every Python process's path. Adapters import these sources in their existing
    guarded contexts; explicit configuration and environment overrides win.
    """
    index = Path(sys.prefix) / "share" / "ontic-trackers" / "sources.json"
    if not index.is_file():
        return None
    data = json.loads(index.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"Unsupported tracker source index: {index}")
    raw = data["sources"].get(name)
    if raw is None:
        return None
    path = Path(raw)
    if not path.is_absolute() or not path.is_dir():
        raise FileNotFoundError(
            f"Missing installed {name} checkout: {path}. "
            "Rerun scripts/install_models.py --group trackers in this environment."
        )
    return path
