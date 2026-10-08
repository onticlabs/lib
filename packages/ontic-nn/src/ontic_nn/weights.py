"""Pretrained weights served from the ontic store before any public hub.

Every pinned asset the backbones, video-depth models, DINOv2 and the trackers load is also an
immutable ontic model job (``ontic commit-data --kind model``). :data:`ONTIC_WEIGHTS` maps the
public source the code names (an HF repo id or a ``torch.hub`` URL) to that job, and
:func:`ontic_path` serves the file from the CLI's local pull cache, running ``ontic pull`` when
it is absent and downloads are allowed. The hub stays the fallback: a source that is not in the
table, or a pull that fails (no CLI, no ``ontic.toml`` in ``ONTIC_REPO`` / the working directory,
no session), returns ``None`` and the caller fetches as before.

Nothing here imports the CLI; its cache layout (``ONTIC_CACHE_DIR`` / ``XDG_CACHE_HOME`` /
``~/.cache/ontic``, ``jobs/<id>/output/...``) and its command line are the contract.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class OnticWeights:
    """One ontic model job; ``source`` is the path under its ``output/`` holding the files."""

    job: str
    source: str = "data"


#: Public source (HF repo id or torch.hub URL) -> the ontic job that holds the same bytes.
ONTIC_WEIGHTS: Dict[str, OnticWeights] = {
    "depth-anything/DA3NESTED-GIANT-LARGE-1.1": OnticWeights(
        "12bb1b62-2154-4e31-8ea3-2b4869065ac5"
    ),
    "facebook/map-anything": OnticWeights("3d39bd10-db0f-4203-9445-1a8a67cb5063"),
    "facebook/VGGT-Omega": OnticWeights("e96685ea-5bd1-4975-b1a1-3fd71582c990"),
    "yyfz233/Pi3X": OnticWeights("4151f5cf-178f-4693-aef3-aecfa958590e"),
    "nvidia/dvlt": OnticWeights("9fb5b5a3-e8ba-494d-b185-947724cbc733"),
    "Ruicheng/moge-3-vitl": OnticWeights("9275a313-cc21-4e95-8cdb-1058ae092bb1"),
    "lpiccinelli/velodepth": OnticWeights("0e193b20-3ccb-4c08-b98e-7412536cc7b9"),
    "depth-anything/Metric-Video-Depth-Anything-Small": OnticWeights(
        "eb690581-45fd-412b-9c61-6a1f1657d68a"
    ),
    "depth-anything/Metric-Video-Depth-Anything-Base": OnticWeights(
        "4ea88071-7b16-4175-815b-b2d225e37fdb"
    ),
    "depth-anything/Metric-Video-Depth-Anything-Large": OnticWeights(
        "23fe7661-82e9-47e4-bf4b-b5ea8771d8cf"
    ),
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth": OnticWeights(
        "408d98ca-93a3-4ec4-aa89-d07ac1625d0b"
    ),
    "ethz-vlg/mvtracker": OnticWeights("5594b910-5c4a-4c8f-9a9e-fc86683eaed9"),
    "zbww/tapip3d": OnticWeights("7c2dc8b7-a1ba-41df-8dc8-a00665cb8e7d"),
    "facebook/cotracker3": OnticWeights("bae32cb6-f7ac-494d-91d1-115b945b758c"),
    "trackcraft3r/checkpoint": OnticWeights("71bd1d90-d6f7-4e28-bf05-e660fd58258d"),
    "Wan-AI/Wan2.1-T2V-1.3B": OnticWeights(
        "71bd1d90-d6f7-4e28-bf05-e660fd58258d", "data/wan_models/Wan-AI/Wan2.1-T2V-1.3B"
    ),
}


def ontic_cache_root(env: Optional[Dict[str, str]] = None, home: Optional[Path] = None) -> Path:
    """The CLI's local cache root (same precedence as ``ontic``: ``ONTIC_CACHE_DIR``,
    ``XDG_CACHE_HOME``, a legacy ``~/.ontic/cache``, else ``~/.cache/ontic``)."""
    env = os.environ if env is None else env
    home = Path.home() if home is None else home
    if cache_dir := env.get("ONTIC_CACHE_DIR"):
        return Path(cache_dir).expanduser() / "ontic"
    if cache_dir := env.get("XDG_CACHE_HOME"):
        return Path(cache_dir).expanduser() / "ontic"
    legacy = home / ".ontic" / "cache"
    if legacy.is_dir():
        return legacy
    return home / ".cache" / "ontic"


def _pull(entry: OnticWeights) -> bool:
    """``ontic pull <job> --path <source>`` from ``ONTIC_REPO`` (else the working directory)."""
    cli = shutil.which("ontic")
    if cli is None:
        return False
    cwd = os.environ.get("ONTIC_REPO") or None
    proc = subprocess.run(
        [cli, "pull", entry.job, "--path", entry.source, "--json"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
        warnings.warn(
            f"ontic pull {entry.job} failed ({tail[0]}); falling back to the public hub. "
            "Run from the experiments repo or set ONTIC_REPO to it.",
            stacklevel=3,
        )
        return False
    return True


def ontic_path(
    source: str, filename: Optional[str] = None, *, allow_download: bool = True
) -> Optional[Path]:
    """Local path of ``source`` (``ONTIC_WEIGHTS`` key) in the ontic cache, or ``None``.

    With ``filename`` the path is that file inside the artifact, else the artifact directory.
    A miss is pulled when ``allow_download`` is set and no offline guard is active
    (``HF_HUB_OFFLINE=1`` marks the lib's cache-only mode).
    """
    entry = ONTIC_WEIGHTS.get(source)
    if entry is None:
        return None
    root = ontic_cache_root() / "jobs" / entry.job / "output" / entry.source
    target = root / filename if filename else root
    if target.exists():
        return target
    offline = os.environ.get("HF_HUB_OFFLINE") == "1"
    if not allow_download or offline or not _pull(entry):
        return None
    return target if target.exists() else None
