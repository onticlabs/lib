"""Per-dataset view defaults and the named-preset JSON store.

A :class:`ViewConfig` bundles the cloud controls and the workspace crop box. The
baked-in per-dataset defaults (:data:`DATASET_DEFAULTS`) seed a ``<dataset>_default``
preset into the store only when it is missing; the store is the source of truth.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

Vec3 = tuple[float, float, float]

#: Env var overriding the default presets store location.
PRESETS_ENV = "ONTIC_BACKBONE_VIEWER_PRESETS"
#: Default presets store (user-writable).
PRESETS_PATH = Path(
    os.environ.get(PRESETS_ENV)
    or Path.home() / ".config" / "ontic" / "backbone_viewer_presets.json"
)


@dataclass
class ViewConfig:
    """A full set of viewer controls (one preset / one dataset default)."""

    stride: int = 2
    drop_conf_pct: float = 0.0
    voxel_size: float = 0.0
    sfc_stride: int = 1
    fps_max_points: int = 0
    point_size: float = 0.006
    color_mode: str = "RGB"
    align_mode: str = "sim3_points"  # one of runner.ALIGN_MODES
    crop_enabled: bool = False
    ws_min: Vec3 = (-1.0, -1.0, -1.0)
    ws_max: Vec3 = (1.0, 1.0, 1.0)


DATASET_DEFAULTS: dict[str, ViewConfig] = {
    "dextris": ViewConfig(
        drop_conf_pct=26.0,
        voxel_size=0.01,
        ws_min=(-0.95, -0.73, -0.239),
        ws_max=(0.5, 0.69, 0.544),
    ),
    "hocap": ViewConfig(
        drop_conf_pct=50.0,
        voxel_size=0.01,
        crop_enabled=True,
        ws_min=(-0.96, -0.83, -0.3),
        ws_max=(0.84, 0.48, 0.3),
    ),
    "taco": ViewConfig(
        drop_conf_pct=59.0,
        voxel_size=0.01,
        crop_enabled=True,
        ws_min=(-0.354, -0.994, 0.401),
        ws_max=(0.662, 0.323, 1.409),
    ),
    "genesis": ViewConfig(stride=1, point_size=0.004, align_mode="none"),
    "physinone": ViewConfig(stride=4, point_size=0.008),
    "synthrobot": ViewConfig(stride=2, point_size=0.006),
}

GENERIC_DEFAULT = ViewConfig()


def _field_default(cfg_cls, name: str):
    field = next((f for f in dataclasses.fields(cfg_cls) if f.name == name), None)
    if field is None:
        return None
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return None if field.default is dataclasses.MISSING else field.default


def workspace_kwargs(cfg_cls) -> dict:
    """``ws_min`` / ``ws_max`` / ``crop_enabled`` from a dataset config class's
    ``workspace_min`` / ``workspace_max`` field defaults (``{}`` when it has no box)."""
    wmin, wmax = _field_default(cfg_cls, "workspace_min"), _field_default(cfg_cls, "workspace_max")
    if wmin is None or wmax is None:
        return {}
    return {"ws_min": tuple(wmin), "ws_max": tuple(wmax), "crop_enabled": True}


def dataset_defaults(registry: dict | None = None) -> dict[str, ViewConfig]:
    """:data:`DATASET_DEFAULTS` with workspace boxes taken from the dataset configs in
    ``registry`` (``ontic_data.DATASETS`` by default); datasets only in the registry get
    the generic default."""
    if registry is None:
        from ontic_data import DATASETS as registry
    defaults = dict(DATASET_DEFAULTS)
    for name, cfg_cls in registry.items():
        defaults[name] = dataclasses.replace(
            defaults.get(name, GENERIC_DEFAULT), **workspace_kwargs(cfg_cls)
        )
    return defaults


def dataset_default(name: str) -> ViewConfig:
    """The baked-in default for a dataset (generic fallback)."""
    return DATASET_DEFAULTS.get(name, GENERIC_DEFAULT)


def to_jsonable(cfg: ViewConfig) -> dict:
    d = asdict(cfg)
    d["ws_min"] = list(cfg.ws_min)
    d["ws_max"] = list(cfg.ws_max)
    return d


def from_jsonable(d: dict) -> ViewConfig:
    """Inverse of :func:`to_jsonable`; unknown keys are dropped."""
    d = dict(d)
    d["ws_min"] = tuple(d["ws_min"])
    d["ws_max"] = tuple(d["ws_max"])
    valid = {f.name for f in dataclasses.fields(ViewConfig)}
    return ViewConfig(**{k: v for k, v in d.items() if k in valid})


def load_presets(path: Path | str = PRESETS_PATH) -> dict[str, ViewConfig]:
    path = Path(path)
    if not path.exists():
        return {}
    return {name: from_jsonable(d) for name, d in json.loads(path.read_text()).items()}


def _write(path: Path, presets: dict[str, ViewConfig]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({n: to_jsonable(c) for n, c in presets.items()}, indent=2))


def save_preset(name: str, cfg: ViewConfig, path: Path | str = PRESETS_PATH) -> None:
    """Add/overwrite preset ``name`` in the store, keeping the rest."""
    path = Path(path)
    presets = load_presets(path)
    presets[name] = cfg
    _write(path, presets)


def ensure_default_presets(
    path: Path | str = PRESETS_PATH, defaults: dict[str, ViewConfig] | None = None
) -> dict[str, ViewConfig]:
    """All presets, seeding a missing ``<dataset>_default`` for each entry of
    ``defaults`` (:func:`dataset_defaults` by default). Existing entries are never
    overwritten; the write is best-effort so a read-only location still works."""
    path = Path(path)
    presets = load_presets(path)
    added = False
    for ds, cfg in (defaults if defaults is not None else dataset_defaults()).items():
        key = f"{ds}_default"
        if key not in presets:
            presets[key] = cfg
            added = True
    if added:
        try:
            _write(path, presets)
        except OSError:
            pass
    return presets
