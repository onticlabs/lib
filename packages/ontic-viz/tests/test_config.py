"""ViewConfig presets: JSON round trip, store semantics, dataset-config workspace boxes."""

from __future__ import annotations

from dataclasses import dataclass, field

from ontic_viz.backbone_viewer.config import (
    DATASET_DEFAULTS,
    ViewConfig,
    dataset_default,
    dataset_defaults,
    ensure_default_presets,
    from_jsonable,
    load_presets,
    save_preset,
    to_jsonable,
    workspace_kwargs,
)


def test_json_roundtrip_and_unknown_keys():
    cfg = ViewConfig(
        stride=4,
        drop_conf_pct=20.0,
        voxel_size=0.01,
        fps_max_points=1234,
        align_mode="metric_mono",
        crop_enabled=True,
        ws_min=(-1.0, -2.0, -3.0),
        ws_max=(1.0, 2.0, 3.0),
    )
    restored = from_jsonable(to_jsonable(cfg))
    assert restored == cfg and isinstance(restored.ws_min, tuple)
    assert from_jsonable({**to_jsonable(ViewConfig()), "obsolete": 1}) == ViewConfig()


def test_dataset_default_fallback():
    assert dataset_default("taco").crop_enabled
    assert dataset_default("nope") == ViewConfig()
    assert set(DATASET_DEFAULTS) == {
        "dextris",
        "hocap",
        "taco",
        "genesis",
        "physinone",
        "synthrobot",
    }


def test_presets_store_roundtrip(tmp_path):
    path = tmp_path / "sub" / "presets.json"
    assert load_presets(path) == {}
    save_preset("a", ViewConfig(stride=1), path)
    save_preset("b", ViewConfig(stride=2, ws_min=(0.1, 0.2, 0.3)), path)
    save_preset("a", ViewConfig(stride=9), path)
    loaded = load_presets(path)
    assert set(loaded) == {"a", "b"} and loaded["a"].stride == 9
    assert loaded["b"] == ViewConfig(stride=2, ws_min=(0.1, 0.2, 0.3))


def test_ensure_default_presets_seeds_without_overwriting(tmp_path):
    path = tmp_path / "presets.json"
    tuned = ViewConfig(stride=9, drop_conf_pct=59.0, ws_min=(-0.3, -0.9, 0.4))
    save_preset("taco_default", tuned, path)
    save_preset("mine", ViewConfig(stride=7), path)
    cfgs = ensure_default_presets(
        path, defaults={"taco": ViewConfig(), "hocap": ViewConfig(stride=3)}
    )
    assert cfgs["taco_default"] == tuned and cfgs["mine"].stride == 7
    assert cfgs["hocap_default"].stride == 3
    assert load_presets(path)["hocap_default"].stride == 3  # persisted


def test_ensure_default_presets_tolerates_read_only_location():
    cfgs = ensure_default_presets("/proc/nope/presets.json", defaults={"x": ViewConfig()})
    assert "x_default" in cfgs


def test_workspace_from_dataset_config_fields():
    @dataclass
    class WithBox:
        workspace_min: tuple = field(default_factory=lambda: (-1.0, -2.0, 0.0))
        workspace_max: tuple = field(default_factory=lambda: (1.0, 2.0, 1.0))

    @dataclass
    class NoBox:
        workspace_min: object = None
        workspace_max: object = None

    @dataclass
    class Plain:
        root: str = ""

    assert workspace_kwargs(WithBox) == {
        "ws_min": (-1.0, -2.0, 0.0),
        "ws_max": (1.0, 2.0, 1.0),
        "crop_enabled": True,
    }
    assert workspace_kwargs(NoBox) == {} and workspace_kwargs(Plain) == {}
    defaults = dataset_defaults({"taco": WithBox, "genesis": NoBox, "new": Plain})
    assert defaults["taco"].ws_min == (-1.0, -2.0, 0.0) and defaults["taco"].crop_enabled
    assert defaults["genesis"].align_mode == "none" and not defaults["genesis"].crop_enabled
    assert defaults["new"] == ViewConfig()
