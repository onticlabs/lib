"""Downloads must be pinned, complete, and safe to resume before config publication."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "model_weights", Path(__file__).resolve().parents[1] / "scripts/model_weights.py"
)
weights = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(weights)


def asset(name="model.pth", data=b"checkpoint"):
    return dict(name=name, size=len(data), sha256=hashlib.sha256(data).hexdigest())


def test_manifest_covers_viewer_models_and_auxiliary_assets():
    from ontic_nn.video_depth import VIDEO_DEPTH_MODELS
    from ontic_nn.wrappers.registry import BACKBONES

    models = weights.select_models("all")
    targets = {target for m in models for target in m["targets"]}
    assert {f"backbone_checkpoints.{name}" for name in BACKBONES} <= targets
    assert {f"video_checkpoints.{name}" for name in VIDEO_DEPTH_MODELS} <= targets
    assert {m["name"] for m in weights.select_models("trackers")} == {
        "mvtracker",
        "tapip3d",
        "cotracker3",
        "trackcraft3r",
    }
    for model in models:
        for repo in [model, model.get("base_model", {})]:
            if "repo_id" in repo:
                assert len(repo["revision"]) == 40
                int(repo["revision"], 16)
                assert repo["files"]
                assert all(f["size"] > 0 for f in repo["files"])
    craft = weights.select_models("all", ["trackcraft3r"])[0]
    assert {f["name"] for f in craft["base_model"]["files"]} >= {
        "Wan2.1_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors",
        "google/umt5-xxl/tokenizer.json",
    }
    velo = weights.select_models("all", ["velodepth"])[0]
    assert len(velo["torch_assets"]) == 2
    assert all(len(a["sha256"]) == 64 for a in velo["torch_assets"])
    with pytest.raises(ValueError, match="Unknown models"):
        weights.select_models("trackers", ["vggt"])


@pytest.mark.parametrize("corrupt", [False, True])
def test_url_download_publishes_only_verified_files_and_reuses_cache(
    tmp_path, monkeypatch, corrupt
):
    data = b"checkpoint"
    spec = {**asset(), "url": "https://example.test/model.pth"}
    monkeypatch.setattr(
        weights, "urlopen", lambda *a, **kw: io.BytesIO(b"corruption" if corrupt else data)
    )
    destination = tmp_path / "checkpoints/model.pth"
    if corrupt:
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            weights.download_torch_asset(spec, tmp_path)
        assert not destination.exists()
        assert not list(destination.parent.iterdir())
    else:
        assert weights.download_torch_asset(spec, tmp_path) == destination
        monkeypatch.setattr(weights, "urlopen", lambda *a, **kw: pytest.fail("cache redownloaded"))
        assert weights.download_torch_asset(spec, tmp_path).read_bytes() == data
        destination.write_bytes(b"corruption")
        with pytest.raises(RuntimeError, match="SHA256 mismatch"):
            weights.download_torch_asset(spec, tmp_path)


def test_hf_download_is_pinned_and_checks_contents(tmp_path, monkeypatch):
    import huggingface_hub

    spec = dict(repo_id="example/model", revision="a" * 40, files=[asset()])
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        (tmp_path / "model.pth").write_bytes(b"checkpoint")
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    assert weights.download_repository(spec, tmp_path) == tmp_path
    assert calls[0]["allow_patterns"] == ["model.pth"]
    assert calls[0]["revision"] == spec["revision"]
    assert calls[0]["cache_dir"] == str(tmp_path / "hub")


def test_failed_download_preserves_config_then_retry_merges_models(tmp_path, monkeypatch):
    config_path = tmp_path / "config/viewer-models.json"
    config_path.parent.mkdir()
    original = {
        "backbone_checkpoints": {"vggt": "/existing/vggt.pt"},
        "trackers": {"mvtracker": {"repo_path": "/research/mvtracker"}},
    }
    config_path.write_text(json.dumps(original))
    monkeypatch.setattr(weights, "default_directory", lambda: config_path.parent)
    calls = []

    def download(spec, root, *, local_dir=None):
        calls.append((spec, local_dir))
        if local_dir:
            raise RuntimeError("transfer interrupted")
        return root / spec["repo_id"]

    monkeypatch.setattr(weights, "download_repository", download)
    with pytest.raises(RuntimeError, match="interrupted"):
        weights.download_weights(
            "trackers", directory=tmp_path, names=["mvtracker", "trackcraft3r"]
        )
    assert json.loads(config_path.read_text()) == original
    assert calls[-1][1] == tmp_path / "wan_models/Wan-AI/Wan2.1-T2V-1.3B"
    monkeypatch.setattr(
        weights, "download_repository", lambda spec, root, **kw: root / spec["repo_id"]
    )
    weights.download_weights("trackers", directory=tmp_path, names=["mvtracker", "trackcraft3r"])
    config = json.loads(config_path.read_text())
    assert config["backbone_checkpoints"] == original["backbone_checkpoints"]
    assert config["trackers"]["mvtracker"]["repo_path"] == "/research/mvtracker"
    craft = config["trackers"]["trackcraft3r"]
    assert craft["base_model_cache_dir"] == str(tmp_path / "wan_models")
    assert craft["allow_download"] is False


def test_download_dry_run_does_not_create_directories_or_fetch(tmp_path, monkeypatch):
    monkeypatch.setattr(weights, "default_directory", lambda: tmp_path / "config")
    monkeypatch.setattr(
        weights, "download_repository", lambda *a, **kw: pytest.fail("network used")
    )
    weights.download_weights("all", directory=tmp_path / "assets", dry_run=True)
    assert not list(tmp_path.iterdir())


def test_viewer_loads_installer_defaults_and_explicit_config_takes_precedence(
    tmp_path, monkeypatch
):
    from ontic_viz.backbone_viewer import cli

    monkeypatch.setattr(cli.sys, "prefix", str(tmp_path))
    assert cli.load_model_config(None) == {}
    default = tmp_path / "share/ontic-models/viewer-models.json"
    default.parent.mkdir(parents=True)
    default.write_text('{"backbone_checkpoints": {"vggt": "/vggt"}}')
    assert cli.load_model_config(None)["backbone_checkpoints"]["vggt"] == "/vggt"
    explicit = tmp_path / "explicit.json"
    explicit.write_text('{"trackers": {}}')
    assert cli.load_model_config(explicit) == {"trackers": {}}
    with pytest.raises(FileNotFoundError):
        cli.load_model_config(tmp_path / "missing.json")
