"""Pinned inference assets and viewer configuration for the model installer.

Downloads never import research code or deserialize checkpoints. Hugging Face
handles resumable transfers; ordinary URL assets are verified before publication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen

MANIFEST = Path(__file__).with_suffix(".json")


def default_directory() -> Path:
    return Path(sys.prefix) / "share/ontic-models"


def select_models(group: str, names: list[str] | None = None) -> list[dict]:
    models = json.loads(MANIFEST.read_text())
    available = {m["name"] for m in models if group in ("all", m["group"])}
    if names and (unknown := set(names) - available):
        raise ValueError(f"Unknown models for --group {group}: {', '.join(sorted(unknown))}")
    return [m for m in models if m["name"] in (set(names) if names else available)]


def verify_file(path: Path, asset: dict) -> None:
    if path.stat().st_size != asset["size"]:
        raise RuntimeError(f"Incorrect checkpoint size: {path}; remove it and retry")
    if asset.get("sha256"):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != asset["sha256"]:
            raise RuntimeError(f"Checkpoint SHA256 mismatch: {path}; remove it and retry")


def download_torch_asset(asset: dict, hub: Path) -> Path:
    destination = hub / "checkpoints" / asset["name"]
    if destination.is_file():
        verify_file(destination, asset)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            temporary = Path(stream.name)
            with urlopen(asset["url"], timeout=60) as response:
                shutil.copyfileobj(response, stream)
        verify_file(temporary, asset)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def download_repository(spec: dict, root: Path, *, local_dir: Path | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError as exc:
        raise ImportError(
            "Download-only mode needs huggingface-hub. Run the installer without "
            "--download-only first, or install huggingface-hub in this virtualenv."
        ) from exc
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=spec["repo_id"],
                revision=spec["revision"],
                allow_patterns=[f["name"] for f in spec["files"]],
                cache_dir=str(root / "hub"),
                local_dir=str(local_dir) if local_dir else None,
            )
        )
    except HfHubHTTPError as exc:
        raise RuntimeError(
            f"Cannot download {spec['repo_id']}. For gated checkpoints, request access at "
            f"https://huggingface.co/{spec['repo_id']} and run `hf auth login`; then retry. "
            "Also check connectivity and available disk space."
        ) from exc
    for asset in spec["files"]:
        verify_file(snapshot / asset["name"], asset)
    return snapshot


def set_config_value(config: dict, dotted_key: str, value: str | bool) -> None:
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        config = config.setdefault(key, {})
    config[keys[-1]] = value


def write_config(path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(config, stream, indent=2)
        stream.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def download_weights(
    group: str,
    *,
    directory: Path | None = None,
    names: list[str] | None = None,
    dry_run: bool = False,
) -> Path:
    models = select_models(group, names)
    root = (directory or default_directory()).expanduser().resolve()
    config_path = default_directory() / "viewer-models.json"
    total = 0
    for model in models:
        assets = [
            *model.get("files", []),
            *model.get("torch_assets", []),
            *model.get("base_model", {}).get("files", []),
        ]
        size = sum(f["size"] for f in assets)
        total += size
        print(f"WEIGHTS {model['name']}: {size / 1e9:.2f} GB", flush=True)
    print(f"Assets: {total / 1e9:.2f} GB before cache reuse -> {root}", flush=True)
    print(f"Viewer config: {config_path}", flush=True)
    if dry_run:
        return config_path
    # Publish only after every selected model succeeds. Existing configurations
    # for unselected models survive incremental downloads.
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    for model in models:
        print(f"Downloading/verifying {model['name']} ...", flush=True)
        checkpoint = None
        if "repo_id" in model:
            snapshot = download_repository(model, root)
            checkpoint = snapshot / model["checkpoint"] if "checkpoint" in model else snapshot
        for asset in model.get("torch_assets", []):
            path = download_torch_asset(asset, root / "torch/hub")
            if checkpoint is None and asset["name"] == model.get("checkpoint"):
                checkpoint = path
            if "repo_id" in model:
                config["torch_hub_dir"] = str(root / "torch/hub")
        if checkpoint is None:
            raise ValueError(f"No checkpoint declared for {model['name']}")
        for target in model["targets"]:
            set_config_value(config, target, str(checkpoint))
        if model["group"] == "trackers":
            set_config_value(config, f"trackers.{model['name']}.allow_download", False)
        if base := model.get("base_model"):
            cache = root / "wan_models"
            download_repository(base, root, local_dir=cache / base["repo_id"])
            set_config_value(config, f"trackers.{model['name']}.base_model_cache_dir", str(cache))
    write_config(config_path, config)
    print("Checkpoints verified; the viewer will load these paths automatically.", flush=True)
    return config_path
