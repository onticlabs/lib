#!/usr/bin/env python3
"""Install viewer backbones, video depth and all four trackers in one command.

Run ``uv run --no-sync python scripts/install_models.py`` in a Linux Python 3.12+
virtualenv with the desired torch/torchvision/NumPy stack already installed.
TAPIP3D needs a CUDA toolkit matching torch and a C++ compiler. Installation
preserves that stack and never downloads pretrained weights.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig

import install_backbones as backbones

TRACKER_MANIFEST = Path(__file__).with_name("tracker_sources.json")
TRACKER_MODULES = {
    "mvtracker": "mvtracker.models.evaluation_predictor_3dpt",
    "tapip3d": "models.point_tracker_3d",
    "cotracker3": "cotracker.predictor",
    "trackcraft3r": "evaluation.wan_scene_flow_predictor",
}
GROUPS = ("all", "backbones", "trackers")


def selected_groups(group: str) -> tuple[str, ...]:
    return ("backbones", "trackers") if group == "all" else (group,)


def cuda_build_environment() -> dict[str, str]:
    """Fail before installation if the required TAPIP3D build cannot run."""
    import torch

    hint = (
        "Install a toolkit matching torch and set CUDA_HOME. "
        "For a partial installation without TAPIP3D's extension, pass --skip-cuda-build."
    )
    root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(root) / "bin/nvcc") if root else shutil.which("nvcc")
    if not torch.version.cuda or not nvcc or not Path(nvcc).is_file():
        raise RuntimeError(f"TAPIP3D needs CUDA-enabled torch and nvcc. {hint}")
    version = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release (\d+)\.(\d+)", version)
    if match is None or match[1] != torch.version.cuda.split(".")[0]:
        raise RuntimeError(
            f"nvcc must match torch's CUDA major version ({torch.version.cuda}). {hint}"
        )
    if not shutil.which(os.environ.get("CXX", "c++")):
        raise RuntimeError("TAPIP3D needs a C++ compiler; install it or set CXX.")
    if not torch.cuda.is_available() and not os.environ.get("TORCH_CUDA_ARCH_LIST"):
        raise RuntimeError(
            "No usable CUDA device for TAPIP3D's build. Set TORCH_CUDA_ARCH_LIST to the "
            "target GPU architecture for a build host without a GPU, or use --skip-cuda-build."
        )
    return {
        **os.environ,
        "CUDA_HOME": str(Path(nvcc).resolve().parents[1]),
        "MAX_JOBS": os.environ.get("MAX_JOBS", "2"),
    }


def register_trackers(paths: dict[str, Path], site_packages: Path, cache: Path) -> None:
    """Expose unique packages globally; leave generic names to the guarded adapters."""
    backbones.register_sources(
        [paths[name] for name in ("mvtracker", "cotracker3")],
        site_packages,
        filename="ontic_trackers.pth",
    )
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / "sources.json.tmp"
    temporary.write_text(
        json.dumps(
            {"schema_version": 1, "sources": {k: str(v) for k, v in paths.items()}}, indent=2
        )
        + "\n"
    )
    temporary.replace(cache / "sources.json")


def build_pointops(uv: str, repo: Path, environment: dict[str, str]) -> None:
    """Build only TAPIP3D's required extension against the selected torch."""
    source = repo / "third_party/pointops2"
    if not (source / "setup.py").is_file():
        raise FileNotFoundError(f"TAPIP3D source is missing {source / 'setup.py'}")
    backbones.run(
        [
            uv,
            "pip",
            "install",
            "--no-config",
            "--python",
            sys.executable,
            "--no-deps",
            "--no-build-isolation",
            "--reinstall",
            str(source),
        ],
        cwd=source,
        env=environment,
    )


def check_one(name: str) -> int:
    """Import one upstream entry point without constructing a model or fetching weights."""
    import torch

    if name in backbones.RESEARCH_MODULES:
        backbones.check_import(name, torch.cuda.is_available())
        return 0
    from ontic_nn.trackers.sources import installed_repo

    repo = installed_repo(name)
    if repo is None:
        raise ImportError(
            f"No registered {name} source; run scripts/install_models.py --group trackers"
        )
    # This process checks only one upstream, so generic research names cannot
    # collide with another model's imports from an earlier check.
    sys.path.insert(0, str(repo))
    if name == "tapip3d":
        importlib.import_module("pointops2_cuda")
    module = importlib.import_module(TRACKER_MODULES[name])
    print(f"OK   {name} import: {module.__file__}", flush=True)
    return 0


def check_installation(group: str, skip_cuda: bool) -> int:
    groups = selected_groups(group)
    names = [
        *(backbones.RESEARCH_MODULES if "backbones" in groups else []),
        *(TRACKER_MODULES if "trackers" in groups else []),
    ]
    failed = []
    for name in names:
        if name == "tapip3d" and skip_cuda:
            print(
                "SKIP tapip3d: --skip-cuda-build requested; tracker readiness is unverified",
                flush=True,
            )
            continue
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--check-one", name],
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1"},
        )
        if result.returncode:
            failed.append(name)
    if "backbones" in groups:
        print("OK   gtdepth: built into ontic-nn", flush=True)
    print("Import checks only; pretrained weights and inference are not checked.", flush=True)
    if failed:
        print("Failed models: " + ", ".join(failed), file=sys.stderr)
    return int(bool(failed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group", choices=GROUPS, default="all", help="model families to install/check"
    )
    parser.add_argument(
        "--check", action="store_true", help="check imports only, in separate processes"
    )
    parser.add_argument(
        "--skip-cuda-build",
        action="store_true",
        help="partial setup: skip TAPIP3D's extension and import check",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve dependencies and list sources without installing",
    )
    parser.add_argument(
        "--check-one",
        choices=[*backbones.RESEARCH_MODULES, *TRACKER_MODULES],
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    if args.check_one:
        return check_one(args.check_one)
    if args.check:
        return check_installation(args.group, args.skip_cuda_build)
    if sys.prefix == sys.base_prefix or sys.platform != "linux" or sys.version_info < (3, 12):
        parser.error("use a Linux Python >=3.12 virtualenv with your selected ML stack installed")
    uv = shutil.which("uv")
    if not uv or not shutil.which("git"):
        parser.error("uv and git must be on PATH")
    protected = backbones.stack_versions()
    if not {"torch", "torchvision", "numpy"} <= protected.keys():
        parser.error("install your chosen torch, torchvision and numpy first (e.g. uv sync)")
    groups = selected_groups(args.group)
    environment = None
    if "trackers" in groups and not args.skip_cuda_build and not args.dry_run:
        environment = cuda_build_environment()
    print(f"Target: {sys.executable}", flush=True)
    print("Preserving: " + ", ".join(f"{k}=={v}" for k, v in sorted(protected.items())), flush=True)
    # Resolve before touching sources, and use the same protected stack for all packages.
    backbones.install_dependencies(
        uv,
        protected,
        extras=groups,
        data_extras=("all",),
        viz_extras=("robot", "rerun"),
        dry_run=args.dry_run,
    )
    site_packages = Path(sysconfig.get_path("purelib"))
    for group in groups:
        manifest = backbones.MANIFEST if group == "backbones" else TRACKER_MANIFEST
        sources = json.loads(manifest.read_text())
        cache = Path(sys.prefix) / "share" / f"ontic-{group}"
        if args.dry_run:
            for source in sources:
                print(f"SOURCE {source['name']} @ {source['revision']} -> {cache}", flush=True)
            continue
        paths = {s["name"]: backbones.checkout(s, cache) for s in sources}
        if group == "backbones":
            backbones.register_sources(list(paths.values()), site_packages)
        else:
            register_trackers(paths, site_packages, cache)
            if environment is not None:
                build_pointops(uv, paths["tapip3d"], environment)
    if args.dry_run:
        print(
            "Dry run: no sources installed, no CUDA build attempted, no weights downloaded.",
            flush=True,
        )
        return 0
    return check_installation(args.group, args.skip_cuda_build)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.CalledProcessError) as exc:
        print(f"Model installation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
