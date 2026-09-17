#!/usr/bin/env python3
"""Install all viewer backbones into the invoking virtualenv without replacing its ML stack.

Run from the workspace: ``uv run --no-sync python scripts/install_backbones.py``.
Research repositories are pinned, unmodified source checkouts inside the virtualenv.
Their inference dependencies come from ontic-nn[backbones], rather than upstream's
training/demo requirements. No weights are downloaded by installation or import checks.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("backbone_sources.json")
PROTECTED = {"torch", "torchvision", "torchaudio", "triton", "triton-windows", "numpy"}
RESEARCH_MODULES = {
    "da3": "depth_anything_3.api",
    "ma": "mapanything.models.mapanything.model",
    "vggt": "vggt_omega",
    "pi3x": "pi3.models.pi3x",
    "dvlt": "dvlt.model.dvlt.model",
    "moge3": "moge.model.v3",
}


def normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def protected_name(name: str) -> bool:
    return name in PROTECTED or name.startswith(("nvidia-", "cuda-"))


def stack_versions() -> dict[str, str]:
    """Snapshot installed ML distributions, including the exact CUDA wheel versions."""
    result = {}
    for dist in metadata.distributions():
        name = normalized(dist.metadata["Name"])
        if protected_name(name):
            result[name] = dist.version
    return result


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"+ {shlex.join(map(str, command))}", flush=True)
    return subprocess.run(command, check=True, **kwargs)


def git_output(directory: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()


def checkout(source: dict[str, str], cache: Path) -> Path:
    """Fetch into a staging directory; never overwrite a modified or partial checkout."""
    revision = source["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError(f"{source['name']}: expected a full Git commit ID")
    destination = cache / f"{source['name']}-{revision}"
    if destination.exists():
        if git_output(destination, "rev-parse", "HEAD") != revision:
            raise RuntimeError(f"Unexpected revision in {destination}; move it aside and retry")
        if git_output(destination, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(
                f"Modified research source at {destination}; move it aside and retry"
            )
    else:
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".fetch-", dir=cache) as stage:
            staging = Path(stage) / "repo"
            run(["git", "init", "--quiet", str(staging)])
            run(
                [
                    "git",
                    "-C",
                    str(staging),
                    "fetch",
                    "--quiet",
                    "--depth=1",
                    source["url"],
                    revision,
                ]
            )
            run(["git", "-C", str(staging), "checkout", "--quiet", "--detach", "FETCH_HEAD"])
            if git_output(staging, "rev-parse", "HEAD") != revision:
                raise RuntimeError(
                    f"{source['name']}: fetched revision does not match the manifest"
                )
            staging.rename(destination)
    import_path = destination / source["path"]
    if not import_path.is_dir():
        raise RuntimeError(f"Missing source directory: {import_path}")
    return import_path.resolve()


def register_sources(paths: list[Path], site_packages: Path) -> None:
    """Prioritize the pinned sources over previously pip-installed research packages."""
    pth = site_packages / "ontic_backbones.pth"
    content = (
        "# Managed by scripts/install_backbones.py; remove to disable these research sources.\n"
        f"import sys; sys.path[:0] = {list(map(str, paths))!r}\n"
    )
    temporary = pth.with_suffix(".pth.tmp")
    temporary.write_text(content)
    temporary.replace(pth)


def locked_constraints(uv: str) -> list[str]:
    """Use the workspace lock for support libraries, retaining the caller's ML stack."""
    exported = subprocess.check_output(
        [
            uv,
            "export",
            "--frozen",
            "--package",
            "ontic-nn",
            "--extra",
            "backbones",
            "--no-dev",
            "--no-hashes",
            "--no-emit-workspace",
        ],
        cwd=ROOT,
        text=True,
    )
    lines = []
    for line in exported.splitlines():
        name = re.match(r"[\w.-]+", line)
        if name is not None and not protected_name(normalized(name[0])):
            lines.append(line)
    return lines


def install_dependencies(uv: str, protected: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix="ontic-backbones-") as temporary:
        constraints = Path(temporary) / "stack.txt"
        pins = locked_constraints(uv)
        pins.extend(f"{name}=={v}" for name, v in sorted(protected.items()))
        constraints.write_text("\n".join(pins) + "\n")
        run(
            [
                uv,
                "pip",
                "install",
                "--no-config",
                "--python",
                sys.executable,
                "--constraint",
                str(constraints),
                "--editable",
                str(ROOT),
                "--editable",
                f"{ROOT / 'packages/ontic-nn'}[backbones]",
                "--editable",
                str(ROOT / "packages/ontic-data"),
                "--editable",
                str(ROOT / "packages/ontic-viz"),
            ]
        )
    after = stack_versions()
    changed = {
        name: (version, after.get(name))
        for name, version in protected.items()
        if after.get(name) != version
    }
    if changed:
        raise RuntimeError(f"ML stack changed unexpectedly: {changed}")


def check_import(name: str, cuda: bool) -> bool:
    from ontic_nn.wrappers.dvlt import install_flex_attention_shim
    from ontic_nn.wrappers.moge3 import install_triton_autotuner_shim

    if name == "dvlt":
        install_flex_attention_shim()
    if name == "moge3":
        # FlexGEMM queries GPU properties during import, even on a CPU/login node.
        for module in ("moge", "utils3d_moge", "flex_gemm", "triton"):
            if importlib.util.find_spec(module) is None:
                raise ModuleNotFoundError(f"No module named {module!r}")
        if not cuda:
            print(
                "SKIP moge3 import: sources present, but FlexGEMM requires usable CUDA", flush=True
            )
            return False
        install_triton_autotuner_shim()
    module = importlib.import_module(RESEARCH_MODULES[name])
    print(f"OK   {name} import: {module.__file__}", flush=True)
    return True


def check_installation() -> int:
    import torch

    cuda = torch.cuda.is_available()
    print(
        f"torch={torch.__version__}, CUDA runtime={torch.version.cuda}, available={cuda}",
        flush=True,
    )
    failed = []
    for name in RESEARCH_MODULES:
        try:
            check_import(name, cuda)
        except Exception as exc:
            failed.append(name)
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    print("OK   gtdepth: built into ontic-nn", flush=True)
    print("Import checks only; pretrained inference is unverified.", flush=True)
    return int(bool(failed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check the current installation only")
    args = parser.parse_args(argv)
    if args.check:
        return check_installation()
    if sys.prefix == sys.base_prefix:
        parser.error("run with a virtualenv's Python, e.g. .venv/bin/python")
    if sys.platform != "linux" or sys.version_info < (3, 12):
        parser.error("this workspace installer requires Linux and Python >=3.12")
    uv = shutil.which("uv")
    if not uv or not shutil.which("git"):
        parser.error("uv and git must be on PATH")
    protected = stack_versions()
    if not {"torch", "torchvision", "numpy"} <= protected.keys():
        parser.error("install your chosen torch, torchvision and numpy first (e.g. uv sync)")
    print(f"Target: {sys.executable} (Python {sys.version.split()[0]})", flush=True)
    print(
        "Preserving: " + ", ".join(f"{k}=={v}" for k, v in sorted(protected.items())),
        flush=True,
    )
    sources = json.loads(MANIFEST.read_text())
    cache = Path(sys.prefix) / "share" / "ontic-backbones"
    paths = [checkout(source, cache) for source in sources]
    install_dependencies(uv, protected)
    register_sources(paths, Path(sysconfig.get_path("purelib")))
    print("Pinned backbone sources installed. Checking in a fresh Python process.", flush=True)
    return subprocess.call(
        [sys.executable, str(Path(__file__).resolve()), "--check"],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Backbone installation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
