"""Installer invariants: pinned source reuse and preservation of the selected ML stack."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "install_backbones.py"
spec = importlib.util.spec_from_file_location("backbone_installer", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def test_stack_snapshot_includes_cuda_libraries(monkeypatch):
    distributions = [
        SimpleNamespace(metadata={"Name": name}, version=version)
        for name, version in [
            ("torch", "2.12.1+cu130"),
            ("numpy", "2.5.1"),
            ("nvidia_cuda_runtime", "13.0.96"),
            ("cuda-bindings", "13.3.1"),
            ("Pillow", "12.0"),
        ]
    ]
    monkeypatch.setattr(installer.metadata, "distributions", lambda: distributions)
    assert installer.stack_versions() == {
        "torch": "2.12.1+cu130",
        "numpy": "2.5.1",
        "nvidia-cuda-runtime": "13.0.96",
        "cuda-bindings": "13.3.1",
    }


def test_locked_support_constraints_do_not_override_the_selected_cuda_stack(monkeypatch):
    export = (
        "# generated\n"
        "torch==2.12.1\n"
        "nvidia-cudnn-cu13==9.20.0.48 ; sys_platform == 'linux'\n"
        "cuda-toolkit==13.0.2\n"
        "numpy==2.5.1\n"
        "open3d==0.20.0 ; python_version >= '3.13'\n"
    )
    monkeypatch.setattr(installer.subprocess, "check_output", lambda *a, **kw: export)
    assert installer.locked_constraints("uv") == ["open3d==0.20.0 ; python_version >= '3.13'"]


def test_install_constrains_stack_without_pulling_in_audio(monkeypatch):
    protected = {"torch": "2.12.1+cu130", "numpy": "2.5.1", "triton": "3.7.1"}
    monkeypatch.setattr(installer, "stack_versions", lambda: protected.copy())
    monkeypatch.setattr(installer, "locked_constraints", lambda uv, **kw: ["open3d==0.20.0"])
    calls = []

    def install(command):
        constraints = Path(command[command.index("--constraint") + 1]).read_text()
        calls.append(constraints)

    monkeypatch.setattr(installer, "run", install)
    installer.install_dependencies("uv", protected)
    assert len(calls) == 1
    assert set(calls[0].splitlines()) == {
        "torch==2.12.1+cu130",
        "numpy==2.5.1",
        "triton==3.7.1",
        "open3d==0.20.0",
    }


def test_changed_stack_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(installer, "locked_constraints", lambda uv, **kw: [])
    monkeypatch.setattr(installer, "run", lambda command: None)
    monkeypatch.setattr(installer, "stack_versions", lambda: {"torch": "2.4.1"})
    with pytest.raises(RuntimeError, match="ML stack changed"):
        installer.install_dependencies("uv", {"torch": "2.12.1"})


def test_checkout_uses_pin_reuses_cache_and_preserves_edits(tmp_path, monkeypatch):
    upstream = tmp_path / "upstream"
    upstream.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(upstream), *args], text=True).strip()

    git("init", "--quiet")
    (upstream / "model.py").write_text("VERSION = 1\n")
    git("add", "model.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "one")
    revision = git("rev-parse", "HEAD")
    (upstream / "model.py").write_text("VERSION = 2\n")
    git("add", "model.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "two")
    source = {"name": "research", "url": str(upstream), "revision": revision, "path": "."}
    checkout = installer.checkout(source, tmp_path / "cache")
    assert (checkout / "model.py").read_text() == "VERSION = 1\n"

    def no_fetch(*args, **kwargs):
        pytest.fail("an existing pinned checkout must not be fetched again")

    monkeypatch.setattr(installer, "run", no_fetch)
    assert installer.checkout(source, tmp_path / "cache") == checkout
    (checkout / "model.py").write_text("LOCAL EDIT\n")
    with pytest.raises(RuntimeError, match="Modified research source"):
        installer.checkout(source, tmp_path / "cache")
    assert (checkout / "model.py").read_text() == "LOCAL EDIT\n"


def test_registration_prioritizes_pin_over_an_older_install(tmp_path):
    old = tmp_path / "old"
    pinned = tmp_path / "pinned 'source'"
    site = tmp_path / "site"
    for directory in (old, pinned, site):
        directory.mkdir()
    (old / "ontic_test_backbone.py").write_text("VERSION = 'old'\n")
    (pinned / "ontic_test_backbone.py").write_text("VERSION = 'pinned'\n")
    installer.register_sources([pinned], site)
    result = subprocess.check_output(
        [
            sys.executable,
            "-c",
            f"import sys, site; sys.path.insert(0, {str(old)!r}); "
            f"site.addsitedir({str(site)!r}); import ontic_test_backbone; "
            "print(ontic_test_backbone.VERSION)",
        ],
        text=True,
    )
    assert result.strip() == "pinned"


def test_missing_moge_source_is_failure_even_without_cuda(monkeypatch):
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError, match="moge"):
        installer.check_import("moge3", cuda=False)


def test_present_moge_sources_do_not_claim_cpu_import_success(monkeypatch, capsys):
    monkeypatch.setattr(installer.importlib.util, "find_spec", lambda name: object())
    assert installer.check_import("moge3", cuda=False) is False
    assert "SKIP moge3 import" in capsys.readouterr().out


def test_import_failure_makes_check_fail(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def check_import(name, cuda):
        if name == "da3":
            raise ModuleNotFoundError("No module named 'evo'")
        return True

    monkeypatch.setattr(installer, "check_import", check_import)
    assert installer.main(["--check"]) == 1
    assert "No module named 'evo'" in capsys.readouterr().out
