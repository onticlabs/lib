"""Unified setup: source discovery, isolated checks, and explicit CUDA build behavior."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# The command is normally run by filename, which puts scripts on sys.path.
sys.path.insert(0, str(SCRIPTS))
try:
    spec = importlib.util.spec_from_file_location("model_installer", SCRIPTS / "install_models.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
finally:
    sys.path.remove(str(SCRIPTS))


def test_tracker_pins_match_adapter_apis():
    from ontic_nn.trackers.cotracker3 import COTRACKER3_REVISION
    from ontic_nn.trackers.mvtracker import MVTRACKER_REVISION
    from ontic_nn.trackers.tapip3d import TAPIP3D_REVISION
    from ontic_nn.trackers.trackcraft3r import TRACKCRAFT3R_REVISION

    sources = json.loads(installer.TRACKER_MANIFEST.read_text())
    assert {s["name"]: s["revision"] for s in sources} == {
        "mvtracker": MVTRACKER_REVISION,
        "tapip3d": TAPIP3D_REVISION,
        "cotracker3": COTRACKER3_REVISION,
        "trackcraft3r": TRACKCRAFT3R_REVISION,
    }


def test_registration_keeps_generic_names_off_global_path_and_adapters_find_sources(
    tmp_path, monkeypatch
):
    from ontic_nn.trackers import sources
    from ontic_nn.trackers.tapip3d import TAPIP3DConfig, _resolve_repo_path

    cache = tmp_path / "share/ontic-trackers"
    paths = {name: cache / name for name in installer.TRACKER_MODULES}
    for path in paths.values():
        path.mkdir(parents=True)
    marker = paths["tapip3d"] / "models/point_tracker_3d.py"
    marker.parent.mkdir()
    marker.touch()
    site = tmp_path / "site-packages"
    site.mkdir()
    installer.register_trackers(paths, site, cache)
    pth = (site / "ontic_trackers.pth").read_text()
    assert str(paths["mvtracker"]) in pth and str(paths["cotracker3"]) in pth
    assert str(paths["tapip3d"]) not in pth and str(paths["trackcraft3r"]) not in pth
    monkeypatch.setattr(sources.sys, "prefix", str(tmp_path))
    monkeypatch.delenv("ONTIC_TAPIP3D_REPO", raising=False)
    assert _resolve_repo_path(TAPIP3DConfig()) == paths["tapip3d"]
    assert sources.installed_repo("trackcraft3r") == paths["trackcraft3r"]
    paths["trackcraft3r"].rmdir()
    with pytest.raises(FileNotFoundError, match="Rerun scripts/install_models"):
        sources.installed_repo("trackcraft3r")
    # An explicit checkout still wins over the now-broken registration.
    monkeypatch.setenv("ONTIC_TAPIP3D_REPO", str(paths["tapip3d"]))
    assert _resolve_repo_path(TAPIP3DConfig()) == paths["tapip3d"]


def test_cuda_preflight_rejects_mismatched_toolkit(tmp_path, monkeypatch):
    nvcc = tmp_path / "bin/nvcc"
    nvcc.parent.mkdir()
    nvcc.touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(
        installer.subprocess, "check_output", lambda *a, **k: "release 12.4, V12.4.131"
    )
    with pytest.raises(RuntimeError, match="CUDA major version"):
        installer.cuda_build_environment()


@pytest.mark.parametrize("explicit", [False, True])
def test_trackcraft_uses_registered_source_unless_overridden(tmp_path, monkeypatch, explicit):
    from ontic_nn.trackers import trackcraft3r

    repo = tmp_path / ("explicit" if explicit else "installed")
    marker = repo / "evaluation/wan_scene_flow_predictor.py"
    marker.parent.mkdir(parents=True)
    marker.touch()
    for name in ("evaluation", "diffsynth"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    # The loader intentionally retains the source path; restore it after this test.
    monkeypatch.setattr(sys, "path", list(sys.path))

    def installed(name):
        assert not explicit, "An explicit checkout must bypass installed defaults"
        assert name == "trackcraft3r"
        return repo

    def import_predictor(*args, **kwargs):
        assert str(repo) in sys.path
        return SimpleNamespace(__file__=str(marker))

    def stop_before_weights(*args, **kwargs):
        assert kwargs["allow_download"] is False
        raise FileNotFoundError("test checkpoint absent")

    monkeypatch.setattr(trackcraft3r, "installed_repo", installed)
    monkeypatch.setattr(trackcraft3r, "import_research_module", import_predictor)
    monkeypatch.setattr(trackcraft3r, "resolve_checkpoint", stop_before_weights)
    cfg = trackcraft3r.TrackCraft3RConfig(
        repo_path=str(repo) if explicit else None, allow_download=False
    )
    with pytest.raises(FileNotFoundError, match="test checkpoint absent"):
        trackcraft3r.load_predictor(cfg)


def test_cuda_preflight_supports_cross_compilation_only_with_explicit_architecture(
    tmp_path, monkeypatch
):
    nvcc = tmp_path / "bin/nvcc"
    nvcc.parent.mkdir()
    nvcc.touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path))
    monkeypatch.delenv("TORCH_CUDA_ARCH_LIST", raising=False)
    monkeypatch.setattr(torch.version, "cuda", "12.4")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(installer.shutil, "which", lambda name: "/usr/bin/c++")
    monkeypatch.setattr(
        installer.subprocess, "check_output", lambda *a, **k: "release 12.4, V12.4.131"
    )
    with pytest.raises(RuntimeError, match="TORCH_CUDA_ARCH_LIST"):
        installer.cuda_build_environment()
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6")
    env = installer.cuda_build_environment()
    assert env["CUDA_HOME"] == str(tmp_path) and env["TORCH_CUDA_ARCH_LIST"] == "8.6"


def test_extension_build_cannot_resolve_or_replace_torch(tmp_path, monkeypatch):
    source = tmp_path / "third_party/pointops2"
    source.mkdir(parents=True)
    (source / "setup.py").touch()
    calls = []
    monkeypatch.setattr(installer.backbones, "run", lambda cmd, **kw: calls.append((cmd, kw)))
    installer.build_pointops("uv", tmp_path, {"CUDA_HOME": "/cuda"})
    cmd, kwargs = calls[0]
    assert "--no-deps" in cmd and "--no-build-isolation" in cmd
    assert cmd[cmd.index("--python") + 1] == sys.executable
    assert kwargs["cwd"] == source and kwargs["env"]["CUDA_HOME"] == "/cuda"


def test_checks_use_separate_processes_and_report_failures(monkeypatch, capsys):
    calls = []

    def run(cmd, **kw):
        calls.append((cmd, kw))
        return SimpleNamespace(returncode=int(cmd[-1] == "cotracker3"))

    monkeypatch.setattr(installer.subprocess, "run", run)
    assert installer.check_installation("trackers", skip_cuda=True) == 1
    assert {cmd[-1] for cmd, _ in calls} == {"mvtracker", "cotracker3", "trackcraft3r"}
    assert all("--check-one" in cmd and kw["env"]["HF_HUB_OFFLINE"] == "1" for cmd, kw in calls)
    captured = capsys.readouterr()
    assert "SKIP tapip3d" in captured.out and "cotracker3" in captured.err


def test_dry_run_resolves_all_extras_without_fetching_or_building(monkeypatch):
    calls = []
    monkeypatch.setattr(installer.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(
        installer.backbones,
        "stack_versions",
        lambda: dict(torch="2.4.1", torchvision="0.19.1", numpy="1.26.4"),
    )
    monkeypatch.setattr(
        installer.backbones, "install_dependencies", lambda *a, **kw: calls.append(kw)
    )
    monkeypatch.setattr(
        installer.backbones, "checkout", lambda *a: pytest.fail("dry run fetched sources")
    )
    monkeypatch.setattr(
        installer, "cuda_build_environment", lambda: pytest.fail("dry run requires CUDA")
    )
    assert installer.main(["--dry-run"]) == 0
    assert calls == [
        dict(
            extras=("backbones", "trackers"),
            data_extras=("all",),
            viz_extras=("robot", "rerun"),
            dry_run=True,
        )
    ]
