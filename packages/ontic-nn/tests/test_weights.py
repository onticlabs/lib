"""The ontic store serves pinned weights before the hubs, and steps aside when it cannot."""

import json
from pathlib import Path

import pytest

from ontic_nn import weights
from ontic_nn.weights import OnticWeights, ontic_cache_root, ontic_path
from ontic_nn.wrappers.common import resolve_checkpoint

JOB = "00000000-0000-4000-8000-000000000000"


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A fake cache root and one table entry; returns the artifact directory."""
    monkeypatch.setenv("ONTIC_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setitem(weights.ONTIC_WEIGHTS, "org/repo", OnticWeights(JOB))
    art = tmp_path / "ontic" / "jobs" / JOB / "output" / "data"
    return art


def test_cache_root_follows_the_cli_precedence(tmp_path):
    assert ontic_cache_root({"ONTIC_CACHE_DIR": "/c"}, tmp_path) == Path("/c/ontic")
    assert ontic_cache_root({"XDG_CACHE_HOME": "/x"}, tmp_path) == Path("/x/ontic")
    assert ontic_cache_root({}, tmp_path) == tmp_path / ".cache" / "ontic"
    (tmp_path / ".ontic" / "cache").mkdir(parents=True)
    assert ontic_cache_root({}, tmp_path) == tmp_path / ".ontic" / "cache"


def test_cached_artifact_is_served_without_pulling(store, monkeypatch):
    store.mkdir(parents=True)
    (store / "model.pt").write_bytes(b"x")
    monkeypatch.setattr(weights, "_pull", lambda entry: pytest.fail("pulled"))
    assert ontic_path("org/repo") == store
    assert ontic_path("org/repo", "model.pt") == store / "model.pt"
    assert ontic_path("org/repo", "other.pt", allow_download=False) is None
    assert ontic_path("unknown/repo") is None


def test_miss_pulls_once_then_serves(store, monkeypatch):
    calls = []

    def pull(entry):
        calls.append(entry)
        store.mkdir(parents=True)
        (store / "model.pt").write_bytes(b"x")
        return True

    monkeypatch.setattr(weights, "_pull", pull)
    assert ontic_path("org/repo", "model.pt") == store / "model.pt"
    assert calls == [OnticWeights(JOB)]
    assert ontic_path("org/repo", "model.pt") == store / "model.pt"
    assert len(calls) == 1


def test_miss_respects_offline_modes(store, monkeypatch):
    monkeypatch.setattr(weights, "_pull", lambda entry: pytest.fail("pulled"))
    assert ontic_path("org/repo", allow_download=False) is None
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert ontic_path("org/repo") is None


def test_failed_pull_warns_and_falls_back(store, monkeypatch):
    class Proc:
        returncode = 1
        stdout = ""
        stderr = "error: no ontic.toml found\n"

    monkeypatch.setattr(weights.shutil, "which", lambda name: "/usr/bin/ontic")
    monkeypatch.setattr(weights.subprocess, "run", lambda *a, **kw: Proc())
    with pytest.warns(UserWarning, match="no ontic.toml"):
        assert ontic_path("org/repo") is None
    monkeypatch.setattr(weights.shutil, "which", lambda name: None)
    assert ontic_path("org/repo") is None


def test_pull_runs_the_cli_in_the_experiments_repo(store, monkeypatch):
    seen = {}

    def run(cmd, **kw):
        seen.update(cmd=cmd, cwd=kw["cwd"])
        store.mkdir(parents=True)
        (store / "model.pt").write_bytes(b"x")

        class Proc:
            returncode = 0
            stdout = json.dumps({"job": JOB})
            stderr = ""

        return Proc()

    monkeypatch.setattr(weights.shutil, "which", lambda name: "/usr/bin/ontic")
    monkeypatch.setattr(weights.subprocess, "run", run)
    monkeypatch.setenv("ONTIC_REPO", "/experiments")
    assert ontic_path("org/repo", "model.pt") == store / "model.pt"
    assert seen["cmd"] == ["/usr/bin/ontic", "pull", JOB, "--path", "data", "--json"]
    assert seen["cwd"] == "/experiments"


def test_resolve_checkpoint_prefers_the_store_over_the_hub(store, monkeypatch):
    store.mkdir(parents=True)
    (store / "model.pt").write_bytes(b"x")
    monkeypatch.setattr(weights, "_pull", lambda entry: pytest.fail("pulled"))
    explicit = store.parent / "explicit.pt"
    explicit.write_bytes(b"y")
    assert resolve_checkpoint(str(explicit), "org/repo") == str(explicit)
    assert resolve_checkpoint(None, "org/repo") == str(store)
    assert resolve_checkpoint(None, "org/repo", filename="model.pt") == str(store / "model.pt")


def test_table_matches_the_installer_manifest():
    manifest = Path(__file__).resolve().parents[3] / "scripts" / "model_weights.json"
    jobs = {}
    for model in json.loads(manifest.read_text()):
        if "ontic_job" not in model:
            continue  # not in the store yet: the hub serves it
        for repo in (model, model.get("base_model", {})):
            if "repo_id" in repo:
                jobs[repo["repo_id"]] = model["ontic_job"]
        if "repo_id" not in model:
            jobs[model["torch_assets"][0]["url"]] = model["ontic_job"]
    assert {k: v.job for k, v in weights.ONTIC_WEIGHTS.items()} == jobs
    assert len(jobs) >= 14
