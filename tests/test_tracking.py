import json
import sys
import types

import pytest

from ontic_lib import tracking


def test_jsonl_is_the_record(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONTIC_WANDB_RUN_ID", raising=False)
    t = tracking.init("saliency", {"lr": 0.1})
    t.log({"loss": 1.0}, step=1)
    t.log({"loss": 0.5, "acc": 0.9}, step=2)
    t.finish()
    lines = [json.loads(x) for x in
             (tmp_path / "output" / "metrics.jsonl").read_text().splitlines()]
    assert lines[0]["_type"] == "config"
    assert lines[0]["project"] == "saliency" and lines[0]["config"] == {"lr": 0.1}
    assert lines[1]["step"] == 1 and lines[1]["loss"] == 1.0 and "ts" in lines[1]
    assert lines[2] == {**lines[2], "step": 2, "loss": 0.5, "acc": 0.9}


def test_append_on_resume_not_truncate(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONTIC_WANDB_RUN_ID", raising=False)
    t = tracking.init("p")
    t.log({"a": 1}, step=1)
    t.finish()
    t2 = tracking.init("p")
    t2.log({"a": 2}, step=2)
    t2.finish()
    lines = (tmp_path / "output" / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 4  # config, log, config, log


def test_read_metrics_skips_truncated_tail(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text('{"_type": "config", "project": "p", "config": {}, "ts": 1}\n'
                 '{"step": 1, "ts": 2, "loss": 0.5}\n'
                 '{"step": 2, "ts": 3, "lo')  # hard-kill truncation
    recs = tracking.read_metrics(p)
    assert len(recs) == 2
    assert recs[1]["loss"] == 0.5


def test_read_metrics_skips_mid_multibyte_truncation(tmp_path):
    p = tmp_path / "metrics.jsonl"
    good = '{"step": 1, "ts": 2, "note": "café"}\n'.encode()
    truncated = '{"step": 2, "ts": 3, "note": "café"}'.encode()[:-3]  # cut inside "é"
    p.write_bytes(good + truncated)
    recs = tracking.read_metrics(p)
    assert len(recs) == 1 and recs[0]["step"] == 1


def test_wandb_optional_and_never_fatal(monkeypatch, tmp_path):
    bad = types.ModuleType("wandb")
    bad.init = lambda **kw: (_ for _ in ()).throw(RuntimeError("down"))
    monkeypatch.setitem(sys.modules, "wandb", bad)
    monkeypatch.setenv("WANDB_API_KEY", "k")
    monkeypatch.setenv("ONTIC_WANDB_RUN_ID", "abc")
    monkeypatch.chdir(tmp_path)
    t = tracking.init("p")                    # must not raise
    t.log({"x": 1})
    t.finish()
    assert (tmp_path / "output" / "metrics.jsonl").is_file()


def test_flush_per_line_visible_before_finish(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONTIC_WANDB_RUN_ID", raising=False)
    t = tracking.init("p")
    t.log({"x": 1}, step=1)
    # BEFORE finish(): another process (the bootstrap sync thread) must see the line
    text = (tmp_path / "output" / "metrics.jsonl").read_text()
    assert '"x": 1' in text
    t.finish()


# --- Olympus mirror ---------------------------------------------------------


class _FakeOlympusRun:
    def __init__(self):
        self.logged = []
        self.system = []
        self.statuses = []
        self.finish_calls = 0

    def log(self, metrics, step=None):
        self.logged.append((metrics, step))

    def log_system(self, metrics):
        self.system.append(metrics)

    def finish(self):
        self.finish_calls += 1

    def _declare_status(self, status, send=False):
        self.statuses.append((status, send))


def _fake_olympus(monkeypatch, init_raises=False):
    mod = types.ModuleType("olympus")
    run = _FakeOlympusRun()
    mod.init_calls = []
    mod.saved = []

    def init(**kwargs):
        mod.init_calls.append(kwargs)
        if init_raises:
            raise RuntimeError("server down")
        return run

    mod.init = init
    mod.save = lambda path, project=None: mod.saved.append(path)
    monkeypatch.setitem(sys.modules, "olympus", mod)
    return mod, run


def _olympus_env(monkeypatch, **extra):
    monkeypatch.delenv("ONTIC_WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("OLYMPUS_SERVER_URL", raising=False)
    monkeypatch.setenv("ONTIC_LIB_NO_LOG_TAIL", "1")
    monkeypatch.setenv("ONTIC_OLYMPUS_PROJECT", "team-proj")
    monkeypatch.setenv("ONTIC_OLYMPUS_RUN", "exp-run-3")
    for key, value in extra.items():
        monkeypatch.setenv(key, value)


def test_olympus_inactive_without_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONTIC_WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("ONTIC_OLYMPUS_PROJECT", raising=False)
    monkeypatch.delenv("ONTIC_OLYMPUS_RUN", raising=False)
    mod, _ = _fake_olympus(monkeypatch)
    t = tracking.init("p")
    t.log({"x": 1})
    t.finish()
    assert mod.init_calls == []
    assert t._olympus is None


def test_olympus_config_assembled_from_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(
        monkeypatch,
        ONTIC_JOB_ID="j-42",
        ONTIC_EXPERIMENT="saliency",
        ONTIC_EXPERIMENT_SHA="abc123",
        ONTIC_ATTEMPT="1",
        ONTIC_GIT_PARENTS="p1 p2",
        ONTIC_GIT_SUBJECT="try wider net",
        ONTIC_GIT_BRANCH="exp/wider",
    )
    monkeypatch.delenv("ONTIC_RESUME", raising=False)
    monkeypatch.delenv("OLYMPUS_DISABLE_REMOTE_STOP", raising=False)
    mod, _ = _fake_olympus(monkeypatch)
    t = tracking.init("p", {"lr": 0.1})
    kwargs = mod.init_calls[0]
    assert kwargs["project"] == "team-proj" and kwargs["name"] == "exp-run-3"
    assert kwargs["resume"] == "never"
    assert kwargs["config"] == {
        "job_id": "j-42",
        "experiment": "saliency",
        "experiment_sha": "abc123",
        "attempt": "1",
        "git_parents": "p1 p2",
        "git_subject": "try wider net",
        "git_branch": "exp/wider",
        "lr": 0.1,
    }
    # our own poller replaces the SDK's, which must be off before init
    assert tracking.os.environ["OLYMPUS_DISABLE_REMOTE_STOP"] == "1"
    t.finish()


def test_olympus_omits_absent_env_keys(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch, ONTIC_JOB_ID="j-1")
    for env in ("ONTIC_EXPERIMENT", "ONTIC_EXPERIMENT_SHA", "ONTIC_ATTEMPT",
                "ONTIC_DESCRIPTION", "ONTIC_TAGS", "ONTIC_DEPS",
                "ONTIC_GIT_PARENTS", "ONTIC_GIT_SUBJECT", "ONTIC_GIT_BRANCH",
                "ONTIC_RESUME"):
        monkeypatch.delenv(env, raising=False)
    mod, _ = _fake_olympus(monkeypatch)
    t = tracking.init("p")
    assert mod.init_calls[0]["config"] == {"job_id": "j-1"}
    t.finish()


def test_olympus_config_carries_the_run_context_env(monkeypatch, tmp_path):
    """The importer-contract keys: description/tags/deps land verbatim, exactly as
    the launcher formatted them (tags ', '-joined, deps space-separated job ids)."""
    monkeypatch.chdir(tmp_path)
    _olympus_env(
        monkeypatch,
        ONTIC_JOB_ID="j-7",
        ONTIC_DESCRIPTION="Sweep the learning rate; wider net",
        ONTIC_TAGS="lr, sweep",
        ONTIC_DEPS="u1 u2 u3",
    )
    mod, _ = _fake_olympus(monkeypatch)
    t = tracking.init("p")
    cfg = mod.init_calls[0]["config"]
    assert cfg["description"] == "Sweep the learning rate; wider net"
    assert cfg["tags"] == "lr, sweep"
    assert cfg["deps"] == "u1 u2 u3"
    t.finish()


def test_olympus_experiment_config_wins_over_run_context_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(
        monkeypatch,
        ONTIC_DESCRIPTION="from the launcher",
        ONTIC_TAGS="lr",
        ONTIC_DEPS="u1",
    )
    mod, _ = _fake_olympus(monkeypatch)
    t = tracking.init("p", {"description": "the experiment's own", "deps": "mine"})
    cfg = mod.init_calls[0]["config"]
    assert cfg["description"] == "the experiment's own"
    assert cfg["deps"] == "mine"
    assert cfg["tags"] == "lr"  # untouched keys still come from the env
    t.finish()


@pytest.mark.parametrize("env", [{"ONTIC_RESUME": "1"}, {"ONTIC_ATTEMPT": "2"}])
def test_olympus_resumes_same_run(monkeypatch, tmp_path, env):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONTIC_RESUME", raising=False)
    _olympus_env(monkeypatch, **env)
    mod, _ = _fake_olympus(monkeypatch)
    t = tracking.init("p")
    assert mod.init_calls[0]["resume"] == "allow"
    t.finish()


def test_olympus_mirrors_log_and_finish(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch)
    _, run = _fake_olympus(monkeypatch)
    t = tracking.init("p")
    t.log({"loss": 1.0}, step=1)
    t.finish()
    assert run.logged == [({"loss": 1.0}, 1)]
    assert run.finish_calls == 1
    assert run.statuses == []  # clean finish never says failed


def test_metrics_json_holds_final_scalars(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ONTIC_WANDB_RUN_ID", raising=False)
    monkeypatch.delenv("ONTIC_OLYMPUS_PROJECT", raising=False)
    t = tracking.init("p")
    t.log({"loss": 1.0, "note": "warmup", "curve": [1, 2]}, step=1)
    t.log({"loss": 0.25, "acc": 0.9}, step=2)
    t.finish()
    summary = json.loads((tmp_path / "output" / "metrics.json").read_text())
    assert summary == {"loss": 0.25, "acc": 0.9, "note": "warmup"}


def test_exception_in_with_block_declares_failed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch)
    _, run = _fake_olympus(monkeypatch)
    with pytest.raises(RuntimeError):
        with tracking.init("p") as t:
            t.log({"x": 1})
            raise RuntimeError("boom")
    assert run.finish_calls == 1
    assert ("failed", True) in run.statuses
    # the durable record survived the crash path
    assert (tmp_path / "output" / "metrics.jsonl").is_file()
    assert (tmp_path / "output" / "metrics.json").is_file()


def _capture_error_posts(monkeypatch, run, answer=None):
    """Record set_run_error posts and failed declarations on one timeline."""
    events = []

    def post(url, payload, headers, timeout=4.0):
        events.append(("post", url, payload, headers, timeout))
        if isinstance(answer, Exception):
            raise answer
        return answer

    monkeypatch.setattr(tracking, "_post_json", post)
    orig = run._declare_status

    def declare(status, send=False):
        events.append(("status", status))
        orig(status, send)

    run._declare_status = declare
    return events


def test_with_block_failure_posts_traceback_before_failed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(
        monkeypatch,
        OLYMPUS_SERVER_URL="http://olympus.test",
        OLYMPUS_API_KEY="secret",
        OLYMPUS_WORKSPACE="ontic",
    )
    _, run = _fake_olympus(monkeypatch)
    events = _capture_error_posts(monkeypatch, run, answer={"data": "ok"})
    with pytest.raises(RuntimeError):
        with tracking.init("p") as t:
            t.log({"x": 1})
            raise RuntimeError("boom")
    posts = [e for e in events if e[0] == "post" and "set_run_error" in e[1]]
    assert len(posts) == 1
    _, url, payload, headers, timeout = posts[0]
    assert url == "http://olympus.test/api/set_run_error"
    assert payload["project"] == "team-proj" and payload["run"] == "exp-run-3"
    assert "Traceback (most recent call last)" in payload["text"]
    assert payload["text"].rstrip().endswith("RuntimeError: boom")
    assert headers["Authorization"] == "Bearer secret"
    assert headers["x-olympus-workspace"] == "ontic"
    assert timeout == 3.0
    # the traceback reaches the server before the failed declaration
    assert events.index(posts[0]) < events.index(("status", "failed"))
    assert ("failed", True) in run.statuses
    # the durable local record carries the same text
    assert (tmp_path / "output" / "error.txt").read_text() == payload["text"]


def test_excepthook_posts_traceback_once(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch, OLYMPUS_SERVER_URL="http://olympus.test")
    _, run = _fake_olympus(monkeypatch)
    events = _capture_error_posts(monkeypatch, run, answer={"data": "ok"})
    monkeypatch.setattr(sys, "excepthook", lambda *a: None)
    tracking.init("p")
    hook = sys.excepthook  # the mirror's installed hook
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        info = sys.exc_info()
    hook(*info)
    hook(*info)  # exit path and excepthook can both fire; only one send
    posts = [e for e in events if e[0] == "post" and "set_run_error" in e[1]]
    assert len(posts) == 1
    assert posts[0][2]["text"].rstrip().endswith("RuntimeError: boom")
    assert run.statuses.count(("failed", True)) == 1
    text = (tmp_path / "output" / "error.txt").read_text()
    assert text.count("RuntimeError: boom") == 1


def test_error_post_server_down_never_raises(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch, OLYMPUS_SERVER_URL="http://olympus.test")
    _, run = _fake_olympus(monkeypatch)
    _capture_error_posts(monkeypatch, run, answer=ConnectionError("down"))
    with pytest.raises(RuntimeError):  # only the run's own exception
        with tracking.init("p"):
            raise RuntimeError("boom")
    assert ("failed", True) in run.statuses  # still declared failed
    assert "set_run_error failed" in capsys.readouterr().err
    # error.txt is written even when the server is unreachable
    assert "RuntimeError: boom" in (tmp_path / "output" / "error.txt").read_text()


def test_error_text_truncated_to_last_64kb(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch, OLYMPUS_SERVER_URL="http://olympus.test")
    _, run = _fake_olympus(monkeypatch)
    events = _capture_error_posts(monkeypatch, run, answer={"data": "ok"})
    with pytest.raises(RuntimeError):
        with tracking.init("p"):
            raise RuntimeError("x" * 200_000 + " THE-END")
    payload = next(e[2] for e in events if e[0] == "post" and "set_run_error" in e[1])
    assert len(payload["text"].encode()) <= tracking._OLYMPUS_ERROR_MAX_BYTES
    assert payload["text"].rstrip().endswith("THE-END")  # the tail survives


def test_olympus_init_failure_never_fatal(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch)
    _fake_olympus(monkeypatch, init_raises=True)
    t = tracking.init("p")  # must not raise
    t.log({"x": 1})
    t.finish()
    assert t._olympus is None
    assert (tmp_path / "output" / "metrics.jsonl").is_file()


def test_olympus_import_missing_never_fatal(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch)
    monkeypatch.setitem(sys.modules, "olympus", None)  # forces ImportError
    t = tracking.init("p")  # must not raise
    t.log({"x": 1})
    t.finish()
    assert t._olympus is None
    assert (tmp_path / "output" / "metrics.jsonl").is_file()


def _mirror(monkeypatch, tmp_path, answer):
    _olympus_env(monkeypatch)
    run = _FakeOlympusRun()
    out = tmp_path / "output"
    out.mkdir(parents=True, exist_ok=True)
    mirror = tracking._OlympusMirror(run, out)
    mirror._server = "http://olympus.test"
    if isinstance(answer, Exception):
        def post(*a, **k):
            raise answer
    else:
        def post(*a, **k):
            return answer
    monkeypatch.setattr(tracking, "_post_json", post)
    return mirror, out


def test_stop_mode_writes_marker_before_interrupt(monkeypatch, tmp_path):
    mirror, out = _mirror(
        monkeypatch, tmp_path, {"data": {"stop": True, "mode": "stop"}}
    )
    marker = out / ".ontic" / "terminate-requested"
    seen = []
    monkeypatch.setattr(
        tracking._thread, "interrupt_main",
        lambda: seen.append(("interrupt", marker.is_file())),
    )
    assert mirror._poll_once() is True
    assert marker.is_file()
    assert seen == [("interrupt", True)]  # marker existed before the interrupt


def test_interrupt_mode_skips_marker(monkeypatch, tmp_path):
    mirror, out = _mirror(
        monkeypatch, tmp_path, {"data": {"stop": True, "mode": "interrupt"}}
    )
    seen = []
    monkeypatch.setattr(tracking._thread, "interrupt_main", lambda: seen.append("i"))
    assert mirror._poll_once() is True
    assert not (out / ".ontic" / "terminate-requested").exists()
    assert seen == ["i"]


def test_old_server_bare_true_means_stop(monkeypatch, tmp_path):
    mirror, out = _mirror(monkeypatch, tmp_path, {"data": True})
    monkeypatch.setattr(tracking._thread, "interrupt_main", lambda: None)
    assert mirror._poll_once() is True
    assert (out / ".ontic" / "terminate-requested").is_file()


def test_no_stop_and_network_errors_do_nothing(monkeypatch, tmp_path):
    mirror, out = _mirror(monkeypatch, tmp_path, {"data": {"stop": False}})
    seen = []
    monkeypatch.setattr(tracking._thread, "interrupt_main", lambda: seen.append("i"))
    assert mirror._poll_once() is False
    down, _ = _mirror(monkeypatch, tmp_path, ConnectionError("down"))
    assert down._poll_once() is False
    assert seen == []
    assert not (out / ".ontic" / "terminate-requested").exists()


def test_log_tail_feeds_system_logs(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch)
    monkeypatch.delenv("ONTIC_LIB_NO_LOG_TAIL", raising=False)
    _, run = _fake_olympus(monkeypatch)
    t = tracking.init("p")
    print("hello tail")
    t._olympus._flush_tail()
    assert any(entry.get("console") == "hello tail" for entry in run.system)
    t.finish()
    assert not isinstance(sys.stdout, tracking._TeeStream)  # restored


def test_log_tail_opt_out(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch)  # sets ONTIC_LIB_NO_LOG_TAIL=1
    _fake_olympus(monkeypatch)
    t = tracking.init("p")
    assert not isinstance(sys.stdout, tracking._TeeStream)
    t.finish()


def test_experiment_docs_uploaded_with_caps(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    _olympus_env(monkeypatch)
    (tmp_path / "README.md").write_text("about")
    (tmp_path / "RESULTS-01.md").write_text("numbers")
    (tmp_path / "PREREG-a.md").write_text("plan")
    (tmp_path / "RESULTS-huge.md").write_text("x" * 1_000_001)
    mod, _ = _fake_olympus(monkeypatch)
    t = tracking.init("p")
    t.finish()
    names = {tracking.Path(p).name for p in mod.saved}
    assert names == {"README.md", "RESULTS-01.md", "PREREG-a.md"}
