import json
import sys
import types

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
