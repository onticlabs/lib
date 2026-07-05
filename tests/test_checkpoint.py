from __future__ import annotations

import json
import pickle

from ontic_lib.checkpoint import CheckpointManager


def test_save_resume_round_trips_state_exactly(tmp_path):
    mgr = CheckpointManager(tmp_path)
    state = {"model": [1, 2, 3], "opt": {"lr": 0.1}, "nested": {"a": [1, {"b": 2}]}}
    mgr.save(1, state)

    result = mgr.resume()

    assert result is not None
    resumed_state, _meta = result
    assert resumed_state == state


def test_meta_carries_step_run_id_and_saved_at(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(5, {"x": 1}, run_id="run-abc")

    _state, meta = mgr.resume()

    assert meta["step"] == 5
    assert meta["run_id"] == "run-abc"
    assert isinstance(meta["saved_at"], str)
    assert "T" in meta["saved_at"]  # ISO-8601


def test_resume_is_none_on_empty_dir(tmp_path):
    mgr = CheckpointManager(tmp_path)

    assert mgr.resume() is None


def test_latest_json_points_at_newest_after_multiple_saves(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(1, {"x": 1})
    mgr.save(2, {"x": 2})
    mgr.save(3, {"x": 3})

    state, meta = mgr.resume()

    assert state == {"x": 3}
    assert meta["step"] == 3

    latest = json.loads((tmp_path / "latest.json").read_text())
    assert latest["meta"]["step"] == 3


def test_keep_last_prunes_old_checkpoints_but_keeps_latest(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(1, {"x": 1}, keep_last=2)
    mgr.save(2, {"x": 2}, keep_last=2)
    mgr.save(3, {"x": 3}, keep_last=2)

    step_files = sorted(p.name for p in tmp_path.glob("step_*.ckpt"))
    assert step_files == ["step_00000002.ckpt", "step_00000003.ckpt"]
    assert (tmp_path / "latest.json").exists()

    state, meta = mgr.resume()
    assert state == {"x": 3}
    assert meta["step"] == 3


def test_keep_last_never_deletes_the_checkpoint_just_written(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(1, {"x": 1}, keep_last=1)
    mgr.save(2, {"x": 2}, keep_last=1)

    step_files = sorted(p.name for p in tmp_path.glob("step_*.ckpt"))
    assert step_files == ["step_00000002.ckpt"]


def test_atomic_write_leaves_no_tmp_files_behind(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(1, {"x": 1})
    mgr.save(2, {"x": 2}, keep_last=1)

    tmp_leftovers = list(tmp_path.glob("*.tmp*"))
    assert tmp_leftovers == []


def test_custom_save_load_fn_is_honored(tmp_path):
    def save_json(obj, path):
        path.write_text(json.dumps(obj))

    def load_json(path):
        return json.loads(path.read_text())

    mgr = CheckpointManager(tmp_path, save_fn=save_json, load_fn=load_json)
    mgr.save(1, {"x": 1})

    state, meta = mgr.resume()

    assert state == {"x": 1}
    assert meta["step"] == 1
    # confirm it really used json, not pickle
    raw = (tmp_path / "step_00000001.ckpt").read_text()
    json.loads(raw)  # must not raise


def test_default_save_fn_uses_pickle(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(1, {"x": 1})

    raw = (tmp_path / "step_00000001.ckpt").read_bytes()
    obj = pickle.loads(raw)
    assert obj["state"] == {"x": 1}
    assert obj["meta"]["step"] == 1


def test_latest_step_reflects_newest_step(tmp_path):
    mgr = CheckpointManager(tmp_path)

    assert mgr.latest_step is None

    mgr.save(1, {"x": 1})
    assert mgr.latest_step == 1

    mgr.save(7, {"x": 2})
    assert mgr.latest_step == 7


def test_directory_created_if_missing(tmp_path):
    target = tmp_path / "nested" / "checkpoints"
    mgr = CheckpointManager(target)

    mgr.save(1, {"x": 1})

    assert target.is_dir()
    assert mgr.resume()[0] == {"x": 1}


def test_resume_falls_back_to_highest_step_file_if_latest_json_missing(tmp_path):
    mgr = CheckpointManager(tmp_path)
    mgr.save(1, {"x": 1})
    mgr.save(2, {"x": 2})
    (tmp_path / "latest.json").unlink()

    state, meta = mgr.resume()

    assert state == {"x": 2}
    assert meta["step"] == 2


def test_save_returns_checkpoint_path(tmp_path):
    mgr = CheckpointManager(tmp_path)

    path = mgr.save(3, {"x": 1})

    assert path == tmp_path / "step_00000003.ckpt"
    assert path.is_file()
