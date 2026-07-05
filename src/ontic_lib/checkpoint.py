from __future__ import annotations

import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path

_STEP_GLOB = "step_*.ckpt"


def _step_filename(step: int) -> str:
    return f"step_{step:08d}.ckpt"


def _pickle_save(obj, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _pickle_load(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


class CheckpointManager:
    def __init__(self, directory, *, save_fn=None, load_fn=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._save_fn = save_fn or _pickle_save
        self._load_fn = load_fn or _pickle_load

    def _latest_json_path(self) -> Path:
        return self.directory / "latest.json"

    def save(
        self, step: int, state: dict, *, run_id: str | None = None, keep_last: int | None = None
    ) -> Path:
        meta = {
            "step": step,
            "run_id": run_id,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        payload = {"meta": meta, "state": state}

        filename = _step_filename(step)
        final_path = self.directory / filename
        tmp_path = self.directory / f"{filename}.tmp{os.getpid()}"
        self._save_fn(payload, tmp_path)
        os.replace(tmp_path, final_path)

        self._latest_json_path().write_text(
            json.dumps({"filename": filename, "meta": meta})
        )

        if keep_last is not None:
            self._prune(keep_last)

        return final_path

    def _prune(self, keep_last: int) -> None:
        step_files = self._all_step_files()
        if len(step_files) <= keep_last:
            return
        # newest by step first; drop the rest
        to_delete = step_files[keep_last:]
        for _step, path in to_delete:
            path.unlink()

    def _all_step_files(self) -> list[tuple[int, Path]]:
        entries = []
        for path in self.directory.glob(_STEP_GLOB):
            try:
                step = int(path.stem.split("_", 1)[1])
            except (IndexError, ValueError):
                continue
            entries.append((step, path))
        entries.sort(key=lambda e: e[0], reverse=True)
        return entries

    def resume(self) -> tuple[dict, dict] | None:
        latest_json = self._latest_json_path()
        target_path = None

        if latest_json.is_file():
            pointer = json.loads(latest_json.read_text())
            candidate = self.directory / pointer["filename"]
            if candidate.is_file():
                target_path = candidate

        if target_path is None:
            step_files = self._all_step_files()
            if not step_files:
                return None
            target_path = step_files[0][1]

        payload = self._load_fn(target_path)
        return payload["state"], payload["meta"]

    @property
    def latest_step(self) -> int | None:
        result = self.resume()
        if result is None:
            return None
        return result[1]["step"]
