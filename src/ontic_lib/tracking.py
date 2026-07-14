from __future__ import annotations

import json
import os
import time
from pathlib import Path


class Tracker:
    """Appends one JSON line per log() call to output/metrics.jsonl — the durable,
    plain-text metrics record that ships to B2 with the job — and mirrors to W&B
    when configured (best-effort, never fatal). Flushed per line so the bootstrap's
    incremental sync sees fresh data; fsync coalesced to >=1 s. After a hard kill
    the final line may be truncated — read_metrics() skips it; any byte prefix of
    the file is a valid record set."""

    _FSYNC_EVERY_S = 1.0

    def __init__(self, fh, wandb_run):
        self._fh = fh
        self._wandb = wandb_run
        self._last_fsync = 0.0

    def log(self, metrics: dict, step: int | None = None) -> None:
        rec = {"step": step, "ts": time.time(), **metrics}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        now = time.monotonic()
        if now - self._last_fsync >= self._FSYNC_EVERY_S:
            os.fsync(self._fh.fileno())
            self._last_fsync = now
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
        finally:
            self._fh.close()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


def read_metrics(path) -> list[dict]:
    """All decodable records from a metrics.jsonl. A truncated final line (hard-kill
    mid-append, possibly mid multi-byte character) is expected and skipped, as are
    blank lines."""
    records: list[dict] = []
    for raw in Path(path).read_bytes().splitlines():
        if not raw.strip():
            continue
        try:
            records.append(json.loads(raw))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return records


def _maybe_wandb(project: str, config: dict | None):
    run_id = os.environ.get("ONTIC_WANDB_RUN_ID")
    if not run_id:
        return None
    if not (os.environ.get("WANDB_API_KEY")
            or (Path.home() / ".netrc").is_file()):
        return None
    try:
        import wandb
        return wandb.init(project=project, id=run_id, resume="allow",
                          config=config, settings={"silent": True})
    except Exception:
        return None


def init(project: str, config: dict | None = None) -> Tracker:
    out = Path.cwd() / "output"
    out.mkdir(parents=True, exist_ok=True)
    fh = (out / "metrics.jsonl").open("a", encoding="utf-8")
    fh.write(json.dumps({"_type": "config", "project": project,
                         "config": config or {}, "ts": time.time()}) + "\n")
    fh.flush()
    return Tracker(fh, _maybe_wandb(project, config))
