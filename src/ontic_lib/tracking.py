from __future__ import annotations

import os
from pathlib import Path


class Tracker:
    def __init__(self, trackio_run, wandb_run):
        self._trackio = trackio_run
        self._wandb = wandb_run

    def log(self, metrics: dict, step: int | None = None) -> None:
        self._trackio.log(metrics, step=step)
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass

    def finish(self) -> None:
        self._trackio.finish()
        if self._wandb is not None:
            try:
                self._wandb.finish()
            except Exception:
                pass


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
    os.environ.setdefault("TRACKIO_DIR", str(Path.cwd() / "output"))
    Path(os.environ["TRACKIO_DIR"]).mkdir(parents=True, exist_ok=True)
    import trackio
    run = trackio.init(project=project, config=config or {})
    return Tracker(run, _maybe_wandb(project, config))
