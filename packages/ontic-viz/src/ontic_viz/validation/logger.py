"""Sinks for validation media: discard, write to local files, or send to Olympus.

Images are ``(3, height, width)`` floats in [0, 1]; videos are lists of them; point clouds
are PLY-style structured arrays (``x/y/z/red/green/blue``).
"""

from __future__ import annotations

import importlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import pack
from torch import Tensor

from ontic_viz.validation.image_io import fig_to_image, save_image


class VizLogger(ABC):
    """Abstract base for visualization loggers.

    Histograms and figures are no-ops by default so sinks that cannot show them ignore them.
    """

    @abstractmethod
    def log_image(self, key: str, image: Tensor, step: int, caption: str | None = None) -> None: ...

    @abstractmethod
    def log_video(
        self,
        key: str,
        images: list[Tensor],
        step: int,
        fps: int = 30,
        caption: str | None = None,
        loop_reverse: bool = False,
    ) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: dict, step: int) -> None: ...

    @abstractmethod
    def log_point_cloud(self, key: str, vertex_data: np.ndarray, step: int) -> None: ...

    def log_histogram(self, key: str, values: Any, step: int) -> None:
        """Log a 1-D distribution (array-like). No-op unless the sink can show histograms."""

    def log_figure(self, key: str, fig: Any, step: int) -> None:
        """Log a matplotlib figure. No-op unless the sink can show figures."""

    def commit(self, step: int) -> None:
        """Send everything buffered for this step. No-op for non-buffering loggers."""

    def flush(self) -> None:
        """Send anything still buffered, whatever step it belongs to."""


class NoOpVizLogger(VizLogger):
    """No-op logger for non-rank-0 processes in DDP."""

    def log_image(self, key, image, step, caption=None):
        pass

    def log_video(self, key, images, step, fps=30, caption=None, loop_reverse=False):
        pass

    def log_metrics(self, metrics, step):
        pass

    def log_point_cloud(self, key, vertex_data, step):
        pass


def _video_array(images: list[Tensor], loop_reverse: bool) -> np.ndarray:
    """Stack frames into uint8 ``(time, channel, height, width)``; optionally play back."""
    video = torch.stack(images)
    video = (video.clamp(0, 1) * 255).type(torch.uint8).cpu().numpy()
    if loop_reverse:
        video = pack([video, video[::-1][1:-1]], "* c h w")[0]
    return video


def _tracking():
    """``ontic_lib.tracking``, imported on first use: Olympus and its media adapters stay
    optional for everything else in this package."""
    return importlib.import_module("ontic_lib.tracking")


class OlympusVizLogger(VizLogger):
    """Logs visualizations to Olympus through ``ontic_lib.tracking``'s media adapters.

    ``olympus.log()`` has no ``commit`` flag: every call appends its own row, so a step logged
    in several pieces (train metrics, then validation, then media) would land as several
    partial rows. This logger buffers a step's keys and emits exactly one
    ``tracking.run_log()`` per step: on ``commit()``, on ``flush()``, or automatically when
    a later step arrives.
    """

    def __init__(self) -> None:
        self._pending: dict = {}
        self._pending_step: int | None = None

    def _stage(self, metrics: dict, step: int) -> None:
        if self._pending_step is not None and step != self._pending_step:
            self.flush()
        self._pending_step = step
        self._pending.update(metrics)

    def log_image(self, key, image, step, caption=None):
        self._stage({key: _tracking().image(image, caption=caption)}, step)

    def log_video(self, key, images, step, fps=30, caption=None, loop_reverse=False):
        video = _video_array(images, loop_reverse)
        key = key if "/" in key else f"video/{key}"
        self._stage({key: _tracking().video(video[None], fps=fps, caption=caption)}, step)

    def log_metrics(self, metrics, step):
        self._stage(dict(metrics), step)

    def log_point_cloud(self, key, vertex_data, step):
        self._stage({key: _tracking().point_cloud(vertex_data)}, step)

    def log_histogram(self, key, values, step):
        self._stage({key: _tracking().histogram(values)}, step)

    def log_figure(self, key, fig, step):
        self._stage({key: _tracking().figure(fig)}, step)

    def commit(self, step: int) -> None:
        self._stage({"trainer/global_step": step}, step)
        self.flush()

    def flush(self) -> None:
        if self._pending_step is None:
            return
        pending, step = self._pending, self._pending_step
        self._pending, self._pending_step = {}, None
        _tracking().run_log(pending, step=step)


class LocalVizLogger(VizLogger):
    """Saves visualizations under ``log_dir``: PNG images (and figures), MP4 videos
    (``ontic-viz[video]``) and PLY point clouds (``ontic-viz[ply]``). Scalars go to
    ``metrics_file`` when one is given, one JSON object per ``log_metrics`` call in the
    ``ontic_lib.tracking`` layout (``{"step", "ts", <metrics>}``; a tracker mirror can
    replay the file into Olympus from a machine with network access); histograms are
    dropped."""

    def __init__(self, log_dir: Path | str, metrics_file: Path | str | None = None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_fh = None
        if metrics_file is not None:
            Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
            self._metrics_fh = open(metrics_file, "a", encoding="utf-8")

    def log_image(self, key, image, step, caption=None):
        image = image.clamp(0, 1)
        caption = caption if caption is not None else ""
        if f"{step:0>6}" not in caption:
            caption = f"step_{step:0>6}_{caption}"
        path = self.log_dir / f"{key}_{caption}.png"
        path = Path(str(path).replace(" ", "_").replace(":", "-").replace(",", "_"))
        path.parent.mkdir(parents=True, exist_ok=True)
        save_image(image, path)

    def log_video(self, key, images, step, fps=30, caption=None, loop_reverse=False):
        try:
            import moviepy.editor as mpy
        except ImportError as e:
            raise ImportError("MP4 videos need moviepy; install ontic-viz[video]") from e
        video = _video_array(images, loop_reverse)
        key = key if "/" in key else f"video/{key}"
        # Convert (T, C, H, W) -> list of (H, W, C) for moviepy
        frames = [video[i].transpose(1, 2, 0) for i in range(len(video))]
        clip = mpy.ImageSequenceClip(frames, fps=fps)
        caption_str = f"_{caption}" if caption is not None else ""
        save_path = self.log_dir / key / f"step_{step:0>6}_{key.split('/')[-1]}{caption_str}.mp4"
        save_path = Path(str(save_path).replace(" ", "_").replace(":", "-").replace(",", "_"))
        save_path.parent.mkdir(exist_ok=True, parents=True)
        clip.write_videofile(str(save_path), logger=None)

    def log_metrics(self, metrics, step):
        if self._metrics_fh is None:
            return
        rec: dict = {"step": int(step), "ts": time.time()}
        for k, v in metrics.items():
            if isinstance(v, bool):
                rec[k] = float(v)
            elif isinstance(v, (int, float)):
                rec[k] = v
            elif hasattr(v, "numel") and v.numel() == 1:
                rec[k] = float(v)
        self._metrics_fh.write(json.dumps(rec) + "\n")
        self._metrics_fh.flush()

    def log_point_cloud(self, key, vertex_data, step):
        try:
            from plyfile import PlyData, PlyElement
        except ImportError as e:
            raise ImportError("PLY point clouds need plyfile; install ontic-viz[ply]") from e
        path = self.log_dir / f"{key}_step_{step:0>6}.ply"
        path.parent.mkdir(parents=True, exist_ok=True)
        el = PlyElement.describe(vertex_data, "vertex")
        PlyData([el]).write(str(path))

    def log_figure(self, key, fig, step):
        self.log_image(key, fig_to_image(fig), step)
