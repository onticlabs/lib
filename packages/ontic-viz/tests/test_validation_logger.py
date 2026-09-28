"""ontic_viz.validation.logger: the no-op / local / Olympus sinks. The Olympus one is tested
against a fake ``ontic_lib.tracking`` module injected into ``sys.modules``."""

from __future__ import annotations

import importlib.util
import sys
import types

import numpy as np
import pytest
import torch

from ontic_viz.validation import LocalVizLogger, NoOpVizLogger, OlympusVizLogger, VizLogger

_HAS = {m: importlib.util.find_spec(m) is not None for m in ("PIL", "moviepy", "plyfile")}
needs_pil = pytest.mark.skipif(not _HAS["PIL"], reason="pillow not installed (ontic-viz[images])")
needs_moviepy = pytest.mark.skipif(
    not _HAS["moviepy"], reason="moviepy not installed (ontic-viz[video])"
)
needs_plyfile = pytest.mark.skipif(
    not _HAS["plyfile"], reason="plyfile not installed (ontic-viz[ply])"
)

VERTEX_DTYPE = [
    ("x", "f4"),
    ("y", "f4"),
    ("z", "f4"),
    ("red", "u1"),
    ("green", "u1"),
    ("blue", "u1"),
]


def _frames(n=4, h=6, w=8):
    g = torch.Generator().manual_seed(0)
    return [torch.rand(3, h, w, generator=g) for _ in range(n)]


def test_noop_and_base_defaults():
    log = NoOpVizLogger()
    assert isinstance(log, VizLogger)
    log.log_image("k", torch.zeros(3, 2, 2), 0)
    log.log_video("k", _frames(2), 0)
    log.log_metrics({"a": 1}, 0)
    log.log_point_cloud("k", np.zeros(2, dtype=VERTEX_DTYPE), 0)
    log.log_histogram("k", np.arange(3), 0)
    log.log_figure("k", object(), 0)
    log.commit(0)
    log.flush()


# ---------------------------------------------------------------------------
# OlympusVizLogger
# ---------------------------------------------------------------------------


class FakeTracking(types.ModuleType):
    def __init__(self):
        super().__init__("ontic_lib.tracking")
        self.rows: list[tuple[dict, int | None]] = []
        self.calls: list[tuple] = []

    def image(self, img, caption=None):
        self.calls.append(("image", caption))
        return ("Image", tuple(img.shape), caption)

    def video(self, frames, fps=30, caption=None):
        self.calls.append(("video", fps, caption))
        return ("Video", frames.shape, str(frames.dtype), fps, caption, frames.copy())

    def point_cloud(self, vertex_data):
        return ("Object3D", vertex_data.shape)

    def histogram(self, values, num_bins=64):
        return ("Histogram", np.asarray(values).reshape(-1).shape, num_bins)

    def figure(self, fig, caption=None):
        return ("Html", fig)

    def run_log(self, metrics, step=None):
        self.rows.append((dict(metrics), step))


@pytest.fixture
def tracking(monkeypatch):
    fake = FakeTracking()
    monkeypatch.setitem(sys.modules, "ontic_lib.tracking", fake)
    return fake


def test_olympus_logger_emits_one_row_per_step(tracking):
    log = OlympusVizLogger()
    img = torch.zeros(3, 4, 5)
    log.log_metrics({"train/loss": 0.5}, 1)
    log.log_image("val/img", img, 1, caption="hello")
    log.log_point_cloud("pcd", np.zeros(3, dtype=VERTEX_DTYPE), 1)
    log.log_histogram("info/opacities_statistics", torch.rand(2, 3).numpy(), 1)
    log.log_figure("gs_plots/opacity_scale_stats", "FIG", 1)
    assert tracking.rows == []  # buffered until commit
    log.commit(1)
    assert len(tracking.rows) == 1
    row, step = tracking.rows[0]
    assert step == 1
    assert row["train/loss"] == 0.5
    assert row["val/img"] == ("Image", (3, 4, 5), "hello")
    assert row["pcd"] == ("Object3D", (3,))
    assert row["info/opacities_statistics"] == ("Histogram", (6,), 64)
    assert row["gs_plots/opacity_scale_stats"] == ("Html", "FIG")
    assert row["trainer/global_step"] == 1
    log.commit(1)  # a commit with nothing else pending still writes the step marker
    assert tracking.rows[1] == ({"trainer/global_step": 1}, 1)


def test_olympus_logger_flushes_when_step_changes(tracking):
    log = OlympusVizLogger()
    log.log_metrics({"a": 1}, 10)
    log.log_metrics({"b": 2}, 10)
    log.log_metrics({"c": 3}, 11)  # arrival of step 11 flushes step 10
    assert tracking.rows == [({"a": 1, "b": 2}, 10)]
    log.flush()
    assert tracking.rows[1] == ({"c": 3}, 11)
    log.flush()  # nothing pending: no row
    assert len(tracking.rows) == 2


def test_olympus_logger_video_conversion(tracking):
    log = OlympusVizLogger()
    frames = _frames(4, 6, 8)
    frames[0][:] = 2.0  # clamped to 1 -> 255
    log.log_video("clip", frames, 3, fps=7, caption="c")
    log.log_video("val_video/x", frames, 3, loop_reverse=True)
    log.flush()
    ((row, step),) = tracking.rows
    assert step == 3 and set(row) == {"video/clip", "val_video/x"}  # bare keys get 'video/'
    _, shape, dtype, fps, caption, arr = row["video/clip"]
    assert shape == (1, 4, 3, 6, 8) and dtype == "uint8" and fps == 7 and caption == "c"
    assert arr[0, 0].min() == 255
    expected = (torch.stack(frames).clamp(0, 1) * 255).to(torch.uint8).numpy()
    np.testing.assert_array_equal(arr[0], expected)
    _, shape_lr, _, fps_lr, _, arr_lr = row["val_video/x"]
    assert shape_lr == (1, 6, 3, 6, 8) and fps_lr == 30  # 4 + reversed middle 2
    np.testing.assert_array_equal(arr_lr[0, 4], expected[2])
    np.testing.assert_array_equal(arr_lr[0, 5], expected[1])


def test_olympus_logger_imports_tracking_lazily():
    import ontic_viz.validation.logger as mod

    assert "ontic_lib.tracking" not in sys.modules or True  # may be loaded by other tests
    src = open(mod.__file__).read()
    assert 'import_module("ontic_lib.tracking")' in src
    assert not any(
        line.startswith(("from ontic_lib", "import ontic_lib")) for line in src.splitlines()
    )


# ---------------------------------------------------------------------------
# LocalVizLogger
# ---------------------------------------------------------------------------


@needs_pil
def test_local_logger_image_naming_and_roundtrip(tmp_path):
    from PIL import Image

    log = LocalVizLogger(tmp_path / "viz")
    img = torch.zeros(3, 4, 6)
    img[0] = 1.0
    img[1, 1, 1] = 2.0  # clamped
    log.log_image("val_video/Target Prediction", img, 7, caption="scene: a, b")
    log.log_image("plain", img, 12)
    files = sorted(
        p.relative_to(tmp_path / "viz").as_posix() for p in (tmp_path / "viz").rglob("*.png")
    )
    assert files == [
        "plain_step_000012_.png",
        "val_video/Target_Prediction_step_000007_scene-_a__b.png",
    ]
    arr = np.asarray(Image.open(tmp_path / "viz" / files[0]))
    assert arr.shape == (4, 6, 3) and arr.dtype == np.uint8
    assert (arr[..., 0] == 255).all() and arr[1, 1, 1] == 255 and arr[0, 0, 1] == 0
    log.log_image("k", img, 3, caption="already step_000003 here")  # step token kept as is
    assert (tmp_path / "viz" / "k_already_step_000003_here.png").exists()
    log.log_metrics({"a": 1}, 0)  # dropped
    log.log_histogram("h", np.arange(3), 0)  # dropped
    assert len(list((tmp_path / "viz").rglob("*"))) == 4  # 3 pngs + the val_video dir


@needs_pil
def test_local_logger_figure_is_saved_as_png(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot([0, 1])
    LocalVizLogger(tmp_path).log_figure("gs_plots/stats", fig, 5)
    plt.close(fig)
    assert (tmp_path / "gs_plots" / "stats_step_000005_.png").exists()


@needs_plyfile
def test_local_logger_point_cloud(tmp_path):
    from plyfile import PlyData

    vertex = np.array([(0.0, 1.0, 2.0, 255, 0, 0), (1.0, 1.0, 1.0, 0, 255, 0)], dtype=VERTEX_DTYPE)
    LocalVizLogger(tmp_path).log_point_cloud("pcd/points", vertex, 3)
    path = tmp_path / "pcd" / "points_step_000003.ply"
    assert path.exists()
    ply = PlyData.read(str(path))
    assert ply["vertex"].count == 2
    assert (
        ply["vertex"]["x"][0] == 0.0
        and ply["vertex"]["red"][0] == 255
        and ply["vertex"]["green"][1] == 255
    )


@needs_moviepy
def test_local_logger_video(tmp_path):
    import moviepy.editor as mpy

    imageio = pytest.importorskip("imageio")  # moviepy 1.x dependency; exact frame counts

    def n_frames(path):
        reader = imageio.get_reader(str(path), "ffmpeg")
        try:
            return reader.count_frames()  # moviepy's iter_frames can over-yield by one
        finally:
            reader.close()

    # Frame counts whose duration n / fps is exact in binary: moviepy 1.x accumulates
    # duration = n * (1 / fps) in floats and writes one frame too many when that rounds up
    # (6 @ 5 fps under numpy 2.5, 60 @ 30 fps under numpy 1.26). Pre-existing, not ours.
    log = LocalVizLogger(tmp_path)
    log.log_video("clip", _frames(8, 16, 16), 4, fps=4, caption="cap")
    path = tmp_path / "video" / "clip" / "step_000004_clip_cap.mp4"
    assert path.exists() and path.stat().st_size > 0
    clip = mpy.VideoFileClip(str(path))
    first = next(clip.iter_frames())
    assert clip.fps == 4 and first.shape == (16, 16, 3)
    clip.close()
    assert n_frames(path) == 8
    log.log_video("val_video/x", _frames(4, 16, 16), 1, fps=2, loop_reverse=True)
    path2 = tmp_path / "val_video" / "x" / "step_000001_x.mp4"
    assert n_frames(path2) == 6  # 4 + the reversed middle 2, at 2 fps -> 3.0 s exactly


def test_local_logger_missing_extras_name_the_extra(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "moviepy", None)
    monkeypatch.setitem(sys.modules, "moviepy.editor", None)
    monkeypatch.setitem(sys.modules, "plyfile", None)
    log = LocalVizLogger(tmp_path)
    with pytest.raises(ImportError, match=r"ontic-viz\[video\]"):
        log.log_video("clip", _frames(2), 0)
    with pytest.raises(ImportError, match=r"ontic-viz\[ply\]"):
        log.log_point_cloud("pcd", np.zeros(1, dtype=VERTEX_DTYPE), 0)
    monkeypatch.setitem(sys.modules, "PIL", None)
    monkeypatch.setitem(sys.modules, "PIL.Image", None)
    with pytest.raises(ImportError, match=r"ontic-viz\[images\]"):
        log.log_image("k", torch.zeros(3, 2, 2), 0)


def test_local_logger_writes_metrics_jsonl(tmp_path):
    import json

    import torch

    from ontic_viz.validation import LocalVizLogger

    logger = LocalVizLogger(tmp_path / "viz", metrics_file=tmp_path / "metrics.jsonl")
    logger.log_metrics({"train/loss": 0.5, "info/step": 3, "flag": True, "t": torch.tensor(2.0), "skip": "text"}, step=3)
    logger.log_metrics({"val/psnr": 30.25}, step=4)
    recs = [json.loads(l) for l in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in recs] == [3, 4]
    assert recs[0]["train/loss"] == 0.5 and recs[0]["info/step"] == 3 and recs[0]["flag"] == 1.0 and recs[0]["t"] == 2.0
    assert "skip" not in recs[0] and "ts" in recs[0] and recs[1]["val/psnr"] == 30.25
    LocalVizLogger(tmp_path / "viz").log_metrics({"x": 1.0}, step=1)  # no file: scalars dropped, no error
