"""The direct Olympus API and media adapters, against a fake `olympus` module."""

import contextvars
import importlib.util
import os
import shutil
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from ontic_lib import tracking


class _Media:
    """Stand-in for an olympus media class: keeps its constructor arguments."""

    def __init__(self, value, **kwargs):
        self.value = value
        self.kwargs = kwargs


def _fake_olympus(monkeypatch, bound_dir="/bound/olympus"):
    mod = types.ModuleType("olympus")
    mod.calls = []
    run = types.SimpleNamespace(name="run-1", id="run-1")

    def init(**kwargs):
        mod.calls.append(("init", kwargs))
        return run

    mod.init = init
    mod.log = lambda metrics, step=None: mod.calls.append(("log", metrics, step))
    mod.finish = lambda: mod.calls.append(("finish",))
    for name in ("Image", "Video", "Object3D", "Histogram", "Html"):
        setattr(mod, name, type(name, (_Media,), {}))
    mod.context_vars = types.SimpleNamespace(
        current_run=contextvars.ContextVar("current_run", default=None)
    )
    mod.utils = types.SimpleNamespace(OLYMPUS_DATA_DIR=bound_dir)
    monkeypatch.setitem(sys.modules, "olympus", mod)
    return mod, run


# --- run lifecycle -----------------------------------------------------------


def test_init_run_passes_the_reference_arguments(monkeypatch):
    mod, run = _fake_olympus(monkeypatch)
    got = tracking.init_run("proj", "run-1", group="", server_url="", config={"lr": 0.1})
    assert got is run
    assert mod.calls == [
        (
            "init",
            {
                "project": "proj",
                "name": "run-1",
                "group": None,  # empty group / server_url collapse to None
                "server_url": None,
                "config": {"lr": 0.1},
                "resume": "allow",
                "embed": False,
                "auto_log_gpu": True,
                "gpu_log_interval": 10.0,
                "auto_log_cpu": True,
                "cpu_log_interval": 10.0,
            },
        )
    ]


def test_init_run_forwards_group_server_and_system_settings(monkeypatch):
    mod, _ = _fake_olympus(monkeypatch)
    tracking.init_run(
        "proj",
        "r",
        group="g",
        server_url="http://olympus.test",
        resume="never",
        system_metrics=False,
        system_interval=2.5,
    )
    kwargs = mod.calls[0][1]
    assert kwargs["group"] == "g" and kwargs["server_url"] == "http://olympus.test"
    assert kwargs["resume"] == "never" and kwargs["config"] is None
    assert kwargs["auto_log_gpu"] is False and kwargs["auto_log_cpu"] is False
    assert kwargs["gpu_log_interval"] == 2.5 and kwargs["cpu_log_interval"] == 2.5


def test_init_run_binds_the_storage_dir_before_the_sdk(monkeypatch, tmp_path):
    mod, _ = _fake_olympus(monkeypatch)

    def set_storage_dir(output_dir):
        mod.calls.append(("storage", Path(output_dir)))
        return Path(output_dir)

    monkeypatch.setattr(tracking, "set_storage_dir", set_storage_dir)
    tracking.init_run("proj", "r")
    assert [c[0] for c in mod.calls] == ["init"]  # no storage_dir: untouched
    tracking.init_run("proj", "r", storage_dir=tmp_path)
    assert [c[0] for c in mod.calls] == ["init", "storage", "init"]
    assert mod.calls[1] == ("storage", tmp_path)


def test_run_log_and_finish_run_use_the_sdk(monkeypatch):
    mod, _ = _fake_olympus(monkeypatch)
    tracking.run_log({"loss": 0.5}, step=3)
    tracking.run_log({"loss": 0.4})
    tracking.finish_run()
    assert mod.calls == [("log", {"loss": 0.5}, 3), ("log", {"loss": 0.4}, None), ("finish",)]


def test_olympus_missing_names_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "olympus", None)  # forces ImportError
    with pytest.raises(ImportError, match=r"ontic-lib\[olympus\]"):
        tracking.olympus()
    with pytest.raises(ImportError, match=r"ontic-lib\[olympus\]"):
        tracking.html("<b>x</b>")  # the media adapters go through the same door


def test_active_run_is_none_before_olympus_is_imported(monkeypatch):
    monkeypatch.delitem(sys.modules, "olympus", raising=False)
    assert tracking.active_run() is None
    monkeypatch.setitem(sys.modules, "olympus", None)  # a blocked import counts as absent
    assert tracking.active_run() is None
    assert sys.modules["olympus"] is None  # and nothing tried to import it


def test_active_run_reads_the_sdk_context(monkeypatch):
    mod, run = _fake_olympus(monkeypatch)
    assert tracking.active_run() is None
    token = mod.context_vars.current_run.set(run)
    try:
        assert tracking.active_run() is run
    finally:
        mod.context_vars.current_run.reset(token)


# --- storage dir -------------------------------------------------------------


def test_set_storage_dir_explicit_env_wins(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "olympus", raising=False)
    monkeypatch.setenv("OLYMPUS_DATA_DIR", str(tmp_path / "explicit"))
    assert tracking.set_storage_dir(tmp_path / "out") == tmp_path / "explicit"
    assert not (tmp_path / "out").exists()


def test_set_storage_dir_creates_out_olympus(monkeypatch, tmp_path):
    monkeypatch.delitem(sys.modules, "olympus", raising=False)
    monkeypatch.delenv("OLYMPUS_DATA_DIR", raising=False)
    got = tracking.set_storage_dir(tmp_path / "out")
    assert got == (tmp_path / "out" / "olympus").resolve()
    assert got.is_dir()
    assert os.environ["OLYMPUS_DATA_DIR"] == str(got)


def test_set_storage_dir_warns_and_keeps_the_bound_dir_once_imported(monkeypatch, tmp_path):
    monkeypatch.delenv("OLYMPUS_DATA_DIR", raising=False)
    bound = tmp_path / "bound"
    _fake_olympus(monkeypatch, bound_dir=str(bound))
    with pytest.warns(UserWarning, match="already imported"):
        got = tracking.set_storage_dir(tmp_path / "out")
    assert got == bound
    assert "OLYMPUS_DATA_DIR" not in os.environ
    assert not (tmp_path / "out").exists()


# --- media adapters ----------------------------------------------------------


def test_image_from_hwc_uint8_array(monkeypatch):
    mod, _ = _fake_olympus(monkeypatch)
    arr = np.random.default_rng(0).integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    out = tracking.image(arr[:, ::-1], caption="c")  # a non-contiguous view
    assert isinstance(out, mod.Image)
    assert out.value.dtype == np.uint8 and out.value.flags.c_contiguous
    np.testing.assert_array_equal(out.value, arr[:, ::-1])
    assert out.kwargs == {"caption": "c"}


def test_image_from_float_array_scales_to_uint8(monkeypatch):
    _fake_olympus(monkeypatch)
    out = tracking.image(np.array([[[0.0, 0.5, 1.0], [1.5, -0.2, 0.25]]]))
    np.testing.assert_array_equal(out.value, [[[0, 128, 255], [255, 0, 64]]])
    assert out.kwargs == {"caption": None}


def test_image_tensor_matches_array(monkeypatch):
    torch = pytest.importorskip("torch")
    _fake_olympus(monkeypatch)
    arr = np.random.default_rng(1).integers(0, 256, size=(4, 5, 3), dtype=np.uint8)
    chw = torch.from_numpy(arr).permute(2, 0, 1).float() / 255
    from_tensor = tracking.image(chw, caption="c")
    from_array = tracking.image(arr, caption="c")
    np.testing.assert_array_equal(from_tensor.value, from_array.value)
    assert from_tensor.value.shape == (4, 5, 3) and from_tensor.value.dtype == np.uint8
    assert from_tensor.value.flags.c_contiguous
    # a single channel collapses to (H, W); out-of-range values clamp
    gray = tracking.image(torch.full((1, 2, 3), 2.0, requires_grad=True))
    assert gray.value.shape == (2, 3) and gray.value.min() == 255


def test_video_casts_to_uint8_and_ensures_ffmpeg(monkeypatch):
    mod, _ = _fake_olympus(monkeypatch)
    ensured = []
    monkeypatch.setattr(tracking, "ensure_ffmpeg", lambda: ensured.append(True))
    frames = np.full((2, 3, 4, 4), 3.7, dtype=np.float32)
    out = tracking.video(frames, fps=12, caption="v", fmt="gif")
    assert ensured == [True]
    assert isinstance(out, mod.Video)
    assert out.value.dtype == np.uint8 and out.value.shape == (2, 3, 4, 4)
    assert out.value.flags.c_contiguous and out.value.max() == 3
    assert out.kwargs == {"caption": "v", "fps": 12, "format": "gif"}
    batched = tracking.video(np.zeros((2, 5, 3, 4, 4), dtype=np.uint8))
    assert batched.value.shape == (2, 5, 3, 4, 4)
    assert batched.kwargs == {"caption": None, "fps": 30, "format": "mp4"}


def test_point_cloud_accepts_xyz_rgb_and_ply_rows(monkeypatch):
    mod, _ = _fake_olympus(monkeypatch)
    xyz = np.arange(12, dtype=np.float64).reshape(4, 3)
    out = tracking.point_cloud(xyz, caption="pc")
    assert isinstance(out, mod.Object3D)
    assert out.value.dtype == np.float32 and out.value.shape == (4, 3)
    assert out.value.flags.c_contiguous
    np.testing.assert_array_equal(out.value, xyz)
    assert out.kwargs == {"caption": "pc"}

    xyzrgb = np.array([[0, 0, 0, 255.6, -3, 12.4], [1, 2, 3, 0.5, 300, 128]], dtype=np.float64)
    out = tracking.point_cloud(xyzrgb)
    np.testing.assert_array_equal(out.value[:, :3], xyzrgb[:, :3])
    np.testing.assert_array_equal(out.value[:, 3:], [[255, 0, 12], [0, 255, 128]])
    assert out.kwargs == {"caption": None}
    assert tracking.point_cloud(np.ones((2, 7))).value.shape == (2, 6)  # extras dropped

    ply = np.zeros(
        2, dtype=[(n, "f4") for n in "xyz"] + [(n, "u1") for n in ("red", "green", "blue")]
    )
    ply["x"], ply["z"] = [1, 2], [3, 4]
    ply["red"], ply["blue"] = [10, 20], [255, 0]
    out = tracking.point_cloud(ply)
    assert out.value.shape == (2, 6) and out.value.dtype == np.float32
    np.testing.assert_array_equal(out.value, [[1, 0, 3, 10, 0, 255], [2, 0, 4, 20, 0, 0]])


def test_histogram_flattens_values(monkeypatch):
    mod, _ = _fake_olympus(monkeypatch)
    out = tracking.histogram([[1.0, 2.0], [3.0, 4.0]], num_bins=8)
    assert isinstance(out, mod.Histogram)
    np.testing.assert_array_equal(out.value, [1.0, 2.0, 3.0, 4.0])
    assert out.kwargs == {"num_bins": 8}
    assert tracking.histogram(np.zeros((2, 3))).kwargs == {"num_bins": 64}


def test_html_wraps_markup(monkeypatch):
    mod, _ = _fake_olympus(monkeypatch)
    out = tracking.html("<b>hi</b>", caption="h")
    assert isinstance(out, mod.Html)
    assert out.value == "<b>hi</b>" and out.kwargs == {"caption": "h"}


def test_figure_hands_the_figure_to_html(monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    mod, _ = _fake_olympus(monkeypatch)
    fig = Figure()
    fig.subplots().plot([0, 1], [1, 0])
    out = tracking.figure(fig, caption="f")
    assert isinstance(out, mod.Html)
    assert out.value is fig and out.kwargs == {"caption": "f"}


# --- ffmpeg ------------------------------------------------------------------


def _runs(exe):
    try:
        return subprocess.run([exe, "-version"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


_HAS_STATIC_FFMPEG = importlib.util.find_spec("imageio_ffmpeg") is not None
_HAS_SYSTEM_FFMPEG = shutil.which("ffmpeg") is not None and _runs(shutil.which("ffmpeg"))


@pytest.mark.skipif(not (_HAS_SYSTEM_FFMPEG or _HAS_STATIC_FFMPEG), reason="no ffmpeg available")
def test_ensure_ffmpeg_returns_a_runnable_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("OLYMPUS_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PATH", os.environ.get("PATH", ""))  # the function may prepend to it
    exe = tracking.ensure_ffmpeg()
    assert exe is not None
    assert subprocess.run([exe, "-version"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _HAS_STATIC_FFMPEG, reason="imageio-ffmpeg not installed")
def test_ensure_ffmpeg_skips_a_broken_binary_on_path(tmp_path, monkeypatch):
    broken = tmp_path / "bin"
    broken.mkdir()
    (broken / "ffmpeg").write_text("#!/bin/sh\nexit 127\n")
    (broken / "ffmpeg").chmod(0o755)
    monkeypatch.setenv("PATH", f"{broken}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("OLYMPUS_DATA_DIR", str(tmp_path / "store"))
    exe = tracking.ensure_ffmpeg()
    assert exe == str(tmp_path / "store" / "bin" / "ffmpeg")
    assert Path(exe).is_symlink()
    assert subprocess.run([exe, "-version"], capture_output=True).returncode == 0
    assert os.environ["PATH"].startswith(str(tmp_path / "store" / "bin"))


def test_ensure_ffmpeg_is_none_without_any_binary(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", None)  # and no static build
    assert tracking.ensure_ffmpeg() is None


# --- import footprint --------------------------------------------------------


def test_import_pulls_in_no_optional_module():
    """`import ontic_lib.tracking` must work with neither torch, numpy nor olympus
    importable: the direct API and the media adapters import them on use only."""
    script = textwrap.dedent(
        """
        import sys
        blocked = ("numpy", "torch", "olympus", "imageio_ffmpeg", "matplotlib")
        sys.modules.update({name: None for name in blocked})
        import ontic_lib.tracking
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=120
    )
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
