"""Metrics tracking: JSON lines under ``./output/metrics.jsonl`` plus best-effort W&B and
Olympus mirroring, and a direct Olympus API for runs that manage their own identity.

Two ways of reaching Olympus live here and must not be mixed in one process:

* :func:`init` -> :class:`Tracker`: for runs launched by the ontic job bootstrap. The run's
  identity comes from the environment (``ONTIC_OLYMPUS_PROJECT`` / ``ONTIC_OLYMPUS_RUN``),
  metrics.jsonl is the durable record, and the mirror's worker thread owns every SDK call
  so the training loop never waits on the server.
* :func:`init_run` / :func:`run_log` / :func:`finish_run` plus the media adapters
  (:func:`image`, :func:`video`, :func:`point_cloud`, :func:`histogram`, :func:`html`,
  :func:`figure`): for interactive sessions and HTCondor jobs that pick their own project
  and run name. The calling thread drives the SDK directly. ``OLYMPUS_DATA_DIR`` is read
  once, when olympus is imported, so :func:`set_storage_dir` has to run first; reaching
  every olympus symbol through :func:`olympus` (lazy) makes that ordering hold by
  construction. Nothing here imports olympus, numpy or torch at module import time.
"""

from __future__ import annotations

import _thread
import json
import os
import sys
import threading
import time
import traceback
import urllib.request
import warnings
from collections import deque
from pathlib import Path

from .deps import optional_import

_OLYMPUS_STOP_POLL_S = 5.0
_OLYMPUS_TAIL_FLUSH_S = 15.0
_OLYMPUS_TAIL_LINES = 200
_OLYMPUS_QUEUE_MAX = 8192
_OLYMPUS_DRAIN_INTERVAL_S = 0.25
_OLYMPUS_FINISH_JOIN_S = 45.0
_OLYMPUS_DOC_MAX_BYTES = 1_000_000
_OLYMPUS_DOC_MAX_FILES = 10
_OLYMPUS_ERROR_MAX_BYTES = 64 * 1024


class Tracker:
    """Appends one JSON line per log() call to output/metrics.jsonl — the durable,
    plain-text metrics record that ships to B2 with the job — and mirrors to W&B
    and Olympus when configured (best-effort, never fatal; ONTIC_LIB_NO_OLYMPUS=1
    disables the Olympus mirror entirely). Flushed per line so the
    bootstrap's incremental sync sees fresh data; fsync coalesced to >=1 s. After a
    hard kill the final line may be truncated — read_metrics() skips it; any byte
    prefix of the file is a valid record set.

    Usable as a context manager: an exception leaving the `with` block declares the
    Olympus run failed before re-raising."""

    _FSYNC_EVERY_S = 1.0

    def __init__(self, fh, wandb_run, olympus_mirror=None, out: Path | None = None):
        self._fh = fh
        self._wandb = wandb_run
        self._olympus = olympus_mirror
        self._out = out if out is not None else Path(fh.name).parent
        self._last_fsync = 0.0
        self._summary: dict = {}

    def log(self, metrics: dict, step: int | None = None) -> None:
        rec = {"step": step, "ts": time.time(), **metrics}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        now = time.monotonic()
        if now - self._last_fsync >= self._FSYNC_EVERY_S:
            os.fsync(self._fh.fileno())
            self._last_fsync = now
        for key, value in metrics.items():
            if isinstance(value, (int, float, str)):
                self._summary[key] = value
        if self._wandb is not None:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass
        if self._olympus is not None:
            self._olympus.log(metrics, step=step)

    def finish(self, failed: bool = False) -> None:
        try:
            self._write_summary()
        finally:
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
        if self._olympus is not None:
            self._olympus.finish(failed=failed)

    def _write_summary(self) -> None:
        """output/metrics.json: the final value of every scalar metric, for ontic's
        summary promotion. Derived purely from what log() saw; best-effort."""
        try:
            path = self._out / "metrics.json"
            path.write_text(json.dumps(self._summary, indent=2) + "\n", encoding="utf-8")
        except Exception:
            pass

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is not None and self._olympus is not None:
            self._olympus.report_error(exc_type, exc, tb)
        self.finish(failed=exc_type is not None)
        return False


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
    if not (os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").is_file()):
        return None
    try:
        import wandb

        return wandb.init(
            project=project, id=run_id, resume="allow", config=config, settings={"silent": True}
        )
    except Exception:
        return None


def _olympus_notice(what: str) -> None:
    print(f"[ontic-lib] olympus mirror: {what}", file=sys.stderr)


def _post_json(url: str, payload: dict, headers: dict, timeout: float = 4.0) -> object:
    """One POST of a JSON body; returns the decoded JSON response. Raises on any
    transport or decode problem — callers swallow."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class _TeeStream:
    """Pass-through wrapper over a real stream that also feeds complete lines to a
    sink. The wrapped stream always gets the bytes first; the sink can never fail
    the write."""

    def __init__(self, orig, sink):
        self._orig = orig
        self._sink = sink
        self._partial = ""

    def write(self, text):
        n = self._orig.write(text)
        try:
            self._partial += str(text)
            while "\n" in self._partial:
                line, self._partial = self._partial.split("\n", 1)
                if line.strip():
                    self._sink(line)
        except Exception:
            pass
        return n

    def flush(self):
        self._orig.flush()

    def __getattr__(self, name):
        return getattr(self._orig, name)


class _OlympusMirror:
    """Best-effort mirror of a run to the team Olympus server. metrics.jsonl never
    depends on it: every path here degrades to a single stderr notice.

    The caller thread never touches the olympus SDK. log() appends a shallow
    copy to a bounded in-process queue (drop-oldest on overflow: the mirror is
    lossy by design, metrics.jsonl remains the durable record) and one worker
    thread owns all SDK interaction, including olympus.init(), so a slow or
    unreachable server can never stall the training loop.

    ONTIC_LIB_NO_OLYMPUS=1 is the operator escape hatch: it disables the mirror
    entirely regardless of env activation, checked once at init.

    Runs its own dashboard-stop poller (the SDK's is disabled via
    OLYMPUS_DISABLE_REMOTE_STOP) so a mode="stop" can drop the ontic terminate
    marker before interrupting."""

    def __init__(self, olympus_module, config: dict | None, out: Path):
        self._olympus = olympus_module
        self._config = config
        self._out = out
        self._project = os.environ.get("ONTIC_OLYMPUS_PROJECT", "")
        self._run_name = os.environ.get("ONTIC_OLYMPUS_RUN", "")
        self._server = (os.environ.get("OLYMPUS_SERVER_URL") or "").rstrip("/")
        self._api_key = os.environ.get("OLYMPUS_API_KEY")
        self._workspace = os.environ.get("OLYMPUS_WORKSPACE")
        self._resume = _olympus_resuming()
        self._run = None
        self._queue: deque[tuple[dict, int | None]] = deque(maxlen=_OLYMPUS_QUEUE_MAX)
        self._shutdown = threading.Event()
        self._ready = threading.Event()
        self._worker: threading.Thread | None = None
        self._tail_lines: deque[str] = deque(maxlen=_OLYMPUS_TAIL_LINES)
        self._orig_stdout = None
        self._orig_stderr = None
        self._prev_excepthook = None
        self._log_warned = False
        self._overflow_noticed = False
        self._finished = False
        self._failed = False
        self._failed_declared = False
        self._error_sent = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if os.environ.get("ONTIC_LIB_NO_LOG_TAIL") != "1":
            self._start_tail()
        if self._server:
            thread = threading.Thread(
                target=self._poll_loop, name="ontic-olympus-stop-poll", daemon=True
            )
            thread.start()
        self._install_excepthook()
        self._worker = threading.Thread(
            target=self._worker_loop, name="ontic-olympus-worker", daemon=True
        )
        self._worker.start()

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Hot path: one shallow dict copy and a lock-free bounded-deque append.
        Never blocks, never talks to the SDK; the worker drains the queue."""
        if self._finished:
            return
        if len(self._queue) >= self._queue.maxlen and not self._overflow_noticed:
            self._overflow_noticed = True
            _olympus_notice(
                "metric queue full; dropping oldest mirrored points (metrics.jsonl is unaffected)"
            )
        self._queue.append((dict(metrics), step))

    def finish(self, failed: bool = False) -> None:
        if self._finished:
            return
        self._finished = True
        self._failed = failed
        self._stop_tail()
        self._shutdown.set()
        if self._worker is not None:
            self._worker.join(timeout=_OLYMPUS_FINISH_JOIN_S)
            if self._worker.is_alive():
                _olympus_notice(
                    "could not flush to the server in time; metrics.jsonl is unaffected"
                )
        self._restore_excepthook()

    # -- worker thread: owns every olympus SDK call ------------------------

    def _worker_loop(self) -> None:
        try:
            self._run = self._olympus.init(
                project=self._project,
                name=self._run_name,
                config=self._config,
                resume="allow" if self._resume else "never",
                embed=False,
            )
        except Exception as e:
            _olympus_notice(f"init failed ({e}); mirror disabled")
        finally:
            self._ready.set()
        if self._run is None:
            self._queue.clear()
            return
        self._upload_docs(self._olympus)
        next_tail_flush = time.monotonic() + _OLYMPUS_TAIL_FLUSH_S
        while not self._shutdown.wait(_OLYMPUS_DRAIN_INTERVAL_S):
            self._drain_queue()
            if time.monotonic() >= next_tail_flush:
                next_tail_flush = time.monotonic() + _OLYMPUS_TAIL_FLUSH_S
                self._flush_tail()
        self._drain_queue()
        self._flush_tail()
        try:
            self._run.finish()
        except Exception as e:
            _olympus_notice(f"finish failed ({e}); metrics.jsonl is unaffected")
        if self._failed:
            self._declare_failed()

    def _drain_queue(self) -> None:
        while True:
            try:
                metrics, step = self._queue.popleft()
            except IndexError:
                return
            try:
                self._run.log(metrics, step=step)
            except Exception as e:
                if not self._log_warned:
                    self._log_warned = True
                    _olympus_notice(f"log failed ({e}); metrics.jsonl is unaffected")

    def report_error(self, exc_type, exc, tb) -> None:
        """Ship the failure traceback before the failed declaration: appended to
        output/error.txt so the durable B2 record carries it, and POSTed to
        set_run_error so the dashboard can show why. Best-effort, once per run
        (the with-block exit and the excepthook can both fire for the same
        exception)."""
        if self._error_sent:
            return
        self._error_sent = True
        try:
            text = "".join(traceback.format_exception(exc_type, exc, tb))
        except Exception:
            return
        data = text.encode("utf-8", errors="replace")
        if len(data) > _OLYMPUS_ERROR_MAX_BYTES:
            text = data[-_OLYMPUS_ERROR_MAX_BYTES:].decode("utf-8", errors="ignore")
        try:
            with (self._out / "error.txt").open("a", encoding="utf-8") as fh:
                fh.write(text)
        except Exception:
            pass
        if not self._server:
            return
        try:
            payload = _post_json(
                f"{self._server}/api/set_run_error",
                {"project": self._project, "run": self._run_name, "text": text},
                self._headers(),
                timeout=3.0,
            )
        except Exception as e:
            _olympus_notice(f"set_run_error failed ({e}); output/error.txt has the traceback")
            return
        if isinstance(payload, dict) and payload.get("error"):
            _olympus_notice(
                f"set_run_error rejected ({payload['error']}); output/error.txt has the traceback"
            )

    def _declare_failed(self) -> None:
        """The SDK only ever declares finished; a crashed run must say failed.
        Sent after finish() so the flush happens first and the final word wins."""
        if self._failed_declared:
            return
        self._failed_declared = True
        try:
            self._run._declare_status("failed", send=True)
        except Exception:
            pass

    # -- uncaught exceptions -----------------------------------------------

    def _install_excepthook(self) -> None:
        prev = sys.excepthook
        self._prev_excepthook = prev

        def hook(exc_type, exc, tb):
            try:
                self.report_error(exc_type, exc, tb)
            except Exception:
                pass
            try:
                self.finish(failed=True)
            except Exception:
                pass
            prev(exc_type, exc, tb)

        self._hook = hook
        sys.excepthook = hook

    def _restore_excepthook(self) -> None:
        if self._prev_excepthook is not None and sys.excepthook is getattr(self, "_hook", None):
            sys.excepthook = self._prev_excepthook
        self._prev_excepthook = None

    # -- stop poller -------------------------------------------------------

    def _headers(self) -> dict:
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._workspace:
            headers["x-olympus-workspace"] = self._workspace
        headers["x-olympus-project"] = self._project
        return headers

    def _poll_loop(self) -> None:
        while not self._shutdown.wait(_OLYMPUS_STOP_POLL_S):
            if self._poll_once():
                return

    def _poll_once(self) -> bool:
        """One stop-request round trip. True once a stop was acted on."""
        try:
            payload = _post_json(
                f"{self._server}/api/get_run_stop_request",
                {"project": self._project, "run": self._run_name},
                self._headers(),
            )
        except Exception:
            return False
        answer = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
        if isinstance(answer, dict):
            stop = answer.get("stop") is True
            mode = answer.get("mode") or "stop"
        else:
            stop = answer is True
            mode = "stop"
        if not stop:
            return False
        if mode == "stop":
            try:
                marker = self._out / ".ontic" / "terminate-requested"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            except Exception:
                pass
        _olympus_notice(
            f"dashboard requested mode={mode} for run '{self._run_name}'; "
            "interrupting the main thread"
        )
        try:
            _thread.interrupt_main()
        except Exception:
            pass
        return True

    # -- log tail ----------------------------------------------------------

    def _start_tail(self) -> None:
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = _TeeStream(self._orig_stdout, self._tail_lines.append)
        sys.stderr = _TeeStream(self._orig_stderr, self._tail_lines.append)

    def _flush_tail(self) -> None:
        """Worker-thread only. The sink is a lock-free bounded-deque append;
        popleft here is safe against concurrent appends."""
        if self._run is None:
            return
        while True:
            try:
                line = self._tail_lines.popleft()
            except IndexError:
                return
            try:
                self._run.log_system({"console": line})
            except Exception:
                return

    def _stop_tail(self) -> None:
        if self._orig_stdout is not None and isinstance(sys.stdout, _TeeStream):
            sys.stdout = self._orig_stdout
        if self._orig_stderr is not None and isinstance(sys.stderr, _TeeStream):
            sys.stderr = self._orig_stderr
        self._orig_stdout = None
        self._orig_stderr = None

    # -- experiment docs ---------------------------------------------------

    def _upload_docs(self, olympus_module) -> None:
        """Committed experiment docs sitting in the job dir, onto the run's files."""
        try:
            cwd = Path.cwd()
            candidates = [cwd / "README.md"]
            candidates += sorted(cwd.glob("RESULTS-*.md"))
            candidates += sorted(cwd.glob("PREREG-*.md"))
            count = 0
            for path in candidates:
                if count >= _OLYMPUS_DOC_MAX_FILES:
                    break
                if not path.is_file() or path.stat().st_size > _OLYMPUS_DOC_MAX_BYTES:
                    continue
                try:
                    olympus_module.save(str(path))
                    count += 1
                except Exception:
                    continue
        except Exception:
            pass


# One key per env var, verbatim: the launcher already formats the values (tags
# ", "-joined, deps space-separated producer job ids), and the backfill importer
# writes the same `description`/`tags`/`deps` keys, so live and imported runs
# present one config contract to the dashboard and query_runs.
_OLYMPUS_CONFIG_ENV = (
    ("job_id", "ONTIC_JOB_ID"),
    ("experiment", "ONTIC_EXPERIMENT"),
    ("experiment_sha", "ONTIC_EXPERIMENT_SHA"),
    ("attempt", "ONTIC_ATTEMPT"),
    ("description", "ONTIC_DESCRIPTION"),
    ("tags", "ONTIC_TAGS"),
    ("deps", "ONTIC_DEPS"),
    ("git_parents", "ONTIC_GIT_PARENTS"),
    ("git_subject", "ONTIC_GIT_SUBJECT"),
    ("git_branch", "ONTIC_GIT_BRANCH"),
)


def _olympus_resuming() -> bool:
    if os.environ.get("ONTIC_RESUME") == "1":
        return True
    try:
        return int(os.environ.get("ONTIC_ATTEMPT", "1")) > 1
    except ValueError:
        return False


def _maybe_olympus(config: dict | None, out: Path):
    if os.environ.get("ONTIC_LIB_NO_OLYMPUS") == "1":
        return None
    if not (os.environ.get("ONTIC_OLYMPUS_PROJECT") and os.environ.get("ONTIC_OLYMPUS_RUN")):
        return None
    os.environ["OLYMPUS_DISABLE_REMOTE_STOP"] = "1"
    try:
        import olympus
    except Exception:
        _olympus_notice("olympus package not importable; mirror disabled")
        return None
    try:
        cfg: dict = {}
        for key, env in _OLYMPUS_CONFIG_ENV:
            value = os.environ.get(env)
            if value:
                cfg[key] = value
        cfg.update(config or {})
        mirror = _OlympusMirror(olympus, cfg, out)
        mirror.start()
        return mirror
    except Exception as e:
        _olympus_notice(f"start failed ({e}); mirror disabled")
        return None


def init(project: str, config: dict | None = None) -> Tracker:
    out = Path.cwd() / "output"
    out.mkdir(parents=True, exist_ok=True)
    fh = (out / "metrics.jsonl").open("a", encoding="utf-8")
    fh.write(
        json.dumps(
            {"_type": "config", "project": project, "config": config or {}, "ts": time.time()}
        )
        + "\n"
    )
    fh.flush()
    return Tracker(
        fh, _maybe_wandb(project, config), olympus_mirror=_maybe_olympus(config, out), out=out
    )


# --- Direct Olympus API: the calling thread drives the SDK --------------------


def olympus():
    """The olympus SDK module, imported on first use (after `set_storage_dir`)."""
    return optional_import(
        "olympus",
        package="ontic-lib",
        extra="olympus",
        what="tracking's direct Olympus API",
        repo_url="ssh://git@github.com/onticlabs/olympus.git",
    )


def active_run():
    """The SDK's current run, or None. Never imports olympus itself."""
    sdk = sys.modules.get("olympus")
    if sdk is None:
        return None
    return sdk.context_vars.current_run.get()


def set_storage_dir(output_dir: str | Path) -> Path:
    """Put the local run store under ``<output_dir>/olympus`` unless ``OLYMPUS_DATA_DIR``
    is set; returns the directory in effect. Must run before olympus is imported (the
    SDK reads the variable once); warns and keeps the already-bound directory otherwise."""
    explicit = os.environ.get("OLYMPUS_DATA_DIR")
    if explicit:
        return Path(explicit)
    target = (Path(output_dir) / "olympus").resolve()
    sdk = sys.modules.get("olympus")
    if sdk is not None:
        bound = sdk.utils.OLYMPUS_DATA_DIR
        warnings.warn(
            f"olympus was already imported; run store stays at {bound} instead of {target}. "
            "Export OLYMPUS_DATA_DIR to choose it explicitly.",
            stacklevel=2,
        )
        return Path(bound)
    target.mkdir(parents=True, exist_ok=True)
    os.environ["OLYMPUS_DATA_DIR"] = str(target)
    return target


def init_run(
    project: str,
    name: str,
    *,
    group: str | None = None,
    server_url: str | None = None,
    config: dict | None = None,
    resume: str = "allow",
    system_metrics: bool = True,
    system_interval: float = 10.0,
    storage_dir: str | Path | None = None,
):
    """Start (or resume) an Olympus run from the calling thread; returns the SDK run.

    Resume keys on the run *name* (olympus has no separately addressable run id), so
    ``name`` must be a pure function of the caller's config for a restarted job to land
    back on the same run. ``storage_dir`` goes through :func:`set_storage_dir` first.
    System metrics (GPU via nvidia-ml-py, CPU/RAM via psutil) are sampled on an SDK
    background thread every ``system_interval`` seconds; a missing dependency silently
    disables its half."""
    if storage_dir is not None:
        set_storage_dir(storage_dir)
    return olympus().init(
        project=project,
        name=name,
        group=group or None,
        server_url=server_url or None,
        config=config,
        resume=resume,
        embed=False,
        auto_log_gpu=system_metrics,
        gpu_log_interval=system_interval,
        auto_log_cpu=system_metrics,
        cpu_log_interval=system_interval,
    )


def finish_run() -> None:
    """Finish the run started by `init_run`."""
    olympus().finish()


def run_log(metrics: dict, step: int | None = None) -> None:
    """Log scalars and media objects on the run started by `init_run`."""
    olympus().log(metrics, step=step)


def ensure_ffmpeg() -> str | None:
    """A *working* `ffmpeg` on PATH, borrowing imageio-ffmpeg's static build if needed.

    olympus encodes video by shelling out to `ffmpeg`, and some container images ship
    without it or with a build that cannot run (`ffmpeg -version` fails). Such a binary
    is ignored in favour of the static one imageio-ffmpeg carries under a versioned
    name, linked in as `ffmpeg` under ``$OLYMPUS_DATA_DIR/bin`` (else
    ``~/.cache/olympus/bin``) and prepended to PATH. Returns the resolved path, or None
    when no runnable binary can be found."""
    import shutil
    import subprocess

    def runs(exe: str) -> bool:
        try:
            proc = subprocess.run([exe, "-version"], capture_output=True, timeout=20)
        except (OSError, subprocess.SubprocessError):
            return False
        return proc.returncode == 0

    found = shutil.which("ffmpeg")
    if found and runs(found):
        return found
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    if not exe.exists():
        return None
    bindir = Path(os.environ.get("OLYMPUS_DATA_DIR", Path.home() / ".cache" / "olympus")) / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    link = bindir / "ffmpeg"
    if link.is_symlink() or link.exists():
        link.unlink()
    try:
        link.symlink_to(exe)
    except OSError:
        return None
    os.environ["PATH"] = f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"
    return str(link) if runs(str(link)) else None


# -- media adapters: tensors and arrays in, olympus media objects out ----------


def image(img, caption: str | None = None):
    """CHW float tensor in [0, 1] (or HWC uint8 array) -> `olympus.Image`."""
    import numpy as np

    torch = sys.modules.get("torch")  # a tensor argument implies torch is already loaded
    if torch is not None and isinstance(img, torch.Tensor):
        arr = img.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
    else:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)
    return olympus().Image(np.ascontiguousarray(arr), caption=caption)


def video(frames, fps: int = 30, caption: str | None = None, fmt: str = "mp4"):
    """uint8 frames shaped (F, C, H, W) or (B, F, C, H, W) -> `olympus.Video`. Makes sure
    a working `ffmpeg` is on PATH first (see `ensure_ffmpeg`)."""
    import numpy as np

    ensure_ffmpeg()
    arr = np.ascontiguousarray(np.asarray(frames))
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return olympus().Video(arr, caption=caption, fps=fps, format=fmt)


def point_cloud(vertex_data, caption: str | None = None):
    """(N, 3) xyz or (N, 6) xyz+rgb -> `olympus.Object3D`.

    Also accepts a PLY-style structured array (`x/y/z/red/green/blue` fields). RGB is
    rounded and clipped into the integer [0, 255] range olympus requires; olympus itself
    thins anything past its own point cap."""
    import numpy as np

    arr = np.asarray(vertex_data)
    if arr.dtype.names:
        arr = np.stack([arr[n] for n in arr.dtype.names], axis=-1)
    arr = arr.astype(np.float32)
    if arr.shape[-1] >= 6:
        arr = arr[:, :6].copy()
        arr[:, 3:6] = np.clip(np.round(arr[:, 3:6]), 0, 255)
    else:
        arr = arr[:, :3].copy()
    return olympus().Object3D(np.ascontiguousarray(arr), caption=caption)


def histogram(values, num_bins: int = 64):
    """Any array-like, flattened -> `olympus.Histogram`."""
    import numpy as np

    return olympus().Histogram(np.asarray(values).reshape(-1), num_bins=num_bins)


def html(markup: str, caption: str | None = None):
    """An HTML string -> `olympus.Html`."""
    return olympus().Html(markup, caption=caption)


def figure(fig, caption: str | None = None):
    """A matplotlib or plotly figure -> `olympus.Html`, which renders it (olympus has no
    Image-from-figure)."""
    return olympus().Html(fig, caption=caption)


# ---------------------------------------------------------------------------
# GPU utilisation of the calling process's device, in Olympus's own key layout
# (``gpu/<index>/utilization`` ...), so offline runs and mirrors look like the
# client's automatic system metrics. Read inside the training process: a mirror
# on a login node would describe the wrong machine.
# ---------------------------------------------------------------------------

_NVSMI_FIELDS = "utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu"


def _parse_nvsmi_line(line: str) -> dict:
    vals = [v.strip() for v in line.split(",")]
    def num(i):
        try:
            return float(vals[i])
        except (ValueError, IndexError):
            return None
    util, mem_util, used_mib, total_mib, power, limit, temp = (num(i) for i in range(7))
    out = {}
    if util is not None:
        out["utilization"] = util
    if mem_util is not None:
        out["memory_utilization"] = mem_util
    if used_mib is not None:
        out["allocated_memory"] = used_mib / 1024
    if total_mib is not None:
        out["total_memory"] = total_mib / 1024
    if used_mib is not None and total_mib:
        out["memory_usage"] = used_mib / total_mib
    if power is not None:
        out["power"] = power
    if power is not None and limit:
        out["power_percent"] = power / limit
    if temp is not None:
        out["temp"] = temp
    return out


def gpu_metrics(device: int | None = None, label: str | int | None = None) -> dict:
    """Utilisation, memory (GiB and fraction), power and temperature of one CUDA device, keyed
    ``gpu/<label>/<name>`` like Olympus's automatic GPU metrics. ``device`` defaults to the
    current torch device; ``label`` (e.g. the distributed rank) defaults to the device index.
    Uses pynvml when installed, else one ``nvidia-smi`` query; ``{}`` when neither works."""
    if device is None:
        try:
            import torch

            device = torch.cuda.current_device() if torch.cuda.is_available() else 0
        except Exception:
            device = 0
    label = device if label is None else label
    vals: dict = {}
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        vals = {"utilization": float(util.gpu), "memory_utilization": float(util.memory),
                "allocated_memory": mem.used / 2**30, "total_memory": mem.total / 2**30, "memory_usage": mem.used / mem.total}
        try:
            vals["power"] = pynvml.nvmlDeviceGetPowerUsage(h) / 1000
            vals["power_percent"] = vals["power"] / (pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000)
        except Exception:
            pass
        try:
            vals["temp"] = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        except Exception:
            pass
    except Exception:
        try:
            import subprocess

            out = subprocess.run(
                ["nvidia-smi", f"--id={device}", f"--query-gpu={_NVSMI_FIELDS}", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            ).stdout.strip().splitlines()
            vals = _parse_nvsmi_line(out[0]) if out else {}
        except Exception:
            vals = {}
    return {f"gpu/{label}/{k}": v for k, v in vals.items()}
