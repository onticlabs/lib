"""Hot-path benchmark for the Olympus mirror in ontic_lib.tracking.

Not a pytest file. Run one scenario per process so env vars and the olympus
SDK's module state stay clean:

    uv run --extra olympus python tests/bench_tracking.py off
    uv run --extra olympus python tests/bench_tracking.py local
    uv run --extra olympus python tests/bench_tracking.py server
    uv run --extra olympus python tests/bench_tracking.py slow
    uv run --extra olympus python tests/bench_tracking.py tee_off
    uv run --extra olympus python tests/bench_tracking.py tee_on
    uv run --extra olympus python tests/bench_tracking.py tee_slow

Scenarios time Tracker.log() (or print()) per call on the caller thread:

- off:     mirror disabled (no ONTIC_OLYMPUS_* env). The metrics.jsonl floor.
- local:   mirror on, olympus in local SQLite mode (no server URL).
- server:  mirror on, olympus logging to a fake healthy HTTP server.
- slow:    mirror on, the fake server answers /api/bulk_log after a 5 s stall,
           modelling a slow or wedged production server.
- tee_off / tee_on: 10k print() calls with the stdout tee inactive vs active.
- tee_slow: paced print() calls while the slow server wedges the SDK sender.

Media never reaches the mirror: Tracker.log() serializes every record with
json.dumps for metrics.jsonl first, which raises TypeError on olympus media
objects, so the mirror only ever sees JSON-scalar payloads.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

N_SCALAR = 10_000
N_SLOW = 2_000
N_PRINT = 10_000
SLOW_BULK_LOG_S = 5.0


def _percentiles(ns: list[int]) -> str:
    ns = sorted(ns)
    us = [x / 1000 for x in ns]
    med = statistics.median(us)
    p99 = us[int(len(us) * 0.99) - 1]
    return (
        f"n={len(us)} median={med:.1f}us mean={statistics.fmean(us):.1f}us "
        f"p99={p99:.1f}us max={max(us):.0f}us total={sum(us) / 1e6:.2f}s"
    )


class _FakeOlympusHandler(BaseHTTPRequestHandler):
    bulk_log_delay = 0.0

    def _reply(self, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/").endswith("version"):
            self._reply({"api_version": 1})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        if self.path.endswith("/api/get_project_identity"):
            self._reply({"data": {"project_id": str(uuid.uuid4())}})
            return
        if self.path.endswith("/api/get_run_stop_request"):
            self._reply({"data": False})
            return
        if self.path.endswith("/api/bulk_log") and self.bulk_log_delay:
            time.sleep(self.bulk_log_delay)
        self._reply({"data": None})

    def log_message(self, *args):
        pass


def _start_fake_server(bulk_log_delay: float = 0.0) -> str:
    handler = type("Handler", (_FakeOlympusHandler,), {"bulk_log_delay": bulk_log_delay})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


def _setup_env(tmp: str, mirror: bool, server_url: str | None, tail: bool) -> None:
    os.environ["OLYMPUS_DATA_DIR"] = os.path.join(tmp, "olympus-data")
    os.environ["HF_HOME"] = os.path.join(tmp, "hf-home")
    os.environ["XDG_CONFIG_HOME"] = os.path.join(tmp, "config")  # no stored login
    for key in ("ONTIC_WANDB_RUN_ID", "OLYMPUS_SERVER_URL", "OLYMPUS_API_KEY"):
        os.environ.pop(key, None)
    if mirror:
        os.environ["ONTIC_OLYMPUS_PROJECT"] = "bench-proj"
        os.environ["ONTIC_OLYMPUS_RUN"] = f"bench-{uuid.uuid4().hex[:8]}"
    else:
        os.environ.pop("ONTIC_OLYMPUS_PROJECT", None)
        os.environ.pop("ONTIC_OLYMPUS_RUN", None)
    if server_url:
        os.environ["OLYMPUS_SERVER_URL"] = server_url
        os.environ["OLYMPUS_API_KEY"] = "bench-key"
    if not tail:
        os.environ["ONTIC_LIB_NO_LOG_TAIL"] = "1"
        os.environ.pop("ONTIC_LIB_LOG_TAIL", None)
    else:
        os.environ.pop("ONTIC_LIB_NO_LOG_TAIL", None)
        os.environ["ONTIC_LIB_LOG_TAIL"] = "1"


def _bench_log_calls(n: int, label: str, pace_s: float = 0.0):
    """Time each log() call. pace_s > 0 spreads the calls out so they overlap
    several background send cycles, the shape of a real training loop."""
    from ontic_lib import tracking

    t = tracking.init("bench")
    samples: list[int] = []
    for i in range(n):
        if pace_s:
            time.sleep(pace_s)
        start = time.perf_counter_ns()
        t.log({"loss": (i % 100) / 100.0, "acc": (i % 7) / 7.0}, step=i)
        samples.append(time.perf_counter_ns() - start)
    print(f"[{label}] log(): {_percentiles(samples)}", file=sys.stderr)
    stalls = sorted(samples, reverse=True)[:5]
    print(
        f"[{label}] top stalls: {[f'{x / 1e6:.1f}ms' for x in stalls]}",
        file=sys.stderr,
    )
    return t


def _bench_prints(label: str, n: int = N_PRINT, pace_s: float = 0.0) -> None:
    samples: list[int] = []
    for i in range(n):
        if pace_s:
            time.sleep(pace_s)
        start = time.perf_counter_ns()
        print(f"step {i}: loss=0.5 acc=0.9 lr=1e-4 grad_norm=2.34")
        samples.append(time.perf_counter_ns() - start)
    print(f"[{label}] print(): {_percentiles(samples)}", file=sys.stderr)
    stalls = sorted(samples, reverse=True)[:5]
    print(
        f"[{label}] top stalls: {[f'{x / 1e6:.1f}ms' for x in stalls]}",
        file=sys.stderr,
    )


def main() -> None:
    scenario = sys.argv[1]
    tmp = tempfile.mkdtemp(prefix=f"bench-tracking-{scenario}-")
    os.chdir(tmp)

    if scenario == "off":
        _setup_env(tmp, mirror=False, server_url=None, tail=False)
        t = _bench_log_calls(N_SCALAR, scenario)
        t.finish()
    elif scenario == "local":
        _setup_env(tmp, mirror=True, server_url=None, tail=False)
        t = _bench_log_calls(N_SCALAR, scenario)
        t.finish()
    elif scenario == "server":
        url = _start_fake_server()
        _setup_env(tmp, mirror=True, server_url=url, tail=False)
        t = _bench_log_calls(N_SCALAR, scenario)
        t.finish()
    elif scenario == "slow":
        url = _start_fake_server(bulk_log_delay=SLOW_BULK_LOG_S)
        _setup_env(tmp, mirror=True, server_url=url, tail=False)
        _bench_log_calls(N_SLOW, scenario, pace_s=0.005)
        os._exit(0)  # skip finish(): a wedged server must not stall the bench
    elif scenario == "tee_slow":
        # Prints while a slow server wedges the SDK sender: the 15 s tail flush
        # must never make print() wait for the network.
        url = _start_fake_server(bulk_log_delay=SLOW_BULK_LOG_S)
        _setup_env(tmp, mirror=True, server_url=url, tail=True)
        from ontic_lib import tracking

        devnull = open(os.devnull, "w")
        sys.stdout = devnull
        t = tracking.init("bench")

        def _keep_sender_busy():
            for i in range(10_000):
                t.log({"loss": 0.5}, step=i)
                time.sleep(0.1)

        threading.Thread(target=_keep_sender_busy, daemon=True).start()
        _bench_prints(scenario, n=20_000, pace_s=0.001)
        os._exit(0)
    elif scenario in ("tee_off", "tee_on"):
        _setup_env(tmp, mirror=True, server_url=None, tail=scenario == "tee_on")
        from ontic_lib import tracking

        devnull = open(os.devnull, "w")
        real_stdout = sys.stdout
        sys.stdout = devnull
        t = tracking.init("bench")
        _bench_prints(scenario)
        t.finish()
        sys.stdout = real_stdout
    else:
        raise SystemExit(f"unknown scenario: {scenario}")


if __name__ == "__main__":
    main()
