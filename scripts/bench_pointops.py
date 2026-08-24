"""Benchmark ontic_lib.pointops torch reference vs the vendored CUDA kernels.

Compares, across batch counts and cloud sizes:
  * FPS           — reference (CPU / GPU) vs pointops.farthest_point_sampling
  * Morton codes  — reference (CPU / GPU) vs serialize_cuda.morton_encode
  * Hilbert codes — reference (CPU / GPU) vs serialize_cuda.hilbert_encode
  * 3D RoPE       — pure-torch Point3DRoPE vs the fused point_rope_cuda kernel

Run with a python whose torch can execute on the local GPU, e.g.:

  PYTHONPATH=src:<extra> python scripts/bench_pointops.py

The CUDA packages are found via normal imports; missing ones are skipped.
Timing is best-of-N wall clock with cuda synchronization.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


def _try(name):
    try:
        return __import__(name)
    except Exception:
        return None


def timeit(fn, *, warmup=3, iters=10, sync=True):
    for _ in range(warmup):
        fn()
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        if sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e3  # ms


def fmt(ms):
    return "      —" if ms is None else f"{ms:9.2f}"


def row(label, cpu, gpu, cuda):
    speed = ""
    if cuda is not None and (gpu or cpu):
        base = gpu if gpu is not None else cpu
        speed = f"  x{base / cuda:7.1f}"
    print(f"  {label:<24}{fmt(cpu)}{fmt(gpu)}{fmt(cuda)}{speed}")


def bench_fps():
    from ontic_lib.pointops import furthest_point_indices

    ops = _try("pointops")
    print("\nFPS  (batch × N points -> K = N/8 samples)          torch-cpu  torch-gpu   cuda-ker   cuda vs gpu")
    for batch in (1, 4, 16):
        for n in (1024, 8192, 65536):
            k = n // 8
            g = torch.Generator().manual_seed(0)
            clouds = [torch.rand(n, 3, generator=g) for _ in range(batch)]
            cpu = (
                timeit(lambda: [furthest_point_indices(c, k) for c in clouds], warmup=1, iters=3)
                if n <= 8192
                else None
            )
            gpu = cuda = None
            if torch.cuda.is_available():
                gclouds = [c.cuda() for c in clouds]
                gpu = timeit(lambda: [furthest_point_indices(c, k) for c in gclouds], iters=5)
                if ops is not None:
                    packed = torch.cat(gclouds).contiguous()
                    offset = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32) * n
                    new_offset = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32) * k
                    cuda = timeit(
                        lambda: ops.farthest_point_sampling(packed, offset, new_offset), iters=10
                    )
            row(f"B={batch:<3} N={n:<7} K={k}", cpu, gpu, cuda)


def bench_serialization():
    from ontic_lib.pointops.serialization import hilbert_encode, morton_encode

    ser = _try("serialize_cuda")
    for name, ref, cuda_fn in (
        ("Morton", morton_encode, None if ser is None else ser.morton_encode),
        ("Hilbert", hilbert_encode, None if ser is None else (lambda c: ser.hilbert_encode(c, 16))),
    ):
        print(f"\n{name}  (N int grid coords, depth 16)              torch-cpu  torch-gpu   cuda-ker   cuda vs gpu")
        for n in (10_000, 100_000, 1_000_000):
            g = torch.Generator().manual_seed(0)
            grid = torch.randint(0, 2**16, (n, 3), generator=g, dtype=torch.int64)
            cpu = timeit(lambda: ref(grid, depth=16), iters=5)
            gpu = cuda = None
            if torch.cuda.is_available():
                ggrid = grid.cuda()
                gpu = timeit(lambda: ref(ggrid, depth=16), iters=10)
                if cuda_fn is not None:
                    ggrid32 = ggrid.to(torch.int32).contiguous()
                    cuda = timeit(lambda: cuda_fn(ggrid32), iters=10)
            row(f"N={n:<9}", cpu, gpu, cuda)


def bench_rope():
    # append, not prepend: an installed point_rope_cuda (with built _C) must win
    # over the source tree's package dir of the same name
    sys.path.append(str(REPO / "ext" / "point_rope"))
    try:
        from pointrope_torch import Point3DRoPE  # noqa: PLC0415
    except Exception:
        print("\nRoPE: pointrope_torch not importable — skipped")
        return
    rope_cuda = _try("pointrope_cuda")
    if not torch.cuda.is_available():
        print("\nRoPE: no CUDA device — skipped")
        return
    heads, dim = 8, 48
    module = Point3DRoPE(head_dim=dim).cuda()
    print("\n3D RoPE  (N tokens, 8 heads, head_dim 48)           torch-cpu  torch-gpu   cuda-ker   cuda vs gpu")
    for n in (10_000, 100_000, 500_000):
        g = torch.Generator().manual_seed(0)
        q = torch.rand(n, heads, dim, generator=g).cuda()
        k = torch.rand(n, heads, dim, generator=g).cuda()
        coord = torch.rand(n, 3, generator=g).cuda()
        gpu = timeit(lambda: module(q, k, coord), iters=10)
        cuda = None
        if rope_cuda is not None:
            cuda = timeit(
                lambda: rope_cuda.apply_rope(q, k, coord, module.base, module.F0), iters=10
            )
        row(f"N={n:<9}", None, gpu, cuda)


if __name__ == "__main__":
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}", end="")
    if torch.cuda.is_available():
        print(f" | {torch.cuda.get_device_name(0)}")
    else:
        print()
    print("times in ms, best-of-N; '—' = not applicable / skipped")
    bench_fps()
    bench_serialization()
    bench_rope()
