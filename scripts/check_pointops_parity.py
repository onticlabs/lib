"""Empirical parity check: ontic_lib.pointops torch reference vs CUDA kernels.

Measures, on a live GPU, how close the implementations actually are:

  * Morton / Hilbert — exact value equality over random + edge-case grids
  * hilbert_encode_approx — divergence rate vs the exact kernel
  * FPS — fraction of trials with identical index sequences, and the quality
    gap (max min-distance spread) when they differ
  * 3D RoPE — max abs / rel error of the fused kernel vs the torch module

Run like the benchmark:  PYTHONPATH=src:<ext-target> python scripts/check_pointops_parity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent


def _try(name):
    try:
        return __import__(name)
    except Exception:
        return None


def check_serialization():
    from ontic_lib.pointops.serialization import hilbert_encode, morton_encode

    ser = _try("serialize_cuda")
    if ser is None or not torch.cuda.is_available():
        print("serialization: kernels or GPU unavailable — skipped")
        return
    print("Serialization (values, torch.equal against reference):")
    g = torch.Generator().manual_seed(0)
    for depth in (8, 12, 16):
        hi = 2**depth
        rand = torch.randint(0, hi, (200_000, 3), generator=g, dtype=torch.int64)
        edges = torch.tensor(
            [[0, 0, 0], [hi - 1, hi - 1, hi - 1], [hi - 1, 0, 0], [0, hi - 1, 0],
             [0, 0, hi - 1], [1, 2, 3], [hi // 2, hi // 2, hi // 2]]
        )
        grid = torch.cat([rand, edges]).cuda()
        grid32 = grid.to(torch.int32).contiguous()
        # .to(int64) also covers the raw C bindings, which return uint64
        m_eq = torch.equal(
            morton_encode(grid, depth=depth), ser.morton_encode(grid32).to(torch.int64)
        )
        h_eq = torch.equal(
            hilbert_encode(grid, depth=depth), ser.hilbert_encode(grid32, depth).to(torch.int64)
        )
        approx = ser.hilbert_encode_approx(grid32, depth).to(torch.int64)
        exact = ser.hilbert_encode(grid32, depth).to(torch.int64)
        div = (approx != exact).float().mean().item()
        o_eq = torch.equal(
            torch.argsort(approx, stable=True), torch.argsort(exact, stable=True)
        )
        print(
            f"  depth={depth:2d}: morton exact={m_eq}  hilbert exact={h_eq}  "
            f"| approx-kernel: {div:6.2%} values differ, same order={o_eq}"
        )


def check_fps():
    from ontic_lib.pointops import furthest_point_indices

    ops = _try("pointops")
    if ops is None or not torch.cuda.is_available():
        print("FPS: kernel or GPU unavailable — skipped")
        return
    print("\nFPS (index-sequence equality vs reference, 20 seeds each):")
    for n, k in ((512, 64), (4096, 512), (32768, 4096)):
        same = 0
        worst_gap = 0.0
        for seed in range(20):
            g = torch.Generator().manual_seed(seed)
            pts = torch.rand(n, 3, generator=g).cuda()
            ref = furthest_point_indices(pts, k)
            offset = torch.tensor([n], dtype=torch.int32, device="cuda")
            new_offset = torch.tensor([k], dtype=torch.int32, device="cuda")
            cud = ops.farthest_point_sampling(pts.contiguous(), offset, new_offset).long()
            if torch.equal(ref, cud):
                same += 1
            else:

                def spread(idx):
                    d = torch.cdist(pts[idx], pts[idx])
                    d.fill_diagonal_(float("inf"))
                    return d.min(dim=1).values.min().item()

                worst_gap = max(worst_gap, abs(spread(ref) - spread(cud)))
        note = "" if same == 20 else f"  (max min-spread gap when differing: {worst_gap:.2e})"
        print(f"  N={n:<6} K={k:<5}: {same}/20 identical{note}")


def check_rope():
    sys.path.append(str(REPO / "ext" / "point_rope"))
    rope_cuda = _try("pointrope_cuda")
    if rope_cuda is None or not torch.cuda.is_available():
        print("\nRoPE: kernel or GPU unavailable — skipped")
        return
    from pointrope_torch import Point3DRoPE  # noqa: PLC0415

    print("\n3D RoPE (fused kernel vs torch module):")
    module = Point3DRoPE(head_dim=48).cuda()
    for n in (1_000, 100_000):
        g = torch.Generator().manual_seed(0)
        q = torch.rand(n, 8, 48, generator=g).cuda()
        k = torch.rand(n, 8, 48, generator=g).cuda()
        coord = (torch.rand(n, 3, generator=g).cuda() - 0.5) * 4
        q_t, k_t = module(q, k, coord)
        q_c, k_c = rope_cuda.apply_rope(q, k, coord, module.base, module.F0)
        abs_err = max((q_t - q_c).abs().max().item(), (k_t - k_c).abs().max().item())
        denom = q_t.abs().clamp_min(1e-6)
        rel_err = ((q_t - q_c).abs() / denom).max().item()
        print(f"  N={n:<7}: max abs err {abs_err:.2e}   max rel err {rel_err:.2e}")


if __name__ == "__main__":
    print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
    check_serialization()
    check_fps()
    check_rope()
