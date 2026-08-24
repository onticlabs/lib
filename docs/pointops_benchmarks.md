# pointops: torch reference vs CUDA kernels — benchmarks

Speed comparison between the pure-torch reference implementations in
`ontic_lib.pointops` and the vendored CUDA kernels in `ext/`
(`pointops`, `serialize_cuda`, `point_rope_cuda`), measured to decide where
CUDA dispatch is worth it (see the drop-in analysis at the bottom).

Reproduce with `scripts/bench_pointops.py`.

## Environment

- **GPU**: NVIDIA RTX A5000 (24 GB, sm_86), idle during measurement
- **torch**: 2.4.1+cu124 (Python 3.10) — kernels built with nvcc 12.4,
  `TORCH_CUDA_ARCH_LIST=8.6`
- **Timing**: best-of-N wall clock with `torch.cuda.synchronize()`, warmup
  discarded; milliseconds
- **Date / lib state**: 2026-08-24, branch `feat/geometry-depth` (`b4e9e72` + bench)

## FPS — furthest-point sampling (K = N/8)

The torch reference is a per-cloud Python loop of K greedy steps; the CUDA
kernel handles the whole batch in one launch (offset-packed).

| batch | N | K | torch CPU | torch GPU | CUDA kernel | kernel vs torch-GPU |
|--:|--:|--:|--:|--:|--:|--:|
| 1 | 1 024 | 128 | 5.4 | 7.5 | **0.25** | 30× |
| 1 | 8 192 | 1 024 | 156.0 | 62.7 | **3.4** | 18× |
| 1 | 65 536 | 8 192 | — | 553.8 | **153.1** | 3.6× |
| 4 | 1 024 | 128 | 21.6 | 31.3 | **0.32** | 97× |
| 4 | 8 192 | 1 024 | 631.7 | 246.3 | **3.6** | 68× |
| 4 | 65 536 | 8 192 | — | 2 280.6 | **154.5** | 15× |
| 16 | 1 024 | 128 | 85.8 | 125.2 | **0.63** | 200× |
| 16 | 8 192 | 1 024 | 2 549.6 | 1 018.2 | **4.1** | 250× |
| 16 | 65 536 | 8 192 | — | 8 959.5 | **292.2** | 31× |

The reference's cost is dominated by K sequential steps × B sequential clouds
(kernel-launch latency on GPU, vector passes on CPU); the kernel's batch
parallelism makes the gap grow with batch count. At training-relevant scales
(B≥4, N≥8k) the kernel is **1–3 orders of magnitude faster** and turns a
250 ms–9 s operation into 3–300 ms.

## Morton codes (depth 16)

| N | torch CPU | torch GPU | CUDA kernel | kernel vs torch-GPU |
|--:|--:|--:|--:|--:|
| 10 000 | 0.45 | 0.21 | **0.02** | 10× |
| 100 000 | 0.65 | 0.23 | **0.02** | 10× |
| 1 000 000 | 12.0 | 0.77 | **0.08** | 10× |

The torch reference is already fast (LUT-based, sub-ms up to 1M points). The
kernel's 10× is real but saves at most ~0.7 ms — only worth routing in hot
inner loops.

## Hilbert codes (depth 16)

| N | torch CPU | torch GPU | CUDA kernel | kernel vs torch-GPU |
|--:|--:|--:|--:|--:|
| 10 000 | 24.2 | 7.6 | **0.02** | 347× |
| 100 000 | 87.2 | 7.9 | **0.02** | 362× |
| 1 000 000 | 1 713.6 | 53.0 | **0.11** | 470× |

**The standout result.** The reference (bit-array port of numpy-hilbert-curve)
does ~num_bits×num_dims tensor passes; the kernel does the whole transform in
registers per point. 8–53 ms → 0.02–0.11 ms, flat in N until memory-bound.
This dominates `space_filling_stride(order="hilbert")` cost.

## 3D RoPE (8 heads × head_dim 48)

Forward-only fused kernel vs the pure-torch `Point3DRoPE` module (both GPU).

| N tokens | torch module | CUDA kernel | speedup |
|--:|--:|--:|--:|
| 10 000 | 0.58 | **0.25** | 2.3× |
| 100 000 | 5.03 | **2.16** | 2.3× |
| 500 000 | 24.85 | **10.58** | 2.3× |

Steady 2.3× from fusing the cos/sin table + rotate-half chain into one kernel.
Note the kernel path is forward-only w.r.t. coordinates; training with coord
gradients must use the torch path.

## Measured parity (`scripts/check_pointops_parity.py`, same A5000 env)

- **Morton / Hilbert (exact kernel)**: `torch.equal` against the reference
  over 200k random + edge-case coords at depths 8/12/16 — **bit-exact in
  every case**, confirming the earlier source-level transliteration proof at
  execution level.
- **`hilbert_encode_approx`**: **100% of values differ AND the sort order
  differs** from the exact kernel — it is a different curve, not a faster
  equivalent. Never mix approx codes with reference/exact codes.
- **FPS**: index sequences identical to the reference in 40/40 trials at
  N≤4096; at N=32768/K=4096, **14/20 identical** — and in every diverging
  trial the min-spread quality gap was exactly 0. So: quality-identical,
  not index-stable at scale (float ties + reduction order).
- **RoPE**: fused kernel vs torch module — max abs error 1.2e-7 (float32
  epsilon level); relative error up to ~1e-2 only on near-zero outputs.

## Dispatch (implemented in `ontic_lib.pointops`)

Functions with a kernel counterpart take `impl="torch" | "cuda" | "auto"`:

| op | default | rationale |
|---|---|---|
| `morton_encode` / `hilbert_encode` / `encode_grid` / `space_filling_stride*` | **`"auto"`** | bit-exact — routing can only change speed (10× / 350–470×), never results |
| `furthest_point_indices` / `furthest_point_sample` | **`"torch"`** | 15–250× available, but not index-stable on ties — opt into `"cuda"`/`"auto"` per call site |
| RoPE | module flag | already switchable via `Point3DRoPE(use_cuda=True)`, forward-only |

`"cuda"` demands a CUDA tensor and the installed extension (raises otherwise);
`"auto"` routes exactly when both are available and falls back to torch.
