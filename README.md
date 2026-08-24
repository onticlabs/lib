# ontic-lib

Shared, promoted code for Ontic experiments.

## What belongs here

`ontic-lib` is for code that has proven itself across experiments, not for
one-off experiment logic. Concretely:

- **Eval harnesses** — scoring/benchmark code reused across multiple
  experiments or projects.
- **Stable model code** — architectures, layers, or wrappers that have
  stabilized and are no longer being actively iterated on inside a single
  experiment.
- **Dataset loaders** — data loading/preprocessing code shared by more than
  one experiment.
- **Cross-cutting infrastructure**, such as the tracking shim in
  `ontic_lib.tracking`, which fans out metric logging to
  [trackio](https://pypi.org/project/trackio/) (always) and Weights & Biases
  (best-effort, optional) so training runs never crash because a metrics
  SaaS is down.

## The promotion rule

Code starts life inside an experiment's own directory (scaffolded by
`ontic-cli`). It only gets promoted into `ontic-lib` once it has been
**copied into a third experiment** — i.e. it has proven itself reusable at
least twice over. Until then, duplication across two experiments is fine
and expected; premature abstraction is not.

When code is promoted:

1. Move it into `src/ontic_lib/` here, with tests in `tests/`.
2. Update the experiments that copied it to import from `ontic_lib` instead
   of keeping their own copy.
3. Bump the version and tag a release so experiments can pin against it.

## Tracking shim

```python
from ontic_lib.tracking import init

tracker = init("my-project", config={"lr": 0.1})
tracker.log({"loss": 1.0}, step=1)
tracker.finish()
```

- Always logs to `trackio`. `TRACKIO_DIR` defaults to `./output` (relative
  to the current working directory) so the metrics DB lands inside the job
  record; set `TRACKIO_DIR` explicitly beforehand to override.
- Additionally logs to Weights & Biases iff all of the following hold:
  - `wandb` is importable (install the `wandb` extra: `pip install
    ontic-lib[wandb]`),
  - `WANDB_API_KEY` is set or a `~/.netrc` file exists (W&B credentials are
    present), and
  - `ONTIC_WANDB_RUN_ID` is set (the bootstrap sets this from
    `cert.wandb.run_id` so the run resumes by id rather than starting a new
    one).
- Every W&B call (`init`, `log`, `finish`) is wrapped so a W&B/metrics-SaaS
  failure can never crash a training run.

## Checkpointing

```python
from ontic_lib.checkpoint import CheckpointManager

mgr = CheckpointManager("./output/checkpoints", keep_last=3)

# in the training loop
mgr.save(step, {"model": model.state_dict(), "opt": opt.state_dict()},
         run_id=run_id, keep_last=3)

# on resume
result = mgr.resume()
if result is not None:
    state, meta = result
    model.load_state_dict(state["model"])
    start_step = meta["step"]
```

- Backend-agnostic: serialization defaults to `pickle`, but any framework can
  plug in its own format, e.g. `CheckpointManager(dir, save_fn=torch.save,
  load_fn=torch.load)`.
- Writes are atomic (temp file + `os.replace`) so a crash mid-save never
  leaves a corrupt checkpoint behind.
- A `latest.json` pointer file makes `resume()` and `latest_step` O(1) and
  independent of the serialization format.
- `keep_last` prunes older `step_*.ckpt` files, always keeping the one just
  written.

## Metrics

```python
from ontic_lib.metrics import MetricsAccumulator, psnr
from ontic_lib.metrics.image import compute_psnr_values

score = psnr(prediction, target, max_val=1.0)
batch_score = compute_psnr_values(batch["target"], prediction)

acc = MetricsAccumulator()
for batch in loader:
    acc.add({"loss": loss.item(), "psnr": psnr(pred, batch)}, n=batch_size)
print(acc.mean())
acc.reset()
```

- `psnr(a, b, max_val=1.0)`: scalar, torch-backed peak signal-to-noise ratio
  in dB between two tensor-like inputs; returns `inf` for identical inputs.
- `ontic_lib.metrics.image`: per-image torch helpers for PSNR, SSIM, and
  LPIPS-style model outputs.
- `ontic_lib.metrics.particles3d`: particle-cloud and rendered-rollout metrics.
- `MetricsAccumulator`: a weighted running mean over per-step metric dicts —
  handy for averaging batch metrics of varying batch size over an epoch.

## Geometry

The geometric packages follow the function-type split used by PyTorch3D and
Kaolin. The conventions are pinned in the `ontic_lib` package docstring and
every function conforms: torch-first with arbitrary leading batch dims,
`(..., 3)` point rows with column-vector transforms
(`x_dst = T_dst_from_src @ x_src`), camera-to-world poses, OpenCV camera axes
(+x right, +y down, +z forward), real-first `(w, x, y, z)` quaternions
(explicit `xyzw` converters exist), and an explicit distinction between
pixel-space and normalized intrinsics and between camera-z depth and ray
distance.

- `transforms` — SO(3) representation conversions (`transforms.rotations`,
  RoMa-backed: matrix ↔ quaternion wxyz/xyzw, 6D, Procrustes, generic
  representation helpers, rotation-vector accumulation) and homogeneous
  SE(3)/Sim(3) ops (`transforms.rigid`: homogenize, apply, invert, transform
  points/vectors/cameras).
- `camera` — standalone pinhole cameras: `camera.intrinsics`
  (normalize/denormalize/resize), `camera.projection` (project/unproject,
  image grids), `camera.rays` (world rays, world-space pixel size).
- `pointops` — batched tensor ops: `pointops.sampling` (voxel pooling, furthest-point
  sampling, space-filling-curve striding), `pointops.serialization` (Morton and
  Hilbert codes for integer grids), `pointops.alignment` (Umeyama-style
  SE(3)/Sim(3) alignment of camera trajectories and point sets),
  `pointops.pointcloud` (`PointCloud` container, AABB crop, depth-views →
  point-cloud construction).
- `splats` — 3D Gaussian-splatting helpers: `splats.gaussians` (covariance
  construction), `splats.sh` (rotation of real-SH coefficient bands; `e3nn`
  extra), and `splats.rendering` (`render_gaussians`: one scene into `V`
  pinhole views through gsplat, taking package-convention cameras — c2w poses
  + normalized intrinsics — with SH or post-activation colors, per-view
  near/far grouped into batched calls, Gaussian masking, and RGB/depth/alpha
  outputs; `gsplat` extra, CUDA only).

## Depth

- `depth.lifting` — lift depth maps through pinhole cameras into world-space
  points (`depth_type="z"` or `"ray"`, chosen explicitly).
- `depth.alignment` — least-squares scale (and scale+shift) fitting of
  predicted depth against a reference, and rescaling depth into a target
  camera-pose frame — the core of metric alignment for affine/scale-invariant
  depth backbones.

## CUDA extensions (optional)

`ext/` vendors three separately-installable CUDA extension packages used by
point-transformer-style models: `pointops` (KNN/ball query, grouping,
sampling, aggregation kernels), `point_rope` (rotary position embeddings for
point tokens, with a pure-torch fallback module), and `point_serialization`
(GPU Morton/Hilbert encoding). They need `nvcc` and an installed `torch`:

```bash
./scripts/install_cuda_ext.sh               # all three
./scripts/install_cuda_ext.sh pointops      # one at a time
```

The core library never imports them automatically: `ontic_lib.pointops` is
the pure-torch reference implementation, and `ontic_lib.pointops.accel`
exposes the installed kernels as an explicit opt-in —

```python
from ontic_lib.pointops import accel

accel.available()   # {"pointops": True, "point_rope": True, "point_serialization": True}
idx = accel.furthest_point_indices(points_cuda, 4096)   # pointops FPS kernel
codes = accel.hilbert_encode(grid_cuda, depth=16)       # serialize_cuda kernel
rope = accel.cuda_point_rope()                          # module handle (Point3DRoPE)
```

The kernels run only on CUDA tensors and are not bitwise-identical to the
reference (FPS tie-breaking, Hilbert curve variant), so call sites choose
explicitly; nothing switches silently based on what is installed.

## Distributed

`ontic_lib.distributed.avg_log_dict_across_ranks` — hang-safe averaging of a
training `log_dict` across ranks: no-op on single-rank, reduces only the
key intersection (rank-divergent keys can never deadlock NCCL), one batched
all-reduce for scalars, never mutates the caller's dict.
