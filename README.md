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

The former `ontic_lib.image_metrics` and `ontic_lib.metrics3d` import paths
remain available as compatibility shims.

## Geometry

`ontic_lib.geometry` is convention-locked tensor geometry for 3D/4D work
(depth and reconstruction models, world models). The conventions are pinned
in the package docstring and every function conforms: torch-first with
arbitrary leading batch dims, `(..., 3)` point rows with column-vector
transforms (`x_dst = T_dst_from_src @ x_src`), camera-to-world poses,
OpenCV camera axes (+x right, +y down, +z forward), real-first `(w, x, y, z)`
quaternions (explicit `xyzw` converters exist), and an explicit distinction
between pixel-space and normalized intrinsics and between camera-z depth and
ray distance.

- `geometry.rotations` — differentiable SO(3) conversions: matrix ↔
  quaternion (wxyz/xyzw), 6D, Procrustes (9D), generic representation
  helpers, rotation-vector accumulation.
- `geometry.transforms` — homogeneous SE(3)/Sim(3) ops: homogenize, apply,
  invert, transform points/vectors/cameras.
- `geometry.cameras` — pinhole intrinsics (normalize/denormalize/resize),
  project/unproject, image grids, world rays, world-space pixel size.
- `geometry.alignment` — Umeyama-style SE(3)/Sim(3) alignment of camera
  trajectories and point sets, anchor transforms.
- `geometry.pointclouds` — `PointCloud` container plus AABB crop, voxel
  pooling, furthest-point sampling, space-filling-curve striding,
  depth-views → point-cloud construction.
- `geometry.space_filling` — Morton and Hilbert codes for integer grids
  (vendored kernels in `_z_order` / `_hilbert`).
- `geometry.gaussians` — covariance construction for anisotropic 3D
  Gaussians (splatting heads).
- `geometry.spherical_harmonics` — rotation of real-SH coefficient bands
  (requires the `e3nn` extra: `pip install ontic-lib[e3nn]`).

## Depth

- `depth.lifting` — lift depth maps through pinhole cameras into world-space
  points (`depth_type="z"` or `"ray"`, chosen explicitly).
- `depth.alignment` — least-squares scale (and scale+shift) fitting of
  predicted depth against a reference, and rescaling depth into a target
  camera-pose frame — the core of metric alignment for affine/scale-invariant
  depth backbones.

## Distributed

`ontic_lib.distributed.avg_log_dict_across_ranks` — hang-safe averaging of a
training `log_dict` across ranks: no-op on single-rank, reduces only the
key intersection (rank-divergent keys can never deadlock NCCL), one batched
all-reduce for scalars, never mutates the caller's dict.
