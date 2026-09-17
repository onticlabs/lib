# ontic_lib — what's in the box

`ontic_lib` is the pure-library half of the workspace: geometry, containers,
I/O, splatting, metrics and training infrastructure. It imports with the core
dependencies only (`torch`, `numpy`, `roma`, `torchmetrics`); everything else
is an explicit extra, imported lazily by the function that needs it and
raising an `ImportError` that names the extra.

All geometric APIs follow the conventions pinned in the `ontic_lib` package
docstring (`src/ontic_lib/__init__.py`): torch-first with arbitrary leading
batch dims, `(..., 3)` point rows with column-vector transforms
(`x_dst = T_dst_from_src @ x_src`), camera-to-world poses, OpenCV camera axes,
real-first `wxyz` quaternions, explicit pixel-space vs normalized intrinsics,
and explicit camera-z depth vs ray distance. Ops with a vendored CUDA kernel
take `impl="torch" | "cuda" | "auto"`.

Tables list the names in each subpackage's `__all__` (plus the submodules that
are imported explicitly). "Extra" is the `ontic-lib[...]` extra a name needs;
blank means core only.

## transforms

SO(3) representation conversions (RoMa-backed) and homogeneous SE(3)/Sim(3)
operations. Rotation "representations" are tensors whose last dim selects the
form: 4 = quaternion (`wxyz` unless `quaternion_order="xyzw"`), 6 = Zhou 6D,
9 = flattened matrix normalized by Procrustes. Transforms are `(..., 4, 4)`
matrices; camera helpers take and return camera-to-world poses.

| name | one line |
| --- | --- |
| `homogenize_points` / `homogenize_vectors` | Append `1` / `0` to `(..., 3)` rows |
| `apply_matrix` | Matrix times column vectors with broadcast leading dims |
| `apply_transform` | `(..., 4, 4)` on homogeneous `(..., 4)` coordinates |
| `transform_points` / `transform_vectors` | `(..., 4, 4)` on `(..., 3)` rows, with / without translation |
| `invert_rigid_transform` | Closed-form inverse via `R.T`, `-R.T @ t` |
| `transform_camera_to_world_se3` / `_sim3` | Apply a world-frame SE(3) / Sim(3) to c2w poses |
| `QuaternionOrder` | `Literal["wxyz", "xyzw"]` |
| `quaternion_to_matrix` / `matrix_to_quaternion` | Canonical `wxyz` ↔ matrix |
| `quaternion_xyzw_to_matrix` / `matrix_to_quaternion_xyzw` | Same for `xyzw` (SciPy / RoMa / legacy) |
| `quaternion_wxyz_to_xyzw` / `quaternion_xyzw_to_wxyz` | Component reorder |
| `rotation_6d_to_matrix` / `matrix_to_rotation_6d` | Zhou et al. 6D ↔ matrix |
| `procrustes_to_matrix` / `matrix_to_procrustes` | 9D (flat matrix, projected to SO(3)) ↔ matrix |
| `rotation_representation_to_matrix` / `matrix_to_rotation_representation` | Dispatch on 4 / 6 / 9 |
| `normalize_rotation_representation` | Re-project a representation onto its manifold |
| `increment_rotation` | Left-multiply a representation by a delta (rotvec or representation) |
| `rotation_matrix_times_representation` | `R @ rep`, returned in the same representation |
| `accumulate_rotation_vectors` | Cumulative left-composition of time-ordered rotvec increments |

## camera

Standalone pinhole cameras, not tied to any renderer. Functions say in their
name/docstring whether they take pixel-space or normalized intrinsics;
`sample_image_grid` returns normalized pixel centers `((x+0.5)/W, (y+0.5)/H)`.
`camera.occupancy` splats a point cloud back into cameras as coverage masks:
each point carries a world-space radius `depth / focal` (one-pixel footprint at
its source depth), scaled into the target camera by `splat_scale`.

| name | one line |
| --- | --- |
| `normalize_intrinsics` / `denormalize_intrinsics` | Pixel-space ↔ normalized `[0, 1]` intrinsics |
| `resize_intrinsics` | Update pixel-space intrinsics after an image resize |
| `project_camera_points` | Camera xyz → image coordinates (units follow the intrinsics) |
| `project_world_points` | World xyz through c2w pose + intrinsics → `(xy, in_front mask)` |
| `unproject_camera_points` | Image coordinates + camera-z depth → camera xyz |
| `sample_image_grid` | Normalized xy pixel centers and integer ij indices for `(H, W)` |
| `world_rays` | World-space ray origins and unit directions for image coordinates |
| `world_pixel_size` | World-space size of one pixel at unit depth (normalized intrinsics) |
| `points_with_radius` | `(V, H, W)` z-depth → world `points (N, 3)` + splat `radius (N,)`, same stride / confidence selection as `pointcloud_from_depth_views` |
| `project_occupancy` | Rasterise `(V, H, W)` bool coverage masks of `points (N, 3)` in each camera |
| `overlay_masks_on_images` | Blend a colour over masked pixels of `(V, 3, H, W)` images |

## depth

Depth lifting and metric alignment. `depth_type="z"` (camera-z) or `"ray"`
(Euclidean distance along the ray) is always chosen explicitly. No `__all__`;
import the submodules.

| name | one line |
| --- | --- |
| `lifting.depth_to_world_points` | Lift `(..., H, W)` depths through c2w poses + normalized intrinsics to world points |
| `lifting.DepthType` | `Literal["z", "ray"]` |
| `alignment.fit_depth_scale` | Robust median scale so `scale * predicted ≈ target` |
| `alignment.fit_depth_scale_and_shift` | Least-squares `(scale, shift)` for affine-invariant depth |
| `alignment.scale_depth_from_camera_poses` | Rescale depth by the Sim(3) scale between two pose sets |

## pointops

Batched tensor ops on point sets, with the *packed* layout used throughout
(`N` rows of `B` groups concatenated; `batch (N,) int64` sorted ascending,
`offset (B,)` cumulative counts, no leading zero). Pure torch is the
reference; Morton/Hilbert encoders default to `impl="auto"` because the CUDA
kernels are bit-exact, FPS defaults to `"torch"` because its kernel is not
index-stable on ties. `pointops.accel` exposes the vendored `ext/` kernels
directly (`available()`, `furthest_point_indices`, `morton_encode`,
`hilbert_encode`, `cuda_point_rope()`); see `docs/pointops_benchmarks.md`.

| name | one line |
| --- | --- |
| `voxel_pool` | Mean-pool points (+ features) in cubic voxels |
| `furthest_point_indices` / `furthest_point_sample` | FPS indices for one `(N, 3)` cloud / sample points + features |
| `space_filling_stride_indices` / `space_filling_stride` | Every `stride`-th point along a Morton/Hilbert curve |
| `SpaceFillingOrder` | `Literal["z", "z-trans", "hilbert", "hilbert-trans"]` |
| `morton_encode` / `morton_decode` | Z-order codes ↔ `(N, 3)` integer grid coords |
| `hilbert_encode` / `hilbert_decode` | Hilbert codes ↔ `(N, 3)` integer grid coords |
| `encode_grid` | One order's codes, group id packed into the high bits |
| `Serialization` | `code / order / inverse (k, N)` for `k` orders of one packed set |
| `serialize` | Build a `Serialization` for grid coords grouped by `batch` |
| `reserialize` | Swap the group bits of existing codes and re-sort |
| `pool_serialization` | Keep head-row codes with `pooling_depth` bits dropped per axis |
| `voxel_coords` | `(N, 3)` float → `(N, 3)` int32 voxel indices |
| `Clusters` | Partition of `N` rows into `M` sorted clusters: `cluster`, `counts`, `head`, `sorted_index`, `ptr` |
| `grid_clusters` / `code_clusters` | Clusters by `(batch, grid // stride)` / by serialization code |
| `cluster_reduce` | Segment `sum/mean/max/min/any` of `(N, ...)` rows into `(M, ...)` |
| `offset_to_counts` / `offset_to_batch` / `batch_to_offset` | Conversions between the packed-layout descriptors |
| `pack_padded` / `unpack_to_padded` / `scatter_to_padded` | Padded `(*G, K, ...)` + mask ↔ packed rows |
| `align_camera_poses_sim3` / `align_camera_poses_se3` | Batched Sim(3) / SE(3) fit from source to target camera poses |
| `align_points_sim3` / `align_cameras_sim3` | Apply a fitted `(R, t, s)` to points / c2w poses |
| `anchor_transform` | Transform making one source camera coincide with one target camera |
| `clamp_scale` | Replace non-finite similarity scales and clamp extremes |
| `AlignmentMode` / `ALIGNMENT_MODES` | `"none"`, `"prescale_gt"`, `"sim3_points"`, `"metric_mono"` |
| `compute_alignment` | Per-sample depth `scale (B,)` + the cameras to lift with, for a mode (not applied) |
| `apply_metric_scale` | `(B, V, H, W)` depth times a per-sample `(B,)` scale |
| `align` | `compute_alignment` + `apply_metric_scale` → `(depth, cameras, intrinsics)` |
| `percentile_conf_threshold` / `conf_drop_mask` | Backbone-agnostic "drop the lowest `drop_pct` %" of a confidence map: absolute threshold / per-sample keep mask |

## structures

Data containers built on the geometry above. `PointCloud` is a plain
`(points, colors)` dataclass with world-space construction from depth views;
`PointBatch` is the packed layout as an object (`coord`, `feat`, `batch`,
optional `time`, `extras`) and the input/output type of PTv3; `Gaussians`
holds batched 3D Gaussians with `*batch` leading dims, post-activation scales,
`wxyz` rotations, `(N, 3, d_sh)` harmonics, optional explicit covariances,
mask and extras, and supports batch indexing, `flatten_batch()`, `to()`, `replace()`
and `covariance()`.

| name | one line |
| --- | --- |
| `PointCloud` | `points (..., N, 3)` + optional `colors` |
| `pointcloud_from_depth_views` | Unproject `(V, H, W)` depth maps into one world-space cloud |
| `aabb_mask` / `crop_to_aabb` | Inclusive axis-aligned box test / filter points + features |
| `padded_aabb` | `(minimum, maximum)` of the box spanning `(..., 3)` points, padded by `margin` |
| `nearest_point_to_ray` | Closest point to a ray, optionally within `maximum_distance` |
| `PointBatch` | Packed points; `from_padded(...)`, `offset`, `counts`, `fields()`, `replace(...)` |
| `Gaussians` | Batched Gaussians; `batch_shape`, `num_gaussians`, `items()`, `covariance()`, `flatten_batch()` |

## io

`save_gaussians(path, g, metadata=...)` / `load_gaussians(path, device=...)`,
format picked by suffix. `.npz` is core and round-trips exactly (including
mask, extras, covariances, batch dims); `.safetensors` is exact too; `.ply`
writes the standard 3DGS vertex layout (`f_dc`, `f_rest`, log-scales,
logit-opacities, `wxyz` rot) and is unbatched with no mask/extras/covariances.

| name | one line | extra |
| --- | --- | --- |
| `save_gaussians` | Write `Gaussians` (+ string metadata) by suffix | `safetensors` / `ply` for those suffixes |
| `load_gaussians` | Read `Gaussians` by suffix onto `device` (default CPU) | same |

## splats

3D Gaussian-splatting helpers. `render_gaussians` renders one scene into `V`
pinhole views through gsplat, taking package-convention cameras (c2w poses,
normalized intrinsics) and converting internally; consecutive cameras sharing
`(near, far)` are batched into one call. `rasterizer="cute"` routes to
`splats.cute`, a CuTeDSL forward kernel with gsplat's exact backward that
batches `B` scenes x `C` cameras into one launch (post-activation colors only,
uniform near/far, needs a CUDA-13-era driver). Both are CUDA-only.

| name | one line | extra |
| --- | --- | --- |
| `covariance_from_scale_rotation` | `R diag(s²) Rᵀ` from scales + matrices | |
| `covariance_from_scale_rotation_representation` | Same from a 4/6/9-D rotation representation | |
| `direction_to_angles` / `rotation_matrix_to_angles` | Unit direction → `(alpha, beta)` / matrix → `(alpha, beta, gamma)` e3nn angles | |
| `rotate_spherical_harmonics` (`rotate_sh`) | Rotate complete real-SH bands by rotation matrices | `e3nn` |
| `render_gaussians` / `RenderOutput` | gsplat rasterization → channels-first RGB / depth / alpha | `gsplat` |
| `cute.batched_render` / `cute.cute_rasterize` | Batched CuTeDSL rasterizer (submodule, not in `__all__`) | `cute` |

## metrics

Evaluation protocols and aggregation, kept separate from geometry and losses.
The top level exports only `MetricsAccumulator` and `psnr`; import
`metrics.image` and `metrics.particles3d` explicitly.

| name | one line | extra |
| --- | --- | --- |
| `MetricsAccumulator` | Weighted running mean over per-step metric dicts (`add`, `mean`, `reset`) | |
| `psnr` | Scalar PSNR (dB) as a Python float; `inf` for identical inputs | |
| `image.scalar_psnr` / `image.compute_psnr_values` | Tensor PSNR, scalar / per-image with optional masks | |
| `image.compute_ssim_values` | Per-image SSIM (torchmetrics by default, or a custom model) | |
| `image.compute_lpips_values` | Per-image LPIPS through a caller-supplied model, chunked | caller's LPIPS model |
| `particles3d.*` | Particle world-model metrics: `to_u8`, `psnr_per_step`, object/plate splitting (`split_obj_plate`, `cluster_obj_plate_3d`, `otsu_zcut`, `radius_components`), `dissolution`, `alpha_spread`, `render_dissolution` | |

## tracking / checkpoint / distributed

Training infrastructure at the top level, all framework-agnostic.

| name | one line | extra |
| --- | --- | --- |
| `tracking.init(project, config)` → `Tracker` | Append one JSON line per `log()` to `./output/metrics.jsonl` (fsync-coalesced); mirrors to W&B best-effort iff `ONTIC_WANDB_RUN_ID` and credentials are set | `wandb` (optional) |
| `tracking.read_metrics` | Decode a `metrics.jsonl`, skipping a truncated last line | |
| `checkpoint.CheckpointManager` | Atomic `save(step, state)`, `resume()`, `latest_step`, `keep_last` pruning; pluggable `save_fn` / `load_fn` (default pickle) | |
| `distributed.avg_log_dict_across_ranks` | Hang-safe DDP average of a metric dict (key intersection only, one batched all-reduce) | |
