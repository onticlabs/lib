# ontic_data — what's in the box

`ontic_data` (`packages/ontic-data`) is the dataset half of the workspace:
loaders for temporal multi-view scenes that all return the same example dict.
Every module imports with `torch` + `numpy` + `einops` + `pyyaml`; format
readers (video decoders, h5py, pandas, PIL, cv2, mujoco) are imported lazily
and raise an `ImportError` naming the extra. Cameras follow the `ontic_lib`
conventions: camera-to-world `(4, 4)` poses, normalised intrinsics, OpenCV
axes.

## Example schema (`ontic_data.example`)

Every dataset `__getitem__` returns a `BatchedTempExample`; the collate keeps
the same keys with a leading batch dim. `BatchedTempViews` (one per `context`
/ `target`) is a `TypedDict` of `(batch, *time, view, ...)` tensors:

| key | shape | meaning |
| --- | --- | --- |
| `extrinsics` | `(B, *T, V, 4, 4)` | camera-to-world |
| `intrinsics` | `(B, *T, V, 3, 3)` | normalised `[0, 1]` |
| `image` | `(B, *T, V, 3, H, W)` | RGB in `[0, 1]` at `DatasetCfg.image_shape` |
| `depth` | `(B, *T, V, 1, H, W)` | metres (datasets with depth only) |
| `near` / `far` / `depth_is_metric` | `(B, *T, V)` | per-view scalars |
| `index` | `(B, *T, V)` int64 | camera indices |
| `state_mask` / `static_float` / `is_novel_view` | optional | per-view masks / flags |

`BatchedTempExample` adds `scene: list[str]`, `actions: {key: (B, *T,
N_copies, N_points, 4)}` (xyz + presence bit; hands, object trajectories, robot
points) and `workspace_min` / `workspace_max (B, 3)`. `to_batched_example` /
`to_batched_views` fold `batch * time` into one leading axis so consumers can
treat everything as `(batch*time, view, ...)`.

## Configs and the registry

`DatasetCfg` (`ontic_data.config`) is the stage-independent base: `image_shape`,
workspace AABB, `view_sampler: ViewSamplerCfg`, `n_step_state` /
`n_step_predict` / `val_n_step_predict`, `near` / `far`, `augment`, `speedup`,
`camera_ixs_allowed`, `consistent_cameras`, `fps`, and horizon-aware loading
(`horizon_aware_loading`, `horizon_load_margin_steps`).

`cfg.build(stage, *, step_fn=None, horizon_fn=None)` replaces the frontier
`build(stage, step_tracker)`: the global step counter and the horizon
curriculum stay with the trainer and are passed in as callables — `StepFn =
() -> int` (None = step 0) and `HorizonFn = (global_step, n_pred_full) ->
n_pred` (None = full horizon). Datasets only ever call `step_fn()` and
`horizon_fn(step, n_pred_full)`; with `horizon_aware_loading` they decode just
`n_step_state + horizon_fn(step_fn() + margin, n_pred_full)` frames while the
view sampler still runs over the full horizon so batches stay bit-exact.

`ontic_data.DATASETS` maps registry keys to config classes:

| key | config | root field(s) | data | extra(s) |
| --- | --- | --- | --- | --- |
| `genesis` | `DatasetGenesisCfg` | `roots: list[str]`, `root_depths` | synthetic scenes, per-frame image files + `metadata.json` | `images` |
| `hocap` | `HocapDatasetCfg` | `root` | 8 RealSense cams, hand labels, optional depth / masks | `hocap` (= `hdf5` + `opencv` + scipy) |
| `taco` | `TacoDatasetCfg` | `root` | 12 cams at 30 FPS, per-camera mp4s, hand / object poses | `video` (or `decord`), `tables` |
| `dextris` | `DextrisDatasetCfg` | `root` | 8 cams `P00..P07` at 60 FPS, mp4s, hand tracking | `video` (or `decord`), `tables` |
| `robot-dextris` | `RobotDextrisDatasetCfg` | `root` | DEXTRIS calibration/videos in `<root>/<sample>/`; no hand labels or default split | `opencv`, `tables` |
| `physinone` | `PhysInOneDatasetCfg` | `root` | synthetic physics scenes, static + moving camera, depth / seg | `images`, `tables` |
| `synthrobot` | `SynthRobotDatasetCfg` | `root` | bimanual `franka_duo` manipulation, h5 stores + RGB / depth mp4s | `hdf5`, `video` (or `decord`), `tables`; `robot` for `action_mode="robot_points"` |

`datasets.mixed.MixedDatasetCfg` (not in the registry) concatenates several
sub-configs keyed by label via `ConcatDataset`, normalising temporal
resolution by `fps`.

## Modules

| module | one line |
| --- | --- |
| `temporal` | `SceneView` (per-sample handle: cameras + pixel access) and `TemporalSceneDataset` (snippet indexing, context/target loop, `record_labels` / `record_n_frames` / `load_sequence_views`); `split_records` (seed-42 split), `loader_horizon`, `add_workspace`, `resize_frames` |
| `view_sampler` | `ViewSamplerCfg` (`num_context_views`, `num_target_views`, `target_includes_context`, fixed `context_views`, `paired_views` + extras) and `ViewSampler`: sorted context / target indices without replacement; `build_camera_pairs` |
| `collate` | `collate_examples` (`default_collate` except `actions`), `collate_actions` (zero-pad per key), `harmonize_target_horizon` (truncate to batch-min horizon), `worker_init_fn` |
| `shims` | Example-level transforms: `apply_augmentation_shim` (random horizontal mirror, cameras fixed up), `apply_crop_shim` (rescale + centre crop to `(H, W)`, intrinsics fixed), `apply_patch_shim` (crop to patch multiples) |
| `video` | `VideoReader(path, backend)`: random-access `(T, H, W, 3)` uint8 frames via `torchcodec` (default), `decord` or `opencv`; `has_video_backend` |
| `depth_codec` | Hue-log depth-in-RGB-video codec: `encode_depth_to_rgb`, `decode_rgb_to_depth`, `decode_rgb_to_unit_log`, `decode_unit_log_torch` (training-side), `classify` into `VALID` / `SKY` / `HOLE` |
| `robot` | `RobotKinematics` (URDF / MJCF forward kinematics), `sample_action_points` / `sample_surface_points`, `to_action_tensor`, `franka_duo_kinematics`; the robotics checkout is found as a sibling or via `ONTIC_ROBOTICS_REPO` |
| `hand` | 21-joint hand keypoints: `project_points_w2c`, `c2w_and_norm_K_to_w2c`, `draw_hand_skeleton[_alpha]`, `overlay_hand_skeletons`, `overlay_hand_trail` (cv2) |
| `datasets.*` | One module per dataset (`Dataset<Name>` + `<Name>SceneView` + the parsing helpers), plus `hocap_seq_loader.SequenceLoader` and `mixed` |

## Extras

| extra | package | enables |
| --- | --- | --- |
| `ontic-data[video]` | `torchcodec` | `VideoReader(backend="torchcodec")`: taco, dextris, synthrobot |
| `ontic-data[decord]` | `decord` (Python < 3.12) | `VideoReader(backend="decord")` |
| `ontic-data[hdf5]` | `h5py` | synthrobot trajectory stores, hocap packed frames |
| `ontic-data[opencv]` | `opencv-python-headless` | hand overlays, hocap frames, robot-dextris video decoding |
| `ontic-data[hocap]` | `hdf5` + `opencv` + `scipy` | the HO-Cap loader |
| `ontic-data[tables]` | `pandas` | `taco_info.csv`, dextris / physinone / synthrobot meta CSVs |
| `ontic-data[images]` | `pillow` | genesis, physinone frames |
| `ontic-data[robot]` | `mujoco` | `ontic_data.robot` kinematics (synthrobot `robot_points`) |
| `ontic-data[all]` | all of the above | |
