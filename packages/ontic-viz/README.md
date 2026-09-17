# ontic-viz

Interactive tools over `ontic-nn` (backbone + metric-depth wrappers) and `ontic-data`
(dataset loaders). Currently one app: the **backbone 3D viewer**, a headless
[viser](https://viser.studio) web GUI that runs any registered depth backbone on any
registered dataset and shows the lifted point cloud next to the GT cameras.

## Running

```bash
uv run ontic-backbone-viewer --port 8080 --device cuda
# from your laptop:
ssh -L 8080:localhost:8080 <host>        # then open http://localhost:8080

# ...or get a public share.viser.studio URL to open from any device (no SSH):
uv run ontic-backbone-viewer --share
```

viser walks up from the requested port when it is taken; trust the printed
`serving on port N` line. With `--share`, the tunnel is kept alive across unclean TLS
teardowns and re-issued (new URL, printed as a `SHARE URL:` line) when it drops, so a
viewer left in tmux stays reachable (`grep "SHARE URL" log | tail -1`).

Flags: `--host`, `--port`, `--device cuda|cuda:N|cpu`, `--stage train|val|test`,
`--share`, `--presets <json>`, and one `--<dataset>-root <path>` per registered
dataset. Roots default to the dev-box paths in `data_source.DEFAULT_ROOTS`
(`/mnt/fast/...`); override them on other machines, e.g.
`--genesis-root /fast/mzhobro/datasets/soft_genesis_elastic`.

Presets (named `ViewConfig`s: stride / drop-conf% / voxel / SFC / FPS / point size /
colour / alignment mode / workspace box) live in
`~/.config/ontic/backbone_viewer_presets.json` (override with `--presets` or
`ONTIC_BACKBONE_VIEWER_PRESETS`). A `<dataset>_default` entry is seeded per dataset only
when missing and never overwritten, so tuned values persist; **Load dataset** applies it.

The robot overlay (SynthRobot) needs `mujoco` (`ontic-viz[robot]`) and the sibling
`onticlabs/robotics` checkout with its duobench submodule
(`git submodule update --init robots/franka_duo/vendor/duobench`; point
`$ONTIC_ROBOTICS_REPO` at it if it is not a sibling). Missing pieces untick the box and
say which.

## Workflow

1. Pick **Backbone** and **Dataset**, press **Load dataset**, choose a **Trajectory** and
   scrub **Timestep**. Every camera shows as a "use cam" checkbox and as an image frustum
   at its GT pose (green = selected as input, grey = not); click a frustum to toggle it.
2. Optionally tick **Condition model on GT cameras** (a model *input*: only backbones that
   ingest cameras — DA3 / MA / Pi3X / gtdepth — use it; greyed out otherwise), choose a
   **Metric alignment**, press **Run inference**. **Input long side** overrides the
   backbone's canonical input resolution (0 = default); it needs a re-run.
3. Everything else re-renders from the cached result: stride, **drop lowest-conf %**
   (a percentile, so it means the same across backbones), voxel size, SFC stride,
   colour mode, alignment, point size. **Apply FPS** furthest-point-samples to the
   slider budget on demand (too slow to be live).
4. **Project -> occupancy** reprojects the cloud into the cameras and tints covered
   pixels on the frustums (pred-camera coverage on the orange predicted-pose frustums,
   GT-camera coverage on the GT frustums when the cloud is aligned to GT).
5. **Ruler**: tick, click two cloud points (each click snaps an endpoint onto the
   nearest point along the click ray), read the distance; drag the markers to adjust.
6. **Hands & workspace**: hand skeletons where the dataset has them; the workspace box
   follows the hands (or the camera rig when there are none) while **Crop to
   workspace** is off and stays fixed once it is on. **Frame scene** points the browser
   camera at the rig (done automatically on trajectory change).

## Alignment modes

Backbones emit **raw** (predicted-scale) depth plus, when they have a camera head,
predicted cameras. The runner keeps both camera sets so the metric alignment is a
downstream, re-renderable choice, implemented once in `ontic_lib.pointops.align`
(the same code path the training pipeline uses):

| Mode | Needs | What it does |
|---|---|---|
| `none` | — | Unproject in the backbone's own frame. Will not overlap GT. |
| `sim3_points` | pred + GT cams (>= 2) | Unproject with the **predicted** cameras, then map the cloud into the GT frame with a Umeyama Sim(3) fit of predicted -> GT cameras. Keeps the model's multi-view consistency (the VGGT/CUT3R/SLAM-ATE protocol). **Default.** |
| `prescale_gt` | pred + GT cams | Take only the scale of that Sim(3), rescale depth, unproject with the **GT** cameras. Pixel-accurate reprojection into GT (what training does) at the cost of forcing predicted depth through GT rays. |
| `metric_mono` | a metric model | Fit one global scale of the depth to a **metric monocular** model (median of ratios over the confident pixels; `Metric model` dropdown, built lazily), rescale depth **and the predicted camera centres**, then place the reconstruction by a rigid SE(3) fit to the GT cameras over all views (centred on the cloud mean without GT cameras). The residual is the genuine pose drift. |

The Sim(3)/SE(3) fits use camera orientations as well as centres, so they are
well-posed from 2 cameras (1 up to scale). Monocular backbones (MoGe-3) predict no
poses: `sim3_points` / `prescale_gt` fall back to `none`, which for them lifts with the
GT cameras. **Show predicted cameras** overlays the predicted poses (orange) in the
current display frame so per-camera drift is readable against the green/grey GT
frustums; the status line prints the fitted scales.

Metric models (`da3`, `unidepth`, `depthpro`, `metric3d`) exchange **normalised**
intrinsics; the viewer feeds the GT intrinsics (predicted ones as fallback). Only
Metric3D requires them. Their checkpoints download into the standard caches on first
build unless a `checkpoint_path` / `cache_dir` is set on the config
(`allow_download=False` forces offline).

The `gtdepth` backbone is DINOv2 features + ground-truth depth (predicts nothing):
a GT-geometry reference to A/B against any predicting backbone. It only works on
datasets that ship depth (hocap, physinone, synthrobot); elsewhere **Run** is disabled
with a red notice rather than rendering a placeholder.

## Adding a backbone or a dataset

The viewer lists whatever the registries contain:

* **Backbone**: add a `<name>BackboneConfig` (dataclass with `build()`) under
  `ontic_nn.wrappers` and register it in `ontic_nn.wrappers.BACKBONES`. If it needs GT
  depth, does not predict poses, or conditions on GT cameras, add the GUI hint to
  `runner.BACKBONE_NEEDS_GT_DEPTH` / `BACKBONE_IS_MONOCULAR` /
  `BACKBONE_ACCEPTS_GT_CAMERAS` (the built model's `accepts_gt_cameras` is authoritative).
* **Metric model**: register its config in `ontic_nn.metric_depth.METRIC_MODELS`.
* **Dataset**: register its `DatasetCfg` in `ontic_data.DATASETS`; it must build a
  `TemporalSceneDataset` (`record_labels` / `record_n_frames` / `load_sequence_views`).
  The CLI gets a `--<name>-root` flag automatically; add a dev-box default to
  `data_source.DEFAULT_ROOTS`, mark GT-depth availability in `data_source.HAS_GT_DEPTH`,
  and a view preset in `config.DATASET_DEFAULTS` if the generic one is a poor start. If
  the config's constructor does not take `root=`, map it in `data_source.CFG_KWARGS`.

## Layout

| Module | Responsibility |
|---|---|
| `data_source.py` | `Frame` + `GenericSource` / `build_source` over `ontic_data.DATASETS`; `DEFAULT_ROOTS`, `HAS_GT_DEPTH`. |
| `runner.py` | `BackboneRunner` (one backbone + one metric model on the GPU, freed on switch, inference mode), `BackboneResult`, `AlignMode`, `metric_unproject`. |
| `render.py` | Colour modes, frustum math, `build_point_cloud`, `pred_cameras_in_display_frame`. |
| `config.py` | `ViewConfig`, per-dataset defaults, preset store. |
| `robot_model.py` | FR3 Duo link geometry posed from recorded joint angles (mujoco). |
| `app.py` | `BackboneViewer`: the viser GUI wiring. |
| `share_tunnel.py`, `cli.py` | Share-URL resilience; the `ontic-backbone-viewer` entry point. |

Geometry lives in `ontic_lib`: `pointops.align` / `compute_alignment`,
`percentile_conf_threshold`, `camera.occupancy`, `structures.pointcloud_from_depth_views`,
`crop_to_aabb`, `nearest_point_to_ray`, `padded_aabb`, and the sampling ops.
