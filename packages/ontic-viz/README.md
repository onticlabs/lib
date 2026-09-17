# ontic-viz

Interactive tools over `ontic-nn` (depth backbones and point trackers) and
`ontic-data` (dataset loaders). The **Geometry & motion viewer** is a
[viser](https://viser.studio) web GUI for calibrated depth clouds and persistent
3D trajectories. It supports MVTracker, TAPIP3D, and TrackCraft3R alongside the
existing depth backbones. The command remains `ontic-backbone-viewer`.

## Running

Install all backbone sources and support dependencies into the existing
environment, preserving its PyTorch/CUDA versions:

```bash
uv run --no-sync python scripts/install_backbones.py
```

See [backbone setup and verification](../ontic-nn/README.md#geometry-backbone-setup)
for prerequisites and import checks.

```bash
# Start with a moving RGB-D scene and known trajectories; no model weights needed.
uv run ontic-backbone-viewer --demo --device cpu --host 127.0.0.1

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
`--demo`, `--share`, `--presets <json>`, and one `--<dataset>-root <path>` per registered
dataset. Roots default to the dev-box paths in `data_source.DEFAULT_ROOTS`
(`/mnt/fast/...`); override them on other machines, e.g.
`--genesis-root /fast/mzhobro/datasets/soft_genesis_elastic`.

**robot-dextris** opens `/mnt/fast/mzhobro/trailer-demo` using the DEXTRIS loader:
eight calibrated RGB cameras at 60 FPS, with recordings directly under the root.
Select it in **Dataset** and press **Load dataset**; override the path with
`--robot-dextris-root /path/to/recordings`. It lists all recordings without a
train/validation split and has no GT depth, hand labels or default workspace crop.
Each recording needs `calibration_result.json` and `<recording>_P00.mp4` through
`<recording>_P07.mp4`; empty or missing-file directories are ignored.

Build its recording index once (requires `ffprobe` on PATH):

```bash
uv run --no-sync python scripts/index_dextris.py \
  --dataset robot-dextris --root /mnt/fast/mzhobro/trailer-demo
```

This writes `dextris_info.csv`, the same metadata format used by DEXTRIS. Startup
then reads frame counts from the CSV. Use `--refresh` after adding or replacing
recordings; incomplete videos or differing camera frame counts are reported and
excluded. The same command supports `--dataset dextris` for nested hand captures.
Robot recordings use `ontic-data[opencv,tables]`: OpenCV's FFmpeg decoder opens the
container index without scanning every frame. The viewer retains only the current
trajectory's readers per data source and reuses them while scrubbing or tracking.

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

1. **Dataset:** load a dataset and trajectory, enable input cameras, and set the
   clip's start frame, frame count and frame step. Image
   frustums show their calibrated poses; green means selected. Camera selection
   survives frame changes.
2. **Depth model:** select a backbone, conditioning and **Backbone checkpoint**
   (or allow missing-weight downloads). The model hint identifies single- or
   multi-view depth. **Current-frame depth** runs that model on the current frame;
   its alignment controls are in this same step. **Depth for tracking** chooses
   recorded depth or the selected backbone's calibrated predictions, plus clip
   resolution. **Preview depth clip** prepares geometry without a tracker.
3. **Tracking:** choose a tracker, point budget, sampling and query boxes.
   The depth-input summary shows what step 2 will supply. **Run tracking** adds
   trajectories and reuses matching prepared geometry.
   Queries are sampled once from the displayed surfaces at the first clip frame.
   They respect **Display** workspace cropping, depth confidence, pixel stride,
   voxel reduction, SFC stride and applied FPS. Farthest-point sampling spreads
   queries in 3D; even coverage is faster.
   Under **3. Tracking → Query boxes**, **Add box** creates a region. Drag its axes to
   move it, edit **Box size** to resize it, and add more boxes for multiple objects.
   Seeds must be visible and inside any included box when **Use query boxes** is on.
   Boxes select starting points; trajectories can subsequently leave them.

4. **Display:** adjust clouds, tracks, workspace filters, overlays and saved presets.

**Playback** remains available below all four tabs.
Play/Pause or scrub **Dataset frame**. Tracking and geometry are cached; playback
performs no model inference. Point IDs and colors persist, invalid samples break
trails, and occluded points can be dimmed or hidden. **Cached result** switches
between up to six completed results for the loaded trajectory, keeping the current
timestep when shared. Frame step refers to dataset frames; playback FPS is a
viewing speed, not a recovered capture rate. Frames outside the cached clip show no tracks.

Changing display filters or query boxes hides entire cached trajectories whose
starting points are excluded. Run tracking again to seed newly included surfaces.
Workspace and depth-confidence filters also apply to each later trajectory
sample. Excluded samples disappear and break trails, including when occluded
tracks are enabled. Confidence follows the current projected point in the
tracking cameras, using the same per-frame threshold as the depth cloud.

**Display:** adjust trail length, visibility threshold, track size and the
background cloud. Existing cloud thinning, occupancy, ruler, hand/robot
overlays, workspace cropping and saved display presets remain available.

**Download tracks (.npz):** save IDs, world trajectories, validity/visibility,
query provenance, dataset frame indices, camera calibration and metadata.
NumPy can read the archive with `allow_pickle=False`.

The demo contains **analytic motion**, not neural predictions. Its moving sphere
lets you exercise the entire interface without pretrained packages or weights.
The UI reports the tracker, geometry source and world units for the displayed run.

### Geometry for tracking

Every clip uses the dataset's synchronized, normalized intrinsics and rigid
camera-to-world poses in meters. Select one explicit depth policy:

| Depth source | Requirements | Preparation |
|---|---|---|
| Sensor / GT depth | Recorded depth | Uses recorded camera-z depth directly; no backbone or DINO load. |
| Backbone depth · sensor scale | Backbone and recorded depth | Fits a median depth scale per frame against recorded depth, then lifts with calibrated dataset cameras. |
| Backbone depth · camera-rig scale | Camera-predicting backbone; at least two distinct calibrated camera centers | Fits predicted-to-calibrated camera scale per frame, scales depth, then lifts with dataset cameras. Degenerate rigs are rejected. |

Calibration scales are recorded in output provenance. **Trajectories are never
aligned per frame.** Single-frame display alignment is disabled for cached tracks
so the cloud and trajectories stay in the same world frame. Geometry is reused
when changing tracker or query settings; dataset, trajectory, clip, camera,
resolution, backbone, conditioning or geometry-policy changes invalidate that cache.
Backbone confidence is retained in clip caches and viewer recordings separately
from depth validity. Sensor depth has no confidence score. For cached clip clouds,
voxel reduction retains a real surface sample per voxel so displayed points and
query pixels coincide. Earlier recordings without confidence remain readable;
rerun their geometry to enable confidence filtering. Segmentation masks are not
yet connected to the viewer; boxes provide a calibration-consistent multi-view
selection without a segmentation model.

MVTracker jointly uses all enabled views and needs at least seven frames.
TAPIP3D uses one selected tracker camera. TrackCraft3R also uses one camera and
requires exactly 12 frames in this viewer, with its native 480 × 832 grid.
Monocular tracking can use depth prepared by a multi-view backbone. These adapters
consume synchronized RGB plus calibrated geometry; this is not a raw-LiDAR tracker.

Install each tracker's optional dependencies and research checkout as described
in [point_tracking.md](../../docs/point_tracking.md). **Model files** accepts a
checkout, checkpoint, and (TrackCraft3R) Wan cache. Missing-weight downloads are
off by default. Cancel takes effect between preparation frames or after the
current model call returns. A failed or cancelled run preserves the previous
completed result. Loading another dataset or trajectory clears it.

### Headless Rerun recordings

Install the optional extra with `uv sync --package ontic-viz --extra rerun`.
The viewer then offers **Download Rerun (.rrd)**. It records calibrated cameras,
RGB images, depth clouds, point IDs as labels, visibility coloring and trails on
a dataset-frame timeline. Recording itself needs no GUI, GPU or Rerun server.

The same exporter works without Viser:

```python
from ontic_viz.backbone_viewer.demo import DemoSource
from ontic_viz.backbone_viewer.tracking import ClipSpec, TrackerSettings, TrackingRunner
from ontic_viz.backbone_viewer.rerun_export import save_rerun

source = DemoSource()
runner = TrackingRunner(device="cpu")
clip = runner.prepare(source, 0, ClipSpec(length=48), ("left", "right"))
run = runner.run(clip, TrackerSettings(name="demo"), source=source, count=512)
save_rerun(run, "tracks.rrd")
```

Open the result with `rerun tracks.rrd`, or verify it without a display using
`rerun rrd verify tracks.rrd`. Neural models still require their upstream packages,
checkpoints and compatible hardware; the analytic demo does not validate their
prediction quality.

## Saved dataset comparisons

`--recording path.viewer.npz` adds a **1. Dataset → Saved runs** chooser; repeat the
flag for multiple recordings. These archives include RGB, calibrated depth,
cameras, query IDs and trajectories, so switching runs needs no model inference.
The corresponding dataset root must remain accessible for loading the trajectory
and for computing new clips. Numerical exports from **Download tracks** contain
trajectories only and are distinct from viewer recordings.

```bash
uv run --no-sync ontic-backbone-viewer --device cuda \
  --recording artifacts/tracking/hocap-sensor-mvtracker.viewer.npz \
  --recording artifacts/tracking/hocap-vggt-mvtracker.viewer.npz \
  --recording artifacts/tracking/hocap-moge3-mvtracker.viewer.npz
```

`--model-config models.json` pre-fills local backbone checkpoints and each
tracker's model files. Switching trackers restores that tracker's paths and
download settings, including any edits made during the session:

```json
{
  "backbone_checkpoints": {"vggt": "/models/vggt.pt", "moge3": "/models/moge3.pt"},
  "trackers": {
    "mvtracker": {"repo_path": "/research/mvtracker", "checkpoint_path": "/models/mvtracker.pth"},
    "tapip3d": {"repo_path": "/research/TAPIP3D", "checkpoint_path": "/models/tapip3d_final.pth"}
  }
}
```

The older singular `"tracker"` entry still applies to MVTracker (or its explicit
`"name"`), without sharing those paths with other trackers. TAPIP3D needs its own
checkout and compiled `pointops2_cuda` extension; see [tracker setup](../../docs/point_tracking.md).

The headless API is `save_recording(run_or_clip, path)` / `load_recording(path)`
in `ontic_viz.backbone_viewer.recording`. Archives use NumPy arrays and JSON,
without pickle. `scripts/research/record_tracking_comparison.py` reproduces real
backbone/MVTracker runs from explicit local paths. See the
[real-data trial report](../../docs/tracking_dataset_trials.md).

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
| `app.py` | `BackboneViewer`: scene, depth and display GUI wiring. |
| `tracking.py` | Clip preparation/cache, geometry calibration, query sampling, inference and NPZ export. |
| `tracking_panel.py` | Background tracking jobs, cancellation, playback, stable colors and trails. |
| `recording.py` | Save/reopen calibrated clips with optional tracks; no inference or pickle on reload. |
| `demo.py` | Deterministic RGB-D scene with known motion and visibility. |
| `rerun_export.py` | Optional headless `.rrd` recording. |
| `share_tunnel.py`, `cli.py` | Share-URL resilience; the `ontic-backbone-viewer` entry point. |

Geometry lives in `ontic_lib`: `pointops.align` / `compute_alignment`,
`percentile_conf_threshold`, `camera.occupancy`, `structures.pointcloud_from_depth_views`,
`crop_to_aabb`, `nearest_point_to_ray`, `padded_aabb`, and the sampling ops.
