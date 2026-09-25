# Dataset and backbone tracking trials

The shared backbone viewer now includes tracking, geometry-only clip previews,
cached comparisons, and saved-run playback. Four runs below use real released
MVTracker weights on an RTX A5000; none use analytic trajectories.

| Dataset | Depth | Frames | Views | Queries | Tracking time | Mean predicted visibility |
|---|---|---:|---:|---:|---:|---:|
| hocap | sensor | 12 | 3 | 128 | 10.0s | 92.5% |
| hocap | vggt | 12 | 3 | 128 | 9.7s | 91.7% |
| hocap | moge3 | 12 | 3 | 128 | 6.2s | 80.3% |
| synthrobot | sensor | 12 | 3 | 128 | 6.7s | 84.1% |

Times exclude geometry preparation, input decoding and export. Visibility is the
model's own probability, not tracking accuracy. All four runs returned finite
trajectories for every query/frame. Ground-truth trajectories were not evaluated.
The HOCAP clip includes stationary surfaces and moving hand/object points.
Queries are selected from the displayed cloud after workspace, confidence and
point-count filters; the first-frame hand region further restricts these trials.
Backbone confidence is included in the updated recordings.

TAPIP3D also completed a pretrained GPU run on the HOCAP sensor clip using camera
`105322251564`: 12 frames, 128 displayed-point queries, 100% finite/valid samples,
and 96.6% mean predicted visibility. Its saved run is
`artifacts/tracking/hocap-sensor-tapip3d.viewer.npz`, with diagnostics in the
adjacent JSON. This uses sensor depth and calibrated dataset cameras; visibility
is not an accuracy measurement. Query identities pass the same display filters
used by the point cloud.

CoTracker3 completed the same 12-frame HOCAP sensor clip with 128 displayed-point
queries across three independent cameras in 2.7 seconds (1.20 GB peak allocated
GPU memory). 96.9% of samples have a usable 3D position; 97.6% pass the upstream
binary visibility decision. Invalid depth and occlusion remain invalid samples,
not background-surface trajectories. These percentages are runtime diagnostics,
not tracking accuracy. The saved recording and JSON are
`artifacts/tracking/hocap-sensor-cotracker3.viewer.npz` and
`artifacts/tracking/hocap-sensor-cotracker3.json`.

## Viewing

Launch the viewer from a configured runtime on AL3, explicitly loading the saved
recordings to compare (repeat `--recording` for each archive):

```bash
python -m ontic_viz.backbone_viewer.cli --host 127.0.0.1 --port 8091 \
  --model-config /mnt/fast/mzhobro/ontic_tracker_checkpoints/viewer-models.json \
  --recording artifacts/tracking/hocap-sensor-cotracker3.viewer.npz
```

The GPU runtime and required source paths are described below. The viewer needs
to be started explicitly; these instructions do not imply an active server.
For browser access, forward the port from your own machine:

```bash
ssh -N -L 8091:127.0.0.1:8091 mzhobro@AL3
```

Open <http://localhost:8091>. Under **1. Dataset → Saved runs**, select HOCAP sensor,
HOCAP VGGT, HOCAP MoGe-3, SynthRobot sensor, HOCAP sensor TAPIP3D, or HOCAP sensor
CoTracker3; press **Play**. **Cached result**
keeps recently opened/computed results for the current dataset trajectory.
Use **2. Depth → Video depth → Run video depth** to inspect new
backbone geometry before running a tracker. **3. Tracking → Query boxes** adds movable, resizable 3D regions.
Multiple included boxes form a union intersected with the displayed points; changing
filters or boxes hides cached tracks whose starting points are excluded.
The model file fields are prefilled for MVTracker, TAPIP3D, TrackCraft3R,
CoTracker3, VGGT and MoGe-3. Each tracker retains separate checkout/checkpoint
settings when selected.
Loading another dataset and selecting enabled cameras uses the same pipeline.
All tracker checkpoints, including TrackCraft3R's Wan base assets, are stored
under `/mnt/fast/mzhobro/ontic_tracker_checkpoints`; see the
[checkpoint inventory and runtime setup](tracker_checkpoints.md).

## Settings and interpretation

HOCAP uses validation trajectory 0, dataset frames 40–73 with step 3, and cameras
`105322251564`, `043422252387`, `105322251225`. The first-frame hand bounds plus
15 cm of padding restrict query seeding. SynthRobot uses validation trajectory 0,
frames 0–55 with step 5, and cameras `1-1`, `3-1`, `birds_eye`.

Images have long side 256. All camera poses/intrinsics come from the dataset;
VGGT and MoGe-3 depth are calibrated per frame by a robust scale against recorded
depth. Trajectories receive no subsequent frame-wise alignment. This trial uses
the base backbone outputs with scale calibration. After scale calibration, HOCAP depth
MAE over valid recorded pixels is about 4.7 cm for VGGT and 14.0 cm for MoGe-3 on
this clip. This small trial is not a model ranking.

Source: MVTracker revision `ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072`, released
`ethz-vlg/mvtracker/mvtracker_200000_june2025.pth`. The weights and source live in
`/mnt/fast/mzhobro/ontic_tracker_checkpoints/mvtracker` and
`/tmp/ontic-mvtracker-source` on this host.
Helper dependencies were installed into `/tmp/ontic-tracking-deps`.

TAPIP3D uses revision `4cb7e69a1687f67d56ec3e506768f51f2c581b46` in
`/tmp/ontic-tapip3d-source`, and the released `zbww/tapip3d/tapip3d_final.pth`
in `/mnt/fast/mzhobro/ontic_tracker_checkpoints/tapip3d`. Its `pointops2` Python package and
`pointops2_cuda` extension are in `/tmp/ontic-tracking-deps`, compiled with CUDA
12.4 for the RTX A5000 (SM 8.6). The full tracker checkpoint includes its encoder
weights, so loading it does not need an additional CoTracker download.

CoTracker3 uses PointWorld data revision `3872ec6` and its CoTracker submodule
revision `82e02e8` in `/tmp/ontic-pointworld-source/third_party/co-tracker`.
The released checkpoint is
`/mnt/fast/mzhobro/ontic_tracker_checkpoints/cotracker3/scaled_online.pth`;
the predictor uses PointWorld's offline settings and native 384 × 512 grid.
No additional CUDA extension or implicit depth model is used.

The GPU runs use the existing research environment at
`frontier_world_model/.venv` (Python 3.10, Torch 2.4.1+cu124); no shared ML stack was
replaced. Package regression tests use the main Python 3.13 environment. Its
Torch 2.12.1+cu130 cannot use this host's current NVIDIA driver. The project's
declared supported Python versions are unchanged.

## Reproducing

```bash
cd /is/sg2/mzhobro/AL_projects/0_mikel_projects/lib
export PYTHONPATH=$PWD/src:$PWD/packages/ontic-nn/src:$PWD/packages/ontic-data/src:$PWD/packages/ontic-viz/src:/tmp/ontic-tracking-deps:/tmp/ontic-mvtracker-source
export MPLCONFIGDIR=/tmp/ontic-viz-matplotlib
TRACKING_PY=/is/sg2/mzhobro/AL_projects/0_mikel_projects/frontier_world_model/.venv/bin/python
"$TRACKING_PY" scripts/research/record_tracking_comparison.py \
  --backbone vggt \
  --backbone-checkpoint /mnt/fast/mzhobro/vggt_omega_checkpoints/vggt_omega_1b_512.pt \
  --tracker-repo /tmp/ontic-mvtracker-source \
  --tracker-checkpoint /mnt/fast/mzhobro/ontic_tracker_checkpoints/mvtracker/mvtracker_200000_june2025.pth \
  --display-presets /tmp/ontic-dataset-viewer-presets.json \
  --output artifacts/tracking
```

Use `--backbone sensor` without a backbone checkpoint, or `--backbone moge3` with
`/mnt/fast/mzhobro/moge3_checkpoints/model.pt`. For SynthRobot, add
`--dataset synthrobot --start 0 --step 5 --views 0 4 16` with sensor depth.
For new CoTracker3 comparisons, add `--tracker cotracker3` and set `--tracker-repo`
and `--tracker-checkpoint` to its paths above. This script also applies its hand
region crop when available; the saved CoTracker3 trial used only the recording's
display filters, without that additional hand crop.

Each run writes a `.viewer.npz` with images/geometry/tracks, a `.tracks.npz`
numerical export, and a JSON report under `artifacts/tracking`. Binary recordings
are ignored by Git. `--recording` accepts the viewer archive, not the numerical
export. Use the module command above to launch playback of the exported archives.
