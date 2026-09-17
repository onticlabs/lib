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

## Viewing

On AL3 the integrated viewer runs on **127.0.0.1:8091**, in tmux session
`ontic-tracker-viz`, from the main workspace. On your own machine:

```bash
ssh -N -L 8091:127.0.0.1:8091 mzhobro@AL3
```

Open <http://localhost:8091>. Under **1. Dataset → Saved runs**, select HOCAP sensor,
HOCAP VGGT, HOCAP MoGe-3, SynthRobot sensor, or HOCAP sensor TAPIP3D; press **Play**. **Cached result**
keeps recently opened/computed results for the current dataset trajectory.
Use **2. Depth model → Depth for tracking → Preview depth clip** to inspect new
backbone geometry before running a tracker. **3. Tracking → Query boxes** adds movable, resizable 3D regions.
Multiple included boxes form a union intersected with the displayed points; changing
filters or boxes hides cached tracks whose starting points are excluded.
The model file fields are prefilled for MVTracker, TAPIP3D, VGGT and
MoGe-3. Each tracker retains separate checkout/checkpoint settings when selected.
Loading another dataset and selecting enabled cameras uses the same pipeline.
TrackCraft3R still requires its own upstream package and weights.

## Settings and interpretation

HOCAP uses validation trajectory 0, dataset frames 40–73 with step 3, and cameras
`105322251564`, `043422252387`, `105322251225`. The first-frame hand bounds plus
15 cm of padding restrict query seeding. SynthRobot uses validation trajectory 0,
frames 0–55 with step 5, and cameras `1-1`, `3-1`, `birds_eye`.

Images have long side 256. All camera poses/intrinsics come from the dataset;
VGGT and MoGe-3 depth are calibrated per frame by a robust scale against recorded
depth. Trajectories receive no subsequent frame-wise alignment. This trial uses
the base backbone outputs with scale calibration; the separate multiview
refinement experiment is not applied here. After scale calibration, HOCAP depth
MAE over valid recorded pixels is about 4.7 cm for VGGT and 14.0 cm for MoGe-3 on
this clip. This small trial is not a model ranking.

Source: MVTracker revision `ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072`, released
`ethz-vlg/mvtracker/mvtracker_200000_june2025.pth`. The weights and source live in
`/tmp/ontic-tracking-models/mvtracker` and `/tmp/ontic-mvtracker-source` on this host.
Helper dependencies were installed into `/tmp/ontic-tracking-deps`.

TAPIP3D uses revision `4cb7e69a1687f67d56ec3e506768f51f2c581b46` in
`/tmp/ontic-tapip3d-source`, and the released `zbww/tapip3d/tapip3d_final.pth`
in `/tmp/ontic-tracking-models/tapip3d`. Its `pointops2` Python package and
`pointops2_cuda` extension are in `/tmp/ontic-tracking-deps`, compiled with CUDA
12.4 for the RTX A5000 (SM 8.6). The full tracker checkpoint includes its encoder
weights, so loading it does not need an additional CoTracker download.

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
  --tracker-checkpoint /tmp/ontic-tracking-models/mvtracker/mvtracker_200000_june2025.pth \
  --display-presets /tmp/ontic-dataset-viewer-presets.json \
  --output artifacts/tracking
```

Use `--backbone sensor` without a backbone checkpoint, or `--backbone moge3` with
`/mnt/fast/mzhobro/moge3_checkpoints/model.pt`. For SynthRobot, add
`--dataset synthrobot --start 0 --step 5 --views 0 4 16` with sensor depth.

Each run writes a `.viewer.npz` with images/geometry/tracks, a `.tracks.npz`
numerical export, and a JSON report under `artifacts/tracking`. Binary recordings
are ignored by Git. `--recording` accepts the viewer archive, not the numerical
export. The current server launcher is `/tmp/ontic-dataset-viewer-server.sh 8091`.
