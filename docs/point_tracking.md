# Point tracking

`ontic_nn.trackers` contains optional pretrained 3D point trackers alongside
`ontic_nn.wrappers` (geometry backbones). Importing the registry does not import
upstream research repositories, load weights, or access the network.

The adapters are MVTracker, TAPIP3D, TrackCraft3R, and CoTracker3 with depth lifting.
They use RGB video
and depth/cameras to follow persistent physical points. They do not accept an
uncoloured XYZ-only scan sequence as a substitute for RGB observations. The
[research comparison](point_tracking_research.md) explains the selection.

For this host's downloaded weights and prefilled viewer paths, see
[local tracker checkpoints](tracker_checkpoints.md).

## Shared interface

| Value | Shape / convention |
| --- | --- |
| `images` | Float RGB `[B,T,V,3,H,W]`, in `[0,1]` |
| `GeometrySequence.depth` | Camera-z depth `[B,T,V,Hd,Wd]` |
| `GeometrySequence.depth_valid` | Optional Boolean mask matching depth; finite positive depth is always required |
| `GeometrySequence.extrinsics` | Camera-to-world rigid matrices `[B,T,V,4,4]` |
| `GeometrySequence.intrinsics` | Normalized pinhole matrices `[B,T,V,3,3]` |
| `PointQueries.ids` | Unique `int64` identities `[B,N]` |
| `PointQueries.time` | `int64` indices into the supplied clip `[B,N]` |
| `PointQueries.xyz_world` | Query positions `[B,N,3]` in the geometry's world frame |
| `PointQueries.source_view`, `source_uv` | Optional paired source indices `[B,N]` and normalized coordinates `[B,N,2]` |
| `TrackerOutput.tracks_world` | Trajectories `[B,T,N,3]`, preserving query IDs and order |
| `TrackerOutput.valid` | Boolean `[B,T,N]`: supported, finite coordinate estimate |
| `TrackerOutput.visibility` | Optional `[B,T,N]` probability; consult `visibility_scope` |

Keep `V=1` for a monocular clip. All tensors must share a device. View ordering
must remain stable across the synchronized clip; missing observations are not
represented by padded black frames. RGB and depth can have different resolutions
but must cover the same image domain.

Normalized pixel centres are `((x + 0.5) / W, (y + 0.5) / H)`. The adapters
convert both intrinsics and coordinates to each upstream convention. Resizing
does not change depth units. Invalid depth is masked before resizing; it cannot
seed a query.

`valid` and `visibility` have different meanings: an occluded point may have a
valid predicted position. Missing visibility is unknown, not a score of zero.
CoTracker3 only provides 2D predictions, so its occluded samples have no valid
3D surface estimate and are marked invalid even if depth exists at that pixel.
Native scores are not assumed to be calibrated across trackers. No adapter
manufactures per-view visibility from an aggregate score.

## Geometry backbones

Depth, camera translations and queries must share one coordinate frame and scale
throughout each sequence. `frame_ids` identifies that frame per batch item;
`units` is a tuple of `"meters"` or `"arbitrary"`. An unspecified unit means
arbitrary scale, consistently across the entire clip.

For time-ordered `BackboneOutput` instances, use:

```python
from ontic_nn.trackers import GeometrySequence

geometry = GeometrySequence.from_backbone_outputs(
    outputs,                          # one BackboneOutput per timestep
    camera_source="predicted",        # or "provided", deliberately selected
    shared_world_frame=True,          # assertion after establishing alignment
    frame_ids=("scene-17",),           # one entry per batch item
    units=("meters",),                # only if metric scale is established
)
```

This stacks depth/cameras and applies finite-depth and sky masks. It does not
register independent reconstructions. In particular, running a multi-view depth
backbone independently at every time can produce a different similarity gauge
at each time. Align the geometry first; asserting `shared_world_frame=True`
does not perform that alignment. The selected camera keys must exist; there is
no implicit fallback to ground truth. A backbone's `depth_conf` is not a track
visibility probability.

Sensor depth with calibrated cameras can be supplied directly:

```python
geometry = GeometrySequence(
    depth=depth,                      # [B,T,V,Hd,Wd]
    extrinsics=camera_to_world,        # [B,T,V,4,4]
    intrinsics=normalized_intrinsics,  # [B,T,V,3,3]
    depth_valid=depth_valid,
    frame_ids=("calibrated-world",),
    units=("meters",),
    provenance={"depth": "sensor", "cameras": "calibration"},
)
```

## Queries and inference

Construct queries from image observations, or provide world positions with
persistent IDs directly. Source coordinates use the original normalized image
domain, independent of the tracker inference resolution.

```python
import torch
from ontic_nn.trackers import PointQueries, TRACKERS

# Example for B=1: two points in camera 0 at the start of the clip.
queries = PointQueries.from_pixels(
    geometry,
    time=torch.tensor([[0, 0]], device=images.device),
    source_view=torch.tensor([[0, 0]], device=images.device),
    source_uv=torch.tensor([[[0.4, 0.5], [0.6, 0.5]]], device=images.device),
    ids=torch.tensor([[17, 42]], device=images.device),
)

# Install upstream code/dependencies and obtain its checkpoint first; see below.
cfg = TRACKERS["mvtracker"](
    checkpoint_path="/models/mvtracker_200000_june2025.pth",
    allow_download=False,
)
tracker = cfg.build().to(images.device).eval()
with torch.no_grad():
    result = tracker(images, queries, geometry=geometry)

assert torch.equal(result.ids, queries.ids)
tracks = result.tracks_world  # [B,T,2,3]
```

`TrackerConfig` follows the backbone configuration pattern: `build()`,
`checkpoint_path`, `cache_dir`, `allow_download`, and `long_side`.
`freeze_tracker=True` is the default. Model capabilities are available as
`config.CAPABILITIES` before loading weights and `tracker.capabilities` after
building. These are offline clip adapters; no causal stream state is exposed.

## Upstream setup and limits

The unified workspace installer installs all backbones and trackers, including
pinned research sources, their Python dependencies and TAPIP3D's CUDA extension:

```bash
uv run --no-sync python scripts/install_models.py --download-weights
```

Use `--group trackers` to install/download trackers alone and `--check` (without
download flags) to verify imports. `--download-weights` includes TrackCraft3R's
Wan assets and generates model paths that the viewer reads automatically.
Omit that flag for code only, or use `--download-only` to fetch weights later.
After installation, the adapters discover the pinned sources automatically;
explicit checkout paths remain supported. See
[model installation](../packages/ontic-nn/README.md#model-installation) for
CUDA/compiler prerequisites and the explicit partial-install option.

For manual setup, individual `ontic-nn[mvtracker]`, `[tapip3d]`, `[cotracker3]`
and `[trackcraft3r]` extras provide Python dependencies only. Use a PyTorch/CUDA
combination supported by your driver and build extensions against that same
environment. No model files are downloaded
at import time. `allow_download=True` permits downloads during `build()`, from
the ontic store first ([weights](ontic_nn.md#weights)) and the hubs after;
`allow_download=False` requires local or already cached assets.

| Registry key | Native setting | API revision inspected |
| --- | --- | --- |
| `mvtracker` | Synchronized multi-view RGB-D; aggregate any-view visibility | [`ceea8ad`](https://github.com/ethz-vlg/mvtracker/tree/ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072) |
| `tapip3d` | Monocular RGB-D; query-view visibility; optional reverse pass | [`4cb7e69`](https://github.com/zbw001/TAPIP3D/tree/4cb7e69a1687f67d56ec3e506768f51f2c581b46) |
| `cotracker3` | Independent RGB tracks per source camera, lifted with supplied depth; query-view visibility | [`82e02e8`](https://github.com/facebookresearch/co-tracker/tree/82e02e8029753ad4ef13cf06be7f4fc5facdda4d) |
| `trackcraft3r` | Monocular dense reference field; queries at `t=0`; constant intrinsics | [`21e8fca`](https://github.com/cvlab-kaist/TrackCraft3r/tree/21e8fcaf4b6375b3044cead210d5808e1d81760b) |

The unified installer pins these revisions in `scripts/tracker_sources.json`.
Manually supplied checkouts are not automatically verified against them. The
wrappers loop over batch items because the native
predictors process one sequence at a time.

**MVTracker.** Install `ontic-nn[mvtracker]` and put the upstream checkout on
`PYTHONPATH` (or install its source). Its optional `pointops` KNN extension can
accelerate inference; upstream has a Torch fallback. The default weight is
`ethz-vlg/mvtracker/mvtracker_200000_june2025.pth`. `image_size=(H,W)` overrides
aspect-preserving `long_side` resizing. The adapter normalizes the scene around
the first-frame depth centroid and camera radius, then reverses that transform
on returned tracks. Select `scene_normalization="none"` for already normalized
geometry or `"manual"` with `scene_scale` and `scene_translation` for an explicit
similarity transform.

With the default 12-frame temporal window, supply at least seven frames.
`bidirectional=True` adds a reverse pass for queries after frame zero; setting
it to `False` leaves pre-query estimates invalid. With `grid_size=0`, clips must
also start tracking early enough for the native window loop to run.

**TAPIP3D (manual setup).** Install `ontic-nn[tapip3d]`, clone the repository, and build its
`third_party/pointops2` CUDA extension following the upstream installation
instructions. Supply `repo_path` (or
`ONTIC_TAPIP3D_REPO`) so the adapter can locate its generic `models`, `datasets`
and `utils` packages. Conflicting imports are rejected with an explanation.
MegaSAM is not required when supplying geometry.

```python
from ontic_nn.trackers import TAPIP3DConfig

cfg = TAPIP3DConfig(
    repo_path="/research/TAPIP3D",
    checkpoint_path="/models/tapip3d_final.pth",
    allow_download=False,
    resolution_factor=None,  # use long_side instead of upstream's area factor 2
    long_side=512,
)
```

The default weight is `zbww/tapip3d/tapip3d_final.pth`. The checkpoint carries
its model configuration and encoder weights. Loading the full checkpoint disables
the training config's CoTracker initialization download; encoder weights are
strictly loaded from TAPIP3D's checkpoint. Default `eval_mode="raw"` uses the supplied world
geometry. `bidirectional=True` adds a reverse-time pass when queries start
after frame zero. Short clips are padded for the upstream temporal window and
trimmed on output. Its inference outputs are detached by upstream; this wrapper
is not a training interface.

**CoTracker3 + depth.** Install `ontic-nn[cotracker3]` and clone CoTracker at the
revision above, which is the submodule pinned by [PointWorld's data branch](https://github.com/NVlabs/PointWorld/tree/3872ec6ee73146aa671192ef79b5dfbedc0246e3).
PointWorld's `real/flow_2d.py` uses `CoTrackerPredictor(offline=True, v2=False,
window_len=16)` with `facebook/cotracker3/scaled_online.pth`. This adapter follows
that combination, despite the checkpoint's `online` name; it is a full-clip API.

```python
from ontic_nn.trackers import CoTracker3Config

tracker = CoTracker3Config(
    repo_path="/research/co-tracker",  # or PointWorld with its submodule initialized
    checkpoint_path="/models/scaled_online.pth",
    allow_download=False,
).build().to("cuda")
result = tracker(images, queries, geometry=geometry)
```

`ONTIC_COTRACKER3_REPO` can replace `repo_path`; without either, the installed
`cotracker` package is used. No extra CUDA extension is needed. The native
predictor grid is 384 × 512 (unaffected by `long_side`); depth stays at its supplied
resolution. `bidirectional=True` supports queries after frame zero, and
`query_chunk_size=8192` bounds the number of explicit queries per predictor call.
Multi-camera inputs require query `source_view`/`source_uv`: each group is tracked
independently and returned in the original ID order, without cross-view fusion.

Like PointWorld, the adapter rounds tracks to depth pixels and unprojects the
sampled surface with calibrated cameras. It uses Ontic's pixel-center convention
when RGB/depth resolutions differ and supports time-varying camera poses.
Invalid depth, outside-image positions and occluded tracks yield `valid=False`
and NaN XYZ. Visibility contains the upstream binary decision (native threshold
0.9), not a calibrated probability. This excludes sampling an occluder's surface
as the hidden point's trajectory. Depth quality directly affects the lifted 3D
motion; no trajectory smoothing, depth estimation or cross-view optimization is
performed. The viewer's displayed-point filters and query boxes select the seeds.

**TrackCraft3R.** Install `ontic-nn[trackcraft3r]` and use the TrackCraft3R
checkout's `diffsynth` fork. It needs both the released tracking checkpoint
(`trackcraft3r/checkpoint/model.safetensors`) and the Wan2.1 base model, including
VAE, text encoder and tokenizer assets. The base model uses a separate
ModelScope cache.

```python
from ontic_nn.trackers import TrackCraft3RConfig

cfg = TrackCraft3RConfig(
    repo_path="/research/TrackCraft3r",
    checkpoint_path="/models/trackcraft3r/model.safetensors",
    base_model_cache_dir="/models/wan_models",
    allow_download=False,
    device="cuda:0",
)
```

Defaults follow the released 12-frame, `480x832` predictor. Set `height` and
`width` to change the inference grid (both multiples of 16); `long_side` does
not control its native grid. Queries are sampled from the dense reference grid
at pixel resolution, preserving input IDs. Invalid background depth is filled
with a per-frame median for its dense geometry encoder and the affected fraction
is reported; invalid query depth is rejected. Its NumPy/PIL inference path does
not support gradients, and `freeze_tracker=False` is rejected. Alternative clip
lengths require explicit configuration and have not been validated with weights.
The upstream `diag_max_depth=80` limit is in the geometry's units and clips
reference-camera z coordinates; set it deliberately for arbitrary-scale geometry
(zero disables clipping).
Device moves with `.to(device)` include its cached prompt; keep the native
`bfloat16` dtype. The unused text encoder remains offloaded on CPU.

## Validation status

CPU tests use deterministic substitute predictors to test geometry transforms,
query identity/order, time/view axes, batching, visibility, checkpoint plumbing,
and model-specific restrictions. They do not establish numerical parity or
tracking accuracy for the pretrained checkpoints. Full inference requires the
upstream dependencies, matching model assets, and suitable hardware.

## Interactive and headless visualization

The [Geometry & motion viewer](../packages/ontic-viz/README.md#workflow) connects
all four adapters to the depth backbones. Start with
`uv run --no-sync ontic-backbone-viewer --demo --device cpu` for cached playback of a known
synthetic motion sequence. Use **1. Dataset** for data, cameras and clip bounds,
**2. Depth model** for backbone geometry, **3. Tracking** for tracker and queries,
and **4. Display** for filters, trails, occlusions and cloud appearance.

Clip geometry uses calibrated dataset cameras with recorded depth, sensor-scaled
backbone depth, or camera-rig-scaled backbone depth. Preparation scales are
recorded separately from tracks. The viewer never fits trajectories per frame.
Monocular trackers select one camera from geometry that may have been prepared
with multiple views. Queries currently start at the first clip frame.

Export `.npz` for numerical analysis or, with `ontic-viz[rerun]`, `.rrd` for
headless recording and later replay. Both preserve dataset frame indices and
point identity. Rerun recording and the analytic demo require no model inference.


The viewer is integrated into the main `ontic-viz` package. Real pretrained
MVTracker inference has now been exercised with HOCAP sensor/VGGT/MoGe-3 geometry
and SynthRobot sensor geometry. See [tracking_dataset_trials.md](tracking_dataset_trials.md)
for the exact settings, observed diagnostics, saved recordings and reproduction
commands. TAPIP3D has also completed real pretrained inference on the HOCAP sensor
clip (one camera, 12 frames, 128 queries), with finite trajectories throughout.
TrackCraft3R remains covered by adapter contract tests without a pretrained run.
CoTracker3 completed a real HOCAP sensor-depth run with three cameras, 12 frames
and 128 displayed-point queries; invalid depth and occlusion are explicitly masked.
These checks establish runtime integration, not tracking accuracy.
