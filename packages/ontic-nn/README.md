# ontic-nn

Neural backbones promoted from Ontic experiments, built on `ontic-lib`:
Point Transformer V3 over `PointBatch` (`ontic_nn.ptv3`), the DINOv2 ViT with
pretrained-weight loading (`ontic_nn.dinov2`), DPT decoder heads
(`ontic_nn.dpt`), a plain kNN point transformer (`ontic_nn.ppt`) and the shared
transformer layers (`ontic_nn.layers`). Every module imports with `torch` +
`einops`; accelerators are lazy, explicit opt-ins.

Optional geometry backbones (`ontic_nn.wrappers`) and 3D point trackers
(`ontic_nn.trackers`) share camera conventions. The trackers preserve point
identities across RGB-D clips; see the [tracking API and setup](../../docs/point_tracking.md).

## Install

As a git dependency of an experiment (the package lives in a subdirectory of
the `ontic-lib` workspace):

```toml
[project]
dependencies = ["ontic-nn[ptv3]"]

[tool.uv.sources]
ontic-nn = { git = "<repo url>", subdirectory = "packages/ontic-nn" }
```

For co-development, point the source at a checkout instead:

```toml
[tool.uv.sources]
ontic-nn = { path = "../lib/packages/ontic-nn", editable = true }
```

Inside this repository it is a workspace member and comes with the `dev`
dependency group.

## Extras

| extra | package | enables |
| --- | --- | --- |
| `ontic-nn[ptv3]` | (none) | Marker for the PTv3 stack; pure torch by default |
| `ontic-nn[spconv]` | `spconv-cu120` (Python < 3.12 only) | `conv_impl="spconv"` for the PTv3 stem / CPE |
| `ontic-nn[timm]` | `timm` | Alternative DINOv2 pretrained-weight loading, if used |
| `ontic-nn[backbones]` | Six image backbone dependency sets plus video depth | Use the installer below to include pinned research sources |
| `ontic-nn[video-depth]` | Video Depth Anything inference dependencies | Temporal metric depth, Small / Base / Large; also included in `backbones` |
| `ontic-nn[velodepth]` | VeloDepth inference dependencies | Metric video geometry propagation; also included in `backbones` |

`attention.backend="flash"` needs `flash-attn` built against the local torch;
it is installed manually, not as an extra. `rope_impl="cuda"` and
`knn_query(impl="cuda")` use the CUDA extensions vendored under `ext/` at the
repository root (`./scripts/install_cuda_ext.sh`).

### Geometry backbone setup

From the workspace root, install the viewer backbones and video depth models into your existing
environment with one command:

```bash
uv run --no-sync python scripts/install_backbones.py
```

This covers **DA3, MapAnything, VGGT-Omega, Pi3X, DVLT, MoGe-3**, and built-in
**gtdepth**, plus **Video Depth Anything** and **VeloDepth**. DA3 also supports joint clip inference.
It requires Linux, Python 3.12+, `git`, `uv`, and an environment
with your chosen PyTorch, torchvision and NumPy already installed. It uses the
invoking Python; `.venv/bin/python scripts/install_backbones.py` is equivalent.
The installer constrains the installed PyTorch family, Triton, NumPy and NVIDIA
runtime packages to their current versions. If the dependencies cannot coexist,
resolution fails instead of upgrading or downgrading that stack.

Support dependencies come from `ontic-nn[backbones]`, using `uv.lock` versions
while retaining the installed ML stack. Python 3.13 uses Open3D
0.20+ (Linux wheels need glibc 2.35+). DA3 needs MoviePy 1.x and its import-time
export dependencies; MapAnything's source revision matches the pinned UniCeption
0.1.6 sources, without pulling in its unused audio dependency.
FlexGEMM uses its Triton implementation, with no separate CUDA extension build.

Research source revisions are recorded in
[`scripts/backbone_sources.json`](../../scripts/backbone_sources.json). The installer
fetches unmodified checkouts inside the virtualenv's `share/ontic-backbones/` and
registers them in `site-packages/ontic_backbones.pth`. Those pinned sources take
precedence over other copies of the same research modules in that environment.
This is an **inference installation**: upstream training/demo dependency pins
(including DA3's Python upper bound and VGGT's NumPy upper bound) are not applied.
It does not install upstream CLI entry points or training/demo extras.

Regular `uv run` and `uv sync` preserve these source registrations. Recreating the
virtualenv requires rerunning the installer. An unchanged rerun reuses the pinned
checkouts; modified tracked source files cause an error instead of being overwritten.
Restart an already-running viewer after installation.

Check imports again:

```bash
uv run --no-sync python scripts/install_backbones.py --check
```

The default check reports MoGe-3 as **skipped** without usable CUDA because
FlexGEMM queries GPU properties during import. Missing dependencies return a
nonzero exit code. These checks do not validate pretrained inference or download
model weights.
Normal viewer inference downloads missing weights; VGGT-Omega requires Hugging
Face checkpoint access. CUDA 13 inference also requires a compatible host driver.

Individual extras such as `ontic-nn[vggt]` still contain support dependencies
only; use the installer to include the research sources. Metric-depth models
and point trackers have separate setup requirements.

## Video depth

`ontic_nn.video_depth` provides three metric model families. See the
[research and priority matrix](../../docs/video_depth_models.md) for the selection rationale.

| Registry key | Temporal inference | Resolution control | Local checkpoint |
| --- | --- | --- | --- |
| `vda_small`, `vda_base`, `vda_large` | Video Depth Anything temporal windows with overlap blending | `input_size=518`, short side, multiple of 14 | Metric `.pth` file |
| `velodepth` | VeloDepth learned propagation between keyframes | `resolution_level=0`, native pixel-budget bucket 0–9 | HF snapshot directory |
| `da3_nested` | DA3 Nested Giant-Large 1.1, all frames jointly with one native metric scale | `input_size=504`, long side, multiple of 14 | HF snapshot directory |

The backbone installer includes all three pinned research sources. For manual setup,
install the corresponding `video-depth`, `velodepth`, or `da3` extra and provide
the upstream repository root as `repo_path`. Upstream training/demo dependencies
and optional CUDA extensions are not required by these adapters.

```python
from ontic_nn.video_depth import VIDEO_DEPTH_MODELS

model = VIDEO_DEPTH_MODELS["vda_small"](
    allow_download=True,  # default False: local checkpoint / HF cache only
    # checkpoint_path="/path/to/metric_video_depth_anything_vits.pth",
    # repo_path="/path/to/Video-Depth-Anything",
    input_size=518,
).build().to("cuda").eval()
depth = model(images)  # RGB floats [0,1], (B,T,V,3,H,W) -> CPU meters (B,T,V,H,W)
```

Each camera is processed as an independent ordered video. VeloDepth returns
camera-z depth, not its separate radial `distance` output. DA3 must use a
**Nested** checkpoint with the metric branch; its joint attention can consume
substantial memory, so start with short clips. VDA must use **metric** weights.
Invalid predictions become zero; all adapters return depth at the input resolution.
CPU uses full precision; CUDA uses autocast unless `fp32=True`. Cancellation
callbacks are checked between camera videos; a running upstream call must finish.
Temporal consistency does not impose cross-camera consistency or replace dataset
camera calibration.

Downloads default to disabled. `checkpoint_path` overrides the HF cache;
`cache_dir` selects the HF cache directory. VeloDepth's upstream constructor also
loads ConvNeXt initializers through `torch.hub`; these must be cached for offline
use. Set `TORCH_HOME` to choose that separate cache. The download flag covers
these initializers too. VeloDepth imports `wandb` as an upstream dependency but
the adapter does not start a logging run.

## Documentation

Per-subpackage descriptions, the PTv3 backend flags and presets, and runnable
snippets are in [`docs/ontic_nn.md`](../../docs/ontic_nn.md); the geometry
conventions every input follows are in
[`docs/ontic_lib.md`](../../docs/ontic_lib.md).
