# ontic_nn — what's in the box

`ontic_nn` (`packages/ontic-nn`) holds neural backbones promoted from Ontic
experiments and built on `ontic_lib`. `import ontic_nn` imports nothing;
subpackages are imported explicitly and every module loads with `torch` +
`einops` alone. Accelerators (`spconv`, `flash-attn`, the CUDA kernels under
`ext/`) and the research packages behind `wrappers` / `metric_depth` are
imported lazily at construction or call time and raise an `ImportError`
naming what to install.

Tables list each subpackage's `__all__`.

## layers

Transformer building blocks shared by the DINOv2 ViT and the point
transformers (vendored from Meta's DINOv2, Apache-2.0). Tokens are
`(B, N, dim)`; attention uses `torch.nn.functional.scaled_dot_product_attention`.

| name | one line |
| --- | --- |
| `Attention` | Multi-head self-attention with a fused `qkv` projection |
| `Block` | Pre-norm block `x + ls1(attn(norm1 x)); x + ls2(ffn(norm2 x))` with LayerScale + DropPath |
| `drop_add_residual_stochastic_depth` | Evaluate the residual branch on a random batch subset, add rescaled |
| `DropPath` / `drop_path` | Per-sample stochastic depth (identity in eval) |
| `LayerScale` | Learnable per-channel residual scale `gamma (dim,)` |
| `Mlp` | `fc1 -> act -> drop -> fc2 -> drop` |
| `PatchEmbed` | Strided-conv patch embedding `(B, C, H, W) -> (B, N, D)` |
| `SwiGLUFFN` / `SwiGLUFFNFused` | SwiGLU feed-forward (used by ViT-g) |

## dinov2

DINOv2 vision transformer (CLS token, optional register tokens, patch tokens,
bicubic pos-embed resizing) plus loading of Meta's released weights. Weights
come through `torch.hub` on first use and are cached; nothing is fetched at
import time.

| name | one line |
| --- | --- |
| `DinoVisionTransformer` | The ViT module |
| `vit_small` / `vit_base` / `vit_large` / `vit_giant2` | Size constructors (`num_register_tokens=...`) |
| `DINOv2` | Randomly initialised model for `"vits" / "vitb" / "vitl" / "vitg"` (patch 14, img 518) |
| `load_pretrained_dinov2` | `Variant` (+ 4 registers if `use_reg`) with Meta's weights |
| `StandaloneDinoExtractor` | Frozen wrapper `(images, mask=None) -> patch features` |
| `Variant` / `EMBED_DIM` / `PATCH_SIZE` | `Literal["vits","vitb","vitl","vitg"]`, embed dims per variant, `14` |

## dpt

DPT decoder heads in the Depth Anything V3 layout; attribute names match
`depth_anything_3.model.dpt` / `dualdpt` so released weights load after a
key-prefix strip. Heads take four `(B, S, N, C)` ViT token maps (coarse to
fine), the image size and `patch_start_idx`, and return dicts of `(B, S, ...)`
maps at `(H, W) / down_ratio`.

| name | one line |
| --- | --- |
| `DPTHead` | Single-branch head: main map (+ confidence), optional sky map |
| `DualDPTHead` | Dual-branch head: main map (+ confidence) and a 6-channel auxiliary map |
| `FeatureFusionBlock` / `ResidualConvUnit` | The fusion / residual-conv building blocks |
| `create_uv_grid` / `position_grid_to_embed` | Normalised UV grid and its sinusoidal embedding |

## ppt

Plain point transformer: kNN attention on a packed cloud interleaved with
low-resolution global attention over the `(V, H, W)` multi-view grid the cloud
came from. Packed layout as in `ontic_lib.pointops`: `p (N, 3)`, `x (N, C)`,
`offset (B,)`, rows ordered `b, v, h, w`. kNN takes `impl="torch"` (brute
force) or `"cuda"` (vendored `pointops` extension).

| name | one line |
| --- | --- |
| `PlainPointTransformer` | `num_blocks` x (kNN block, multi-view low-res attention) |
| `TransformerBlock` / `KNNAttention` | Pre-norm kNN-attention block / single-head neighbour attention |
| `MultiViewLowResAttention` | Global attention over all views at `1 / down_factor` resolution |
| `knn_query` / `knn_query_torch` | Group-respecting kNN → `(M, k)` indices + distances |
| `local_knn_query` | Windowed kNN for a multi-view depth-map cloud (pure torch) |

## ptv3

Point Transformer V3 as a serialized point-transformer U-Net over
`ontic_lib.structures.PointBatch`, with optional time merging for
spatio-temporal inputs.

**How it fits together.** `PointTransformerV3(cfg)` takes a `PointBatch`
(`coord`, `feat (N, in_channels)`, `batch`, optional `time`). `forward` builds a
`StageState` for stage 0: voxel coords at `cfg.grid_size`, dense group ids
(batch, or `(batch, time)` in temporal mode), and a `Serialization` of the
grid under `cfg.orders` (shuffled per forward, driven by `generator`). Every
encoder stage after the first pools (`SerializedPooling` or `GridPooling`,
pushing a `PoolRecord`), optionally merges time steps (`Merger`, pushing a
`MergeRecord`), then runs its blocks (sparse-conv positional encoding →
windowed serialized attention → MLP). The decoder pops the records in reverse:
`Unmerger`, `Unpooling` back onto the parent state, blocks, and finally the
head MLP. The output is a `PointBatch` with `feat (N, out_channels)` in the
input row order (`cls_mode` returns the coarsest stage's points instead;
`feature_pyramid` concatenates all encoder stages at full resolution).

**Backend flags** (all pure torch by default):

| flag | values | what it switches |
| --- | --- | --- |
| `cfg.conv_impl` | `"torch"` / `"spconv"` | Stem and CPE submanifold conv: neighbour-table torch impl vs `spconv` (`ontic-nn[spconv]`) |
| `cfg.attention.backend` | `"sdpa"` / `"flash"` | Windowed attention via `scaled_dot_product_attention` vs `flash_attn_varlen_qkvpacked_func` (no RPE / upcast with flash) |
| `cfg.attention.rope_impl` | `"torch"` / `"cuda"` | 3D RoPE module vs the vendored `point_rope_cuda` kernel (forward-only w.r.t. coords) |

Also in `AttentionCfg`: `rpe` (`"none"` / `"v1"` / `"v2"`), `rope`,
`rope_coord_source`, `rope_stage_rescale`. `PoolingCfg.kind` picks
`"serialized"` or `"grid"` pooling; `TemporalCfg.merge_window` sets the time
fold per stage (`temporal=None` disables it, and is required for inputs
without `time`).

**Presets** (`ontic_nn.ptv3.presets`, each accepts `**overrides`): `base()`
(enc depths 2,2,2,6,2 / 32..512 ch), `medium()` (3,3,3,6,3 / 48..512),
`large()` (3,3,3,12,3 / 48..512), and `fwomo_legacy()` = `base` with
`conv_impl="spconv"` and flash attention, the setup the fwomo checkpoints
were trained with.

**Checkpoints.** `load_fwomo_state_dict(model, state_dict, strict=True)`
loads a fwomo (frontier) `PointTransformerV3.state_dict()` into this
implementation: spconv `(out, k, k, k, in)` weights become `(K³, in, out)`,
`PointSequential` numeric children get their attribute names, the final
`dec.<n>` MLP becomes `dec.head`.

| name | one line |
| --- | --- |
| `PointTransformerV3` | The model; `forward(points, *, generator=None) -> PointBatch` |
| `PointTransformerV3Cfg` | Top-level config (`encoder`, `decoder`, `attention`, `pooling`, `temporal`, `conv_impl`, ...); `validate()` |
| `StageCfg` / `AttentionCfg` / `PoolingCfg` / `TemporalCfg` | Per-stage, attention, pooling and time-merge settings |
| `StageState` | Points of one stage plus cached grid coords, serialization, windows, neighbour tables |
| `PoolRecord` / `MergeRecord` | What unpooling / unmerging need (parent state, cluster ids) |
| `presets` | `base`, `medium`, `large`, `fwomo_legacy` |
| `load_fwomo_state_dict` | Load a fwomo checkpoint (key rename + weight re-layout) |

### Minimal usage

PTv3 on CPU (verified with `uv run --no-sync python`):

```python
import torch
from ontic_lib.structures import PointBatch
from ontic_nn.ptv3 import PointTransformerV3, presets

cfg = presets.base(in_channels=6, out_channels=3, temporal=None)
model = PointTransformerV3(cfg).eval()

coord = torch.rand(2, 500, 3)                        # (B, K, 3) padded
feat = torch.cat([coord, torch.rand(2, 500, 3)], -1)  # (B, K, 6)
points = PointBatch.from_padded(coord, feat)         # packed, batch ids 0/1
with torch.no_grad():
    out = model(points)
out.feat.shape                                       # torch.Size([1000, 3])
```

`Gaussians` + `io` round trip:

```python
import torch
from ontic_lib.structures import Gaussians
from ontic_lib.io import save_gaussians, load_gaussians

n = 100
g = Gaussians(
    means=torch.randn(n, 3),
    scales=torch.rand(n, 3),                               # post-activation
    rotations=torch.tensor([1.0, 0, 0, 0]).repeat(n, 1),   # wxyz
    opacities=torch.rand(n, 1),
    harmonics=torch.rand(n, 3, 1),                         # (N, 3, d_sh)
)
save_gaussians("scene.npz", g, metadata={"source": "demo"})
g2 = load_gaussians("scene.npz")
g2.covariance().shape                                      # torch.Size([100, 3, 3])
```

## wrappers

Pretrained multi-view depth backbones behind one contract. A wrapper takes
`images (B, V, 3, H, W)` in `[0, 1]` plus optional GT cameras (c2w `(B, V, 4, 4)`,
normalised intrinsics `(B, V, 3, 3)`) and optional GT `depth (B, V, 1, H, W)`,
resizes internally to its `long_side` (patch-snapped), and returns a
`BackboneOutput`. Importing `ontic_nn.wrappers` fills the `BACKBONES` registry
(`name -> config class`); the research package itself is imported inside
`build()` / `forward` and the `ImportError` names the `ontic-nn[<extra>]` extra
and the repository. Individual extras carry the support dependencies. For all
viewer backbones, run `uv run --no-sync python scripts/install_backbones.py`
from the workspace root: it installs `ontic-nn[backbones]` plus pinned research
sources while preserving the environment's PyTorch/CUDA stack. See
[setup and verification](../packages/ontic-nn/README.md#geometry-backbone-setup).
Weights come separately from the HF hub.

**Contract.** `BackboneConfig` (dataclass): freezing switches
(`freeze_backbone`, `freeze_dpt_head`, `freeze_cam_dec`, `freeze_cam_enc`),
`long_side` (default 518), `gradient_checkpointing`, and the checkpoint fields
`checkpoint_path` (a downloaded file / snapshot dir, used first), `cache_dir`
(`None` = the HF / torch-hub cache) and `allow_download` (`False` = cache only);
subclasses add their `model_dir` / `model_name` and implement `build()`.
`BackboneBase(nn.Module)`: `forward(images, extrinsics=None, intrinsics=None,
depth=None) -> BackboneOutput`, `patch_size`, `encoder_dim`,
`accepts_gt_cameras` (True where GT cameras *condition* the model rather than
pass through). `BackboneOutput.data` holds `depth`, `depth_conf` (`expp1`, so
`>= 1`), `sky_mask` at `dpt_resolution`; `patch_feat_0..3 (B, V, Ph, Pw, C)` at
`patch_resolution`; `extrinsics` / `intrinsics` (GT, if given) and
`extrinsics_pred` / `intrinsics_pred`; `get_extrinsics()` / `get_intrinsics()`
prefer GT over predicted.

| key | model (upstream package) | cameras | extra |
| --- | --- | --- | --- |
| `da3` | Depth Anything 3 (`depth_anything_3`, ByteDance-Seed) | conditions on GT; predicts | `ontic-nn[da3]` |
| `ma` | MapAnything (`mapanything`, facebookresearch) | conditions on GT; predicts | `ontic-nn[ma]` |
| `vggt` | VGGT-Omega (`vggt_omega`, facebookresearch; gated HF checkpoint) | pose-free; predicts | `ontic-nn[vggt]` |
| `pi3x` | Pi3X (`pi3`, yyfz/Pi3) | conditions on GT; predicts | `ontic-nn[pi3x]` |
| `dvlt` | DVLT "Deja View" looping transformer (`dvlt`, nv-tlabs) | pose-free; predicts | `ontic-nn[dvlt]` |
| `moge3` | MoGe-3 (`moge` + git-only `utils3d_moge`, `flex-gemm`; microsoft) | monocular, no poses | `ontic-nn[moge3]` |
| `gtdepth` | DINOv2 features + ground-truth `depth` (predicts nothing) | GT only | none |

| name | one line |
| --- | --- |
| `BACKBONES` / `register_backbone` | The registry and its class decorator |
| `<Name>BackboneConfig` / `<Name>Backbone` | Per-model config + module (`DA3`, `MA`, `VGGT`, `Pi3X`, `DVLT`, `MoGe3`, `GTDepth`) |
| `load_da3` | Upstream `DepthAnything3` from a snapshot / the hub, or a random-init preset |
| `offline_guard(allow_download)` | Context manager: with `False`, HF hub and `torch.hub` are cache-only |
| `resolve_checkpoint` | Local path of a checkpoint: `checkpoint_path` first, else the HF hub |
| `import_research_module` | `importlib.import_module` with an `ImportError` naming extra + repo |
| `resize_to_long_side` / `amp_dtype` | Patch-snapped long-side resize; bf16-or-fp16 autocast dtype |
| `convert_to_buffer` / `buffers_to_params` / `set_frozen` / `extract_weights` | Parameter-freezing and state-dict helpers |

## metric_depth

Metric monocular depth models behind a second, smaller contract: a
`MetricDepthModel` is frozen and eval-only, `forward(images (B, V, 3, H, W)
in [0, 1], intrinsics=None) -> MetricDepthOutput` with `depth (B, V, Hd, Wd)`
metric z-depth in metres at its `long_side` grid, optional `conf`, and the
model's own `intrinsics` when it self-calibrates. Intrinsics crossing the
boundary are always normalised. `MetricDepthConfig` has the same
`checkpoint_path` / `cache_dir` / `allow_download` fields as the backbones plus
`long_side`; `METRIC_MODELS` / `register_metric_model` mirror the backbone
registry, and `offline_guard` is re-exported.

| key | model | `long_side` | intrinsics | extra |
| --- | --- | --- | --- | --- |
| `da3` | DA3 nested metric checkpoint (`depth-anything/DA3NESTED-GIANT-LARGE-1.1`) | 504 | ignored; predicted ones returned | `ontic-nn[da3]` |
| `depthpro` | Apple Depth Pro (`depth_pro`, `apple/DepthPro`) | 1536 | self-estimated focal | `ontic-nn[depthpro]` |
| `metric3d` | Metric3Dv2 via `torch.hub` (`yvanyin/metric3d`, `variant` vit_small/large/giant2) | 616 | **required** (`REQUIRES_INTRINSICS`) | `ontic-nn[metric3d]` |
| `unidepth` | UniDepthV2 (`unidepth`, `lpiccinelli/unidepth-v2-vitl14`) | 644 | predicts its own | `ontic-nn[unidepth]` |
