# ontic-nn

Neural backbones promoted from Ontic experiments, built on `ontic-lib`:
Point Transformer V3 over `PointBatch` (`ontic_nn.ptv3`), the DINOv2 ViT with
pretrained-weight loading (`ontic_nn.dinov2`), DPT decoder heads
(`ontic_nn.dpt`), a plain kNN point transformer (`ontic_nn.ppt`) and the shared
transformer layers (`ontic_nn.layers`). Every module imports with `torch` +
`einops`; accelerators are lazy, explicit opt-ins.

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

`attention.backend="flash"` needs `flash-attn` built against the local torch;
it is installed manually, not as an extra. `rope_impl="cuda"` and
`knn_query(impl="cuda")` use the CUDA extensions vendored under `ext/` at the
repository root (`./scripts/install_cuda_ext.sh`).

## Documentation

Per-subpackage descriptions, the PTv3 backend flags and presets, and runnable
snippets are in [`docs/ontic_nn.md`](../../docs/ontic_nn.md); the geometry
conventions every input follows are in
[`docs/ontic_lib.md`](../../docs/ontic_lib.md).
