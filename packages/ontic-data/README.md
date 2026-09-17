# ontic-data

Dataset loaders for temporal multi-view scenes, built on `ontic-lib`: one
example schema (`ontic_data.example.BatchedTempExample`), a shared
`TemporalSceneDataset` skeleton, view sampling and collation, and six
registered datasets (`ontic_data.DATASETS`: `genesis`, `hocap`, `taco`,
`dextris`, `physinone`, `synthrobot`) plus a `MixedDatasetCfg`. Every module
imports with `torch` + `numpy` + `einops` + `pyyaml`; format readers are lazy,
explicit extras.

## Install

As a git dependency of an experiment (the package lives in a subdirectory of
the `ontic-lib` workspace):

```toml
[project]
dependencies = ["ontic-data[video,tables]"]

[tool.uv.sources]
ontic-data = { git = "<repo url>", subdirectory = "packages/ontic-data" }
```

For co-development, point the source at a checkout instead:

```toml
[tool.uv.sources]
ontic-data = { path = "../lib/packages/ontic-data", editable = true }
```

Inside this repository it is a workspace member and comes with the `dev`
dependency group.

## Extras

| extra | package | enables |
| --- | --- | --- |
| `ontic-data[video]` | `torchcodec` | `VideoReader(backend="torchcodec")`: taco, dextris, synthrobot |
| `ontic-data[decord]` | `decord` (Python < 3.12) | `VideoReader(backend="decord")` |
| `ontic-data[hdf5]` | `h5py` | synthrobot trajectory stores |
| `ontic-data[opencv]` | `opencv-python-headless` | hand overlays, hocap frame decoding |
| `ontic-data[hocap]` | `hdf5` + `opencv` + `scipy` | the HO-Cap loader |
| `ontic-data[tables]` | `pandas` | dataset meta CSVs |
| `ontic-data[images]` | `pillow` | genesis, physinone frames |
| `ontic-data[robot]` | `mujoco` | `ontic_data.robot` kinematics |
| `ontic-data[all]` | all of the above | |

## Documentation

The example schema, `DatasetCfg.build(stage, step_fn=..., horizon_fn=...)`,
the registry with per-dataset roots and extras, and per-module one-liners are
in [`docs/ontic_data.md`](../../docs/ontic_data.md); camera conventions are in
[`docs/ontic_lib.md`](../../docs/ontic_lib.md).
