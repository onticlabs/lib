# ontic_viz — the backbone viewer

`ontic_viz` (`packages/ontic-viz`) holds interactive inspection tools over
`ontic-nn` and `ontic-data`. There is one app today, `ontic_viz.backbone_viewer`:
a headless [viser](https://viser.studio) web GUI that runs any backbone in
`ontic_nn.wrappers.BACKBONES` (and any metric model in
`ontic_nn.metric_depth.METRIC_MODELS`) on any dataset in `ontic_data.DATASETS`,
and shows the lifted point cloud next to the GT camera frustums, with hand
skeletons, a workspace box, an optional robot overlay, a ruler, and occupancy
reprojection back into the cameras. The package README
(`packages/ontic-viz/README.md`) has the GUI walkthrough.

## Running

```bash
uv run ontic-backbone-viewer --port 8080 --device cuda
uv run ontic-backbone-viewer --share            # public share.viser.studio URL, kept alive
uv run ontic-backbone-viewer --hocap-root /data/hocap --stage val
```

Flags: `--host`, `--port` (viser walks up from it when taken; trust the printed
port), `--device cuda|cuda:N|cpu`, `--stage train|val|test`, `--share`,
`--presets <json>`, and one `--<dataset>-root <path>` per registry key
(defaults in `data_source.DEFAULT_ROOTS`, dev-box `/mnt/fast/...` paths).
Named presets (`ViewConfig`: stride, drop-lowest-conf %, voxel size, SFC
stride, FPS budget, point size, colour mode, alignment mode, workspace box)
live in `~/.config/ontic/backbone_viewer_presets.json`, overridable with
`--presets` or `ONTIC_BACKBONE_VIEWER_PRESETS`; a `<dataset>_default` entry is
seeded from `config.DATASET_DEFAULTS` only when missing. The robot overlay
needs `ontic-viz[robot]` (mujoco) and the robotics checkout
(`ONTIC_ROBOTICS_REPO` or a sibling directory).

## Alignment modes

Backbones return raw predicted-scale depth plus predicted cameras (when they
have a camera head); the runner keeps both camera sets so metric alignment is
a re-renderable choice, implemented once in `ontic_lib.pointops.align` /
`compute_alignment` and exposed as `runner.AlignMode`:

| mode | what it does |
| --- | --- |
| `none` | Unproject in the backbone's own frame (GT cameras for monocular models) |
| `sim3_points` (default) | Unproject with the predicted cameras, map the cloud into GT with a Umeyama Sim(3) fit of predicted → GT cameras |
| `prescale_gt` | Take only the Sim(3) scale, rescale depth, unproject with the GT cameras (what training does) |
| `metric_mono` | Global scale from a metric monocular model, rescale depth and predicted centres, rigid SE(3) fit to GT cameras |

## Adding a backbone or dataset

The viewer lists whatever the registries contain. A new backbone is a
`<Name>BackboneConfig` registered with `ontic_nn.wrappers.register_backbone`;
if it needs GT depth, is monocular, or conditions on GT cameras, add the GUI
hint to `runner.BACKBONE_NEEDS_GT_DEPTH` / `BACKBONE_IS_MONOCULAR` /
`BACKBONE_ACCEPTS_GT_CAMERAS`. A metric model registers in
`ontic_nn.metric_depth.METRIC_MODELS`. A new dataset registers its `DatasetCfg`
in `ontic_data.DATASETS` (it must build a `TemporalSceneDataset`); the CLI gets
`--<name>-root` automatically — add a default to `data_source.DEFAULT_ROOTS`,
GT-depth availability to `data_source.HAS_GT_DEPTH`, a view preset to
`config.DATASET_DEFAULTS`, and a `data_source.CFG_KWARGS` entry if the config
does not take `root=` (genesis takes `roots=`).

## Module map (`ontic_viz.backbone_viewer`)

| module | one line |
| --- | --- |
| `app` | `BackboneViewer`: the viser GUI wiring over the headless core |
| `runner` | `BackboneRunner` (one backbone + one metric model resident on the GPU, freed on switch), `BackboneResult`, `AlignMode`, `metric_unproject`, `build_backbone` / `build_metric_model` |
| `render` | Pure helpers, no viser / GPU: colour modes, `frustum_params`, `build_point_cloud`, `pred_cameras_in_display_frame`, `conf_to_rgb` |
| `config` | `ViewConfig`, `DATASET_DEFAULTS`, the preset JSON store (`load_presets`, `save_preset`, `ensure_default_presets`) |
| `data_source` | `Frame` (all cameras of one timestep), `GenericSource` / `build_source` over `ontic_data.DATASETS`, `DEFAULT_ROOTS`, `HAS_GT_DEPTH`, `CFG_KWARGS` |
| `robot_model` | `DuoRobotModel`: FR3 Duo link geometry posed from recorded joint angles (mujoco) |
| `share_tunnel` | `patch_viser_tunnel` / `start_share_watchdog`: keep the share URL alive, re-issue on drop |
| `cli` | `build_parser` / `main` — the `ontic-backbone-viewer` entry point |
