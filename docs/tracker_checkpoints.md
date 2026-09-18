# Local tracker checkpoints

For a portable code/dependency installation, use
`uv run --no-sync python scripts/install_models.py` from the workspace root;
see [model installation](../packages/ontic-nn/README.md#model-installation).
This page records optional checkpoint locations on AL3, not prerequisites or
default paths for other machines. The installer does not download weights.

The AL3 viewer uses released checkpoints for all four tracker adapters. Persistent
copies live under `/mnt/fast/mzhobro/ontic_tracker_checkpoints` (about 35.8 GB total).
Large model files are outside Git. `manifest.json` in that directory records file
sizes, SHA-256 hashes, and pinned Hugging Face revisions.

| Tracker | Checkpoint relative to that directory | Size | Published source |
|---|---|---:|---|
| MVTracker | `mvtracker/mvtracker_200000_june2025.pth` | 90.5 MB | [ethz-vlg/mvtracker](https://huggingface.co/ethz-vlg/mvtracker/tree/010d5d114e860aae6b2568104927b636cdca01bc) |
| TAPIP3D | `tapip3d/tapip3d_final.pth` | 309.4 MB | [zbww/tapip3d](https://huggingface.co/zbww/tapip3d/tree/08730a588204f258f7a86530df15b2a6426e5f5b) |
| CoTracker3 | `cotracker3/scaled_online.pth` | 101.7 MB | [facebook/cotracker3](https://huggingface.co/facebook/cotracker3/tree/bf55ea50d4390e1820a267f131cd6587240fb2c5) |
| TrackCraft3R | `trackcraft3r/model.safetensors` | 17.76 GB | [trackcraft3r/checkpoint](https://huggingface.co/trackcraft3r/checkpoint/tree/e397c1050bd540062cbd6642ba06a344ca80f116) |

CoTracker3 uses PointWorld's combination of `scaled_online.pth` with its offline
predictor settings; the checkpoint name does not change the clip interface.

TrackCraft3R also needs the [Wan2.1-T2V-1.3B base assets](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/37ec512624d61f7aa208f7ea8140a131f93afc9a):

```text
wan_models/Wan-AI/Wan2.1-T2V-1.3B/
  diffusion_pytorch_model.safetensors       # 5.68 GB
  models_t5_umt5-xxl-enc-bf16.pth           # 11.36 GB
  Wan2.1_VAE.pth                            # 507.6 MB
  config.json
  google/umt5-xxl/
    special_tokens_map.json
    spiece.model
    tokenizer.json
    tokenizer_config.json
```

The viewer's **Wan base model cache** points to the `wan_models` directory,
above `Wan-AI`, rather than the model subdirectory. The tokenizer files are
required even though tracking uses an empty text prompt.

## Viewer configuration

Under **3. Tracking → Model files**, selecting a tracker restores its own
checkpoint and research checkout. **Download missing weights** stays disabled;
local inference does not require network access. Pass `--model-config` with
`/mnt/fast/mzhobro/ontic_tracker_checkpoints/viewer-models.json` when launching
the viewer to load these local paths.

The source checkouts are `/tmp/ontic-mvtracker-source`,
`/tmp/ontic-tapip3d-source`, `/tmp/ontic-pointworld-source/third_party/co-tracker`,
and `/tmp/ontic-trackcraft3r-source`. See [point tracking](point_tracking.md) for
the pinned source revisions, optional dependencies, and model input constraints.
TrackCraft3R uses one selected camera and a 12-frame clip.

On AL3, the GPU runtime remains `frontier_world_model/.venv` with Torch 2.4.1.
TrackCraft3R's additional packages are isolated in `/tmp/ontic-trackcraft3r-deps`:
Transformers 4.57.6, PEFT 0.18.1, ModelScope 1.29.2, Hugging Face Hub 0.36.2,
SentencePiece, ftfy, and NVIDIA ML bindings. The download runtime uses filelock
3.18.0 because the host's SMB mount reports zero hard links, which causes newer
filelock's deleted-file detection to repeatedly retry. The shared environment
was not modified.

## Verification

All released weight files match their published SHA-256 hashes. Offline GPU
checks succeeded for all four trackers using these paths and a 12-frame HOCAP
sensor-depth clip with 32 queries. TrackCraft3R loaded its DiT with zero missing
or unexpected keys and returned finite trajectories for every query/frame.
Its first end-to-end run, including model loading from the network volume, took
629 seconds and peaked at 18.1 GB of allocated GPU memory on the RTX A5000.
This is a runtime check, not an accuracy evaluation or isolated inference timing.
The browser check also confirmed each tracker's prefilled paths and disabled
download setting. Diagnostics are saved beside the weights as `validation.json`.

For browser access, see the [viewer launch and SSH tunnel instructions](tracking_dataset_trials.md#viewing).
