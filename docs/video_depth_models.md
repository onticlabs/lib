# Video depth model priorities

Research date: 2026-09-17. The immediate problem is depth jumping between frames
in the calibrated 3D viewer. Priorities below are an integration judgment for this
workflow, not a universal leaderboard or measurements on the user's recordings.
The requirements are temporal context, usable released code/weights, metric
camera-z depth even without sensor depth, and compatibility with the existing
PyTorch environment. Changing model size does not constitute a new model family.

| Priority | Family | Evidence and fit | Decision |
| --- | --- | --- | --- |
| P0 | [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) | Dedicated temporal model with overlapping windows and released metric Small/Base/Large checkpoints. Already integrated; practical starting point for offline clips. | Keep all sizes. Compare Base/Large when memory permits; Small remains the economical default. |
| P0 | [VeloDepth](https://github.com/lpiccinelli-eth/velodepth) ([3DV 2026 paper](https://arxiv.org/abs/2512.10725)) | Native metric geometry with learned propagation between adaptive keyframes. Directly addresses temporal jumps. Authors report improved consistency and speed in their causal benchmark; keyframe changes can still cause jumps. | Highest-priority new video family. Preserve whole-sequence inference, isolate cameras, expose its native resolution bucket. |
| P1 | [DA3 Nested Giant-Large 1.1](https://github.com/ByteDance-Seed/Depth-Anything-3#-model-cards) | Any-view geometry plus a metric scaling branch. Jointly processing the time sequence can use cross-frame context; the current image-backbone loop cannot. Native metric output, but 1.4B parameters and joint attention make it heavier. Not specifically a motion-aware video architecture. | Add a joint-clip adapter using the refreshed 1.1 checkpoint. Useful comparison, especially for scenes dominated by static geometry. |
| P2 | [DepthCrafter 1.0.1](https://github.com/Tencent/DepthCrafter) | Strong offline diffusion baseline for coherent detailed video disparity. Official demo reports about 26GB at 1024×576 or 9GB at 512×256. Its relative disparity cannot be used directly as meters. | Defer until a clip-wide inverse-depth scale-and-shift calibration path is available. A scalar depth multiplier would be incorrect. |
| P2 | [GeometryCrafter](https://github.com/TencentARC/GeometryCrafter) | Video point maps with diffusion priors; official evaluation includes scale-invariant point maps and affine-invariant depth. Low-resolution inference is still roughly 20GB-class. | Defer metric anchoring and higher memory requirements; worth evaluating for offline 4D reconstruction. |
| P2 | [Online Video Depth Anything](https://arxiv.org/abs/2510.09182) | Causal feature caching and low memory are attractive for live/edge deployment. Paper targets non-metric depth. | Revisit for live streaming with an explicit metric calibration path. Current viewer prepares offline clips. |
| P3 | [FlashDepth](https://github.com/Eyeline-Labs/FlashDepth) | Real-time high-resolution temporal depth, but official setup requires its local Mamba fork and specifies PyTorch 2.4 compatibility. | Defer to an isolated runtime; do not downgrade the shared ML environment. |
| P3 | [RollingDepth](https://github.com/prs-eth/RollingDepth) | Coherent long videos from diffusion depth snippets and global alignment. Official installation requires a modified diffusers fork. | Defer the separate dependency stack and metric anchoring work. |
| Watch | [ICDepth](https://xuanhuahe.github.io/ICDepth/) ([ECCV 2026 paper](https://arxiv.org/abs/2607.01677)) | Recent promising diffusion-transformer result. A usable official code/checkpoint release was not established from the linked project page during this review. | Track availability; do not expose an unverified adapter. |

Benchmark numbers across papers are not directly comparable. In particular,
VeloDepth's main comparison uses causal inference and scale/shift-invariant
accuracy and consistency; that does not establish superiority over offline VDA
or diffusion inference on these recordings. VeloDepth also reports metric results
in its supplement. The next useful quality evaluation is the same moving-scene
clip, resolution, cameras, and metric calibration policy for each family.

All integrated adapters retain dataset camera calibration for lifting and tracking.
Each camera gets an independent temporal sequence; these adapters do not enforce
cross-camera agreement. The optional sensor calibration fits one scale per camera
across the complete clip. There is no per-frame rescaling or display normalization
that could conceal temporal drift. Large frame strides or cuts reduce temporal
overlap; sequence state must not carry between cameras or separate clip calls.

VDA uses upstream short-side inference size; DA3 uses a long-side limit and joint
clip context; VeloDepth uses its documented resolution level 0–9. These controls
must be shown with their actual meaning. DA3 processes the full clip jointly,
without independent chunks that could introduce scale seams; use shorter clips
when GPU memory is limited.

The source installer pins upstream revisions and installs inference support
dependencies without replacing the selected torch/CUDA stack. Weight downloads
remain opt-in. VeloDepth's upstream constructors also load ConvNeXt initializers;
offline operation requires those torch.hub files as well as the HF snapshot.
See the model repositories for their weight/code terms before redistribution.

## Implemented selection

The viewer's **2. Depth → Video depth → Depth source → Video depth · metric** path exposes
`vda_small`, `vda_base`, `vda_large`, `velodepth`, and `da3_nested`. All three
families return metric camera-z depth at the clip's cached image resolution.
VeloDepth and DA3 Nested were added following the review above; the source
installer includes their pinned repositories and inference dependencies.
Model-specific file paths, precision and resolution settings survive switching
models, participate in the geometry cache key, and are stored in recordings.

Automated test coverage includes ordered temporal
context, camera isolation, camera-z versus radial range, metric-branch checks,
offline auxiliary downloads, cancellation, clip-wide sensor scale, viewer model
switching, cache reuse and recording restore. Source imports and the dependency
lock check passed. The installer preserved the selected torch/CUDA stack.

Pretrained CPU smoke tests use three frames from the synthetic demo, with
96×128 output. They verify loading and inference, not temporal quality on real
recordings. CUDA inference remains unverified because this environment's
installed driver cannot initialize its torch CUDA 13 runtime.

| Pretrained adapter | Smoke-test inference setting | Result |
| --- | --- | --- |
| Video Depth Anything Small | Short side 140 | `(1,3,1,96,128)` depth, 100% finite and positive |
| VeloDepth | Native resolution level 0 | `(1,3,1,96,128)` depth, 100% finite and positive |
| DA3 Nested Giant-Large 1.1 | Long side 140, three frames jointly | `(1,3,1,96,128)` depth, 100% finite and positive |

Smoke-test HF weights are cached locally in `.venv/share/ontic-video-weights`;
VeloDepth's auxiliary initializers are in `.venv/share/ontic-video-torch`.
To reuse these caches, set `HF_HUB_CACHE` and `TORCH_HOME` to their absolute paths
before starting the viewer. API calls can alternatively set `cache_dir` for the
HF cache. Downloads remain disabled by default.
