"""TrackCraft3R adapter: dense reference-frame 3D trajectories from one RGB-D view.

Upstream is https://github.com/cvlab-kaist/TrackCraft3r (pinned revision below), a Wan2.1-based
video model that regresses, for **every pixel of the reference frame**, a trajectory through the
clip. The released entry point is ``evaluation.wan_scene_flow_predictor.WanSceneFlowPredictor``::

    predictor = WanSceneFlowPredictor(checkpoint_path=..., model_id="Wan-AI/Wan2.1-T2V-1.3B", ...)
    tracks = predictor.predict(images_pil, query_uv, visibility, intrinsics,
                               depth_map=..., extrinsics_w2c=...)   # -> (T, M, 3) numpy

which samples that dense field at ``query_uv`` and returns trajectories in **frame-0 camera
coordinates**, in the units of the depth map that was passed in (the model's internal
percentile/centroid normalisation is undone inside ``predict``). This adapter turns Ontic's
``PointQueries`` into that reference-frame sampling and maps the result back to the geometry's
world frame with the frame-0 camera-to-world matrix. Nothing is re-anchored or rescaled: the only
transform applied to the upstream output is that one rigid ``c2w`` multiplication.

What the native model constrains, and how it is surfaced here:

* **Single view.** Monocular; ``V > 1`` is rejected rather than looped over.
* **Query time.** The dense field is defined on one reference frame, so every query must sit at
  ``time == 0``. ``arbitrary_query_times=False``; a later query time needs a second pass over a
  re-sliced clip, which this version does not do implicitly.
* **Resolution.** The released configuration is ``480x832``. Both sides must be multiples of 16
  (Wan2.1 VAE downsamples 8x, the DiT patch is 2). Inputs are resized onto that grid here, with
  mask-aware depth resampling, so upstream's own stretch-resize is a no-op.
* **Clip length.** Released demo and evaluation both run 12-frame clips; other lengths need
  ``allow_variable_clip_length=True`` and are not covered by the released evaluation.
* **Depth is mandatory.** ``predict`` asserts ``depth_map`` and ``extrinsics_w2c`` are present,
  and unprojects *every* pixel, so invalid depth is filled (per-frame median) before the call and
  the filled fraction is reported in the output metadata.
* **No gradients.** ``predict`` is a numpy evaluation path, so ``freeze_tracker=False`` is
  rejected instead of silently returning tensors that carry no graph.

Upstream's top-level packages are named ``evaluation`` and ``diffsynth``. ``evaluation`` is
generic enough to collide, so the loader refuses to run when either name is already taken by
something else instead of overwriting ``sys.modules``.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterator

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from ontic_nn.wrappers.common import import_research_module, offline_guard, resolve_checkpoint

from .common import (
    GeometrySequence,
    PointQueries,
    TrackerBase,
    TrackerCapabilities,
    TrackerConfig,
    TrackerOutput,
    empty_tracker_output,
    make_tracker_output,
    pixel_intrinsics,
    project_queries,
    resize_tracker_inputs,
    validate_tracker_inputs,
)
from .registry import register_tracker

TRACKCRAFT3R_REPO_URL = "https://github.com/cvlab-kaist/TrackCraft3r"
TRACKCRAFT3R_REVISION = "21e8fcaf4b6375b3044cead210d5808e1d81760b"  # main, 2026-05-14
TRACKCRAFT3R_HF_REPO = "trackcraft3r/checkpoint"
TRACKCRAFT3R_CHECKPOINT = "model.safetensors"
PREDICTOR_MODULE = "evaluation.wan_scene_flow_predictor"

# Wan2.1 VAE downsamples by 8 and the DiT patch is 2, so both image sides must be 16-divisible.
GRID_MULTIPLE = 16

# Upstream's two top-level package names, each with a file that identifies the checkout.
_PACKAGE_MARKERS = {
    "evaluation": Path("wan_scene_flow_predictor.py"),
    "diffsynth": Path("pipelines/wan_video_new.py"),
}

_CAPABILITIES = TrackerCapabilities(
    multiview=False,
    dense=True,
    arbitrary_query_times=False,
    execution_mode="offline",
    visibility_scope="query_view",
)


# ---------------------------------------------------------------------------
# Config / wrapper
# ---------------------------------------------------------------------------
@register_tracker("trackcraft3r")
@dataclass(kw_only=True)
class TrackCraft3RConfig(TrackerConfig):
    """Config for the released TrackCraft3R predictor.

    ``checkpoint_path`` points at ``model.safetensors`` (a file, or a directory containing it);
    otherwise it is fetched from the ``trackcraft3r/checkpoint`` HF repo into ``cache_dir``. The
    Wan2.1 base model is a *separate*, much larger download that ``diffsynth`` pulls through
    ModelScope: point ``base_model_cache_dir`` at a pre-populated directory (the upstream README
    uses ``MODELSCOPE_CACHE=./checkpoints/wan_models``). ``allow_download=False`` sets both
    HuggingFace and ModelScope offline flags, requiring all model assets to be cached.

    ``repo_path`` is a TrackCraft3R checkout to import from when it is not already installed.
    ``long_side`` is inherited and unused: the grid is fixed by ``height`` / ``width``.
    """

    repo_path: str | None = None
    model_id: str = "Wan-AI/Wan2.1-T2V-1.3B"
    base_model_cache_dir: str | None = None
    hf_repo_id: str = TRACKCRAFT3R_HF_REPO
    checkpoint_filename: str = TRACKCRAFT3R_CHECKPOINT
    device: str = "cuda"

    # Native input grid and clip length (released demo + evaluation defaults).
    long_side: int = 832  # informational; resizing uses height/width
    height: int = 480
    width: int = 832
    num_frames: int = 12
    allow_variable_clip_length: bool = False

    # Upstream model knobs, forwarded verbatim to WanSceneFlowPredictor.
    lora_rank: int = 1024
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"
    regression_timestep: int = -1
    track_latent_length: int = 12
    diag_max_depth: float = 80.0
    pj_norm_percentile_lo: float = 2.0
    pj_norm_percentile_hi: float = 98.0

    # Adapter-side checks.
    query_depth_tolerance: float | None = 0.05
    intrinsics_tolerance_px: float = 0.5

    CAPABILITIES: ClassVar[TrackerCapabilities] = _CAPABILITIES

    def build(self) -> TrackCraft3R:
        return TrackCraft3R(self)


class TrackCraft3R(TrackerBase):
    """Samples TrackCraft3R's dense reference-frame field at the supplied query pixels.

    ``predictor`` is the upstream ``WanSceneFlowPredictor``; leaving it ``None`` loads the real
    one (weights included). Tests inject a stand-in that mirrors the released call signature.
    """

    CAPABILITIES: ClassVar[TrackerCapabilities] = _CAPABILITIES

    def __init__(self, cfg: TrackCraft3RConfig, predictor: Any | None = None) -> None:
        super().__init__()
        if not cfg.freeze_tracker:
            raise ValueError(
                "TrackCraft3R's released predictor is a numpy evaluation path with no autograd "
                "graph, so it cannot be finetuned through this wrapper; keep freeze_tracker=True"
            )
        for name in ("height", "width"):
            size = getattr(cfg, name)
            if size <= 0 or size % GRID_MULTIPLE:
                raise ValueError(
                    f"{name}={size} must be a positive multiple of {GRID_MULTIPLE} "
                    "(Wan2.1 VAE 8x downsampling and DiT patch 2); the released grid is 480x832"
                )
        if cfg.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        self.predictor = predictor if predictor is not None else load_predictor(cfg)
        self._configure_model(_PredictorModule(self.predictor), cfg)

    # -- input restrictions -------------------------------------------------
    def _check_clip(self, t: int) -> None:
        cfg: TrackCraft3RConfig = self.cfg  # type: ignore[assignment]
        if not cfg.allow_variable_clip_length and t != cfg.num_frames:
            raise ValueError(
                f"TrackCraft3R was released and evaluated on {cfg.num_frames}-frame clips, got "
                f"T={t}; slice the clip, set num_frames={t}, or pass "
                "allow_variable_clip_length=True to run an unevaluated length"
            )

    def _clip_intrinsics(self, geometry: GeometrySequence) -> Tensor:
        """One pixel ``K`` per batch item; upstream takes a single ``[fx,fy,cx,cy]`` per clip.

        ``integer_centers=True`` matches upstream's ``np.arange(W)`` unprojection grid, which has
        no half-pixel offset: ``(u - cx) / fx`` with ``u`` integer equals Ontic's normalized
        ``((x + 0.5) / W - cx_n) / fx_n``.
        """
        cfg: TrackCraft3RConfig = self.cfg  # type: ignore[assignment]
        k = pixel_intrinsics(geometry.intrinsics[:, :, 0], cfg.height, cfg.width)
        if (k[..., 0, 1].abs() > 1e-6).any() or (k[..., 1, 0].abs() > 1e-6).any():
            raise ValueError("TrackCraft3R's [fx,fy,cx,cy] camera API cannot represent skew")
        drift = (k - k[:, :1]).abs().amax().item()
        if drift > cfg.intrinsics_tolerance_px:
            raise ValueError(
                "TrackCraft3R takes one [fx,fy,cx,cy] for the whole clip, but the supplied "
                f"intrinsics vary by {drift:.3f} px over time (tolerance "
                f"{cfg.intrinsics_tolerance_px} px); use a clip with constant intrinsics"
            )
        return k[:, 0]

    # -- forward ------------------------------------------------------------
    def forward(
        self, images: Tensor, queries: PointQueries, *, geometry: GeometrySequence
    ) -> TrackerOutput:
        cfg: TrackCraft3RConfig = self.cfg  # type: ignore[assignment]
        b, t, _v, n = validate_tracker_inputs(images, queries, geometry, multiview=False)
        self._check_clip(t)
        if (queries.time != 0).any():
            raise ValueError(
                "TrackCraft3R samples one dense reference frame, so every query time must be 0 "
                "(arbitrary_query_times=False); re-slice the clip so each query starts at its "
                "own frame 0 and run one pass per reference frame"
            )
        if n == 0:
            return empty_tracker_output(
                images, queries, geometry, visibility_scope=_CAPABILITIES.visibility_scope
            )

        uv = project_queries(queries, geometry, depth_tolerance=cfg.query_depth_tolerance)
        height, width = cfg.height, cfg.width
        rgb, depth, depth_valid = resize_tracker_inputs(images, geometry, size=(height, width))
        intrinsics_px = self._clip_intrinsics(geometry)
        scale = uv.new_tensor([width, height])

        tracks: list[Tensor] = []
        visibilities: list[Tensor | None] = []
        valids: list[Tensor] = []
        invalid_depth: list[float] = []
        depth_above_limit: list[float] = []

        with self._grad_context():
            for i in range(b):
                c2w = geometry.extrinsics[i, :, 0].detach().float()
                # Upstream normalises cameras to frame 0 (``load_npz_data(normalize_cam=True)``);
                # doing it here keeps world translations out of the float32 pipeline. The output
                # frame is camera 0 either way.
                extrinsics_w2c = torch.linalg.inv(c2w) @ c2w[0]
                depth_np, invalid_fraction, clipped_fraction = _dense_depth(
                    depth[i, :, 0], depth_valid[i, :, 0], cfg.diag_max_depth
                )
                k = intrinsics_px[i].cpu().numpy()
                query_uv = (uv[i].detach().float() * scale).cpu().numpy().astype(np.float64)

                predicted = self.predictor.predict(
                    _pil_frames(rgb[i, :, 0]),
                    query_uv,
                    np.ones((t, n), dtype=bool),
                    np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2]], dtype=np.float64),
                    depth_map=depth_np,
                    extrinsics_w2c=extrinsics_w2c.cpu().numpy().astype(np.float32),
                )

                predicted = np.asarray(predicted, dtype=np.float32)
                if predicted.shape != (t, n, 3):
                    raise RuntimeError(
                        f"{PREDICTOR_MODULE} returned {predicted.shape}, expected {(t, n, 3)}. "
                        "Every query projects strictly inside the reference frame, so upstream "
                        "should not have dropped any as out of bounds; check the checkout "
                        f"against pinned revision {TRACKCRAFT3R_REVISION}"
                    )
                cam0 = torch.from_numpy(np.ascontiguousarray(predicted)).to(images.device)
                c2w0 = c2w[0].to(images.device)
                tracks.append(cam0 @ c2w0[:3, :3].mT + c2w0[:3, 3])

                # A query whose reference pixel had no valid depth sits on filled-in geometry, so
                # its trajectory has no supported surface. project_queries already rejects these
                # unless query_depth_tolerance is None.
                x, y = _pixel_index(uv[i], width, height)
                valids.append(depth_valid[i, 0, 0][y, x].expand(t, n))
                visibilities.append(_sample_visibility(self.predictor, uv[i], t))
                invalid_depth.append(invalid_fraction)
                depth_above_limit.append(clipped_fraction)

        visibility = (
            torch.stack([v for v in visibilities if v is not None]).to(images.device)
            if all(v is not None for v in visibilities)
            else None
        )
        metadata = {
            "tracker": "trackcraft3r",
            "upstream_repo": TRACKCRAFT3R_REPO_URL,
            "upstream_api_revision": TRACKCRAFT3R_REVISION,
            "checkpoint_source": {
                "path": cfg.checkpoint_path,
                "repo_id": cfg.hf_repo_id,
                "filename": cfg.checkpoint_filename,
            },
            "model_id": cfg.model_id,
            "native_grid": (height, width),
            "clip_length": t,
            "reference_frame": 0,
            "dense_query_source": "reference-frame UV sampling of the predicted dense field",
            "upstream_output_frame": "camera_0",
            "depth_invalid_fraction": tuple(invalid_depth),
            "input_depth_above_diag_limit_fraction": tuple(depth_above_limit),
            "diag_max_depth": cfg.diag_max_depth,
            "visibility_source": (
                f"{PREDICTOR_MODULE}._last_vis_dense" if visibility is not None else "unavailable"
            ),
        }
        return make_tracker_output(
            queries,
            geometry,
            torch.stack(tracks),
            visibility,
            visibility_scope=_CAPABILITIES.visibility_scope,
            valid=torch.stack(valids),
            metadata=metadata,
        )


class _PredictorModule(nn.Module):
    """``nn.Module`` view over the upstream predictor's torch submodules.

    ``WanSceneFlowPredictor`` is a plain object holding a ``WanVideoPipeline``, so registering the
    pipeline's modules here is what lets ``_configure_model`` freeze and ``eval()`` them.

    Device moves include the cached null prompt and update the plain predictor's
    device fields. Its inference inputs and active weights use bfloat16.
    """

    # The text encoder is offloaded after computing the prompt; it is unused in
    # tracking and must not be moved back to CUDA with the active modules.
    PIPE_MODULES = ("dit", "vae", "vae_pj", "vae_vis")

    def __init__(self, predictor: Any) -> None:
        super().__init__()
        self.predictor = predictor
        pipe = getattr(predictor, "pipe", None)
        for name in self.PIPE_MODULES:
            module = getattr(pipe, name, None)
            if isinstance(module, nn.Module):
                self.add_module(name, module)
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, dtype=torch.bfloat16, device=getattr(predictor, "device", "cpu")),
            persistent=False,
        )
        context = getattr(predictor, "_null_context", None)
        if isinstance(context, Tensor):
            self.register_buffer("null_context", context, persistent=False)

    def _apply(self, fn, recurse=True):
        if fn(self._device_anchor).dtype != torch.bfloat16:
            raise TypeError(
                "TrackCraft3R requires bfloat16; move the device without changing dtype"
            )
        super()._apply(fn, recurse=recurse)
        self.predictor.device = self._device_anchor.device
        pipe = getattr(self.predictor, "pipe", None)
        if pipe is not None:
            pipe.device = self._device_anchor.device
        if hasattr(self.predictor, "parallel_vae_decode"):
            self.predictor.parallel_vae_decode = self._device_anchor.device.type == "cuda"
        if hasattr(self, "null_context"):
            self.predictor._null_context = self.null_context
        return self


# ---------------------------------------------------------------------------
# Tensor <-> upstream conversions
# ---------------------------------------------------------------------------
def _pixel_index(uv: Tensor, width: int, height: int) -> tuple[Tensor, Tensor]:
    """Normalized edge-origin ``(N,2)`` → the integer pixel upstream's ``astype(int)`` selects."""
    x = (uv[..., 0] * width).long().clamp(0, width - 1)
    y = (uv[..., 1] * height).long().clamp(0, height - 1)
    return x, y


def _pil_frames(rgb: Tensor) -> list:
    """``(T,3,H,W)`` in ``[0,1]`` → the list of PIL RGB images ``predict`` expects."""
    try:
        from PIL import Image
    except ImportError as e:  # pragma: no cover - exercised only without Pillow
        raise ImportError(
            "TrackCraft3R's predict() takes PIL images, which needs the `pillow` package; "
            "install ontic-nn[trackcraft3r]"
        ) from e
    frames = (rgb.detach().float().clamp(0, 1) * 255).round().to(torch.uint8)
    frames = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # (T,H,W,3), C-contiguous
    return [Image.fromarray(frame) for frame in frames]


def _dense_depth(depth: Tensor, valid: Tensor, max_depth: float) -> tuple[np.ndarray, float, float]:
    """``(T,H,W)`` masked camera-z → the dense float32 map upstream unprojects, plus statistics.

    ``predict`` unprojects every pixel and has no mask input, so holes are filled with the
    per-frame median of the valid depths; leaving them at zero would drag upstream's 2nd/98th
    percentile normalisation towards a cluster of points at the camera centre. Report filled
    pixels and input camera-z above the configured limit. The latter is only a diagnostic:
    upstream clips z *after* transforming points to camera 0, not this input depth map.
    """
    filled = depth.detach().float().clone()
    invalid = ~valid
    if invalid.any():
        for i in range(filled.shape[0]):
            frame_valid = valid[i]
            if not frame_valid.any():
                raise ValueError(
                    f"frame {i} has no valid depth, and TrackCraft3R needs a dense depth map "
                    "for every frame; drop the clip or supply a denser geometry source"
                )
            filled[i] = torch.where(frame_valid, filled[i], filled[i][frame_valid].median())
    clipped = (filled > max_depth).float().mean().item() if max_depth > 0 else 0.0
    return filled.cpu().numpy().astype(np.float32), invalid.float().mean().item(), clipped


def _sample_visibility(predictor: Any, uv: Tensor, t: int) -> Tensor | None:
    """Query-view visibility from the predictor's dense map, when it carries a time axis.

    Upstream keeps the visibility head's output on ``self._last_vis_dense`` rather than returning
    it. Only a ``(T, H, W)`` map is usable per timestep; anything else (including a single 2-D
    map) is reported as unavailable rather than broadcast into per-frame scores. Values are
    passed through unclamped: upstream applies a sigmoid, so anything outside ``[0,1]`` means the
    attribute is not the probability map assumed here and should fail the output check loudly.
    """
    dense = getattr(predictor, "_last_vis_dense", None)
    if dense is None:
        return None
    dense = np.asarray(dense)
    if dense.ndim != 3 or dense.shape[0] != t:
        return None
    x, y = _pixel_index(uv.detach().float().cpu(), dense.shape[2], dense.shape[1])
    sampled = dense[:, y.numpy(), x.numpy()]
    return torch.from_numpy(sampled.astype(np.float32))


# ---------------------------------------------------------------------------
# Upstream import and construction
# ---------------------------------------------------------------------------
def _module_dir(module: Any) -> Path | None:
    paths = list(getattr(module, "__path__", None) or [])
    if paths:
        return Path(paths[0]).resolve()
    file = getattr(module, "__file__", None)
    return Path(file).resolve().parent if file else None


def guard_upstream_packages(repo_path: Path | None) -> None:
    """Refuse to import when TrackCraft3R's generic top-level names are already taken.

    Upstream ships ``evaluation`` and ``diffsynth`` as top-level packages. Replacing an unrelated
    ``evaluation`` in ``sys.modules`` would break whatever imported it first, so this raises with
    the conflicting location instead.
    """
    for package, marker in _PACKAGE_MARKERS.items():
        module = sys.modules.get(package)
        if module is None:
            continue
        directory = _module_dir(module)
        if directory is not None and (directory / marker).is_file():
            if repo_path is None or directory.parent == repo_path:
                continue
            raise ImportError(
                f"a different TrackCraft3R checkout is already imported as {package!r} from "
                f"{directory.parent}; one process cannot also load repo_path {repo_path}"
            )
        raise ImportError(
            f"the top-level package {package!r} is already imported from {directory} and is not "
            f"part of TrackCraft3R ({TRACKCRAFT3R_REPO_URL}), which ships its code as "
            "'evaluation' and 'diffsynth'. Python cannot hold both under one name: run "
            "TrackCraft3R in its own process, or install the checkout so its packages resolve"
        )


@contextlib.contextmanager
def _environment(values: dict[str, str]) -> Iterator[None]:
    previous = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_predictor(cfg: TrackCraft3RConfig) -> Any:
    """Import upstream, resolve the trained weights and construct ``WanSceneFlowPredictor``."""
    repo_path = Path(cfg.repo_path).expanduser().resolve() if cfg.repo_path else None
    if repo_path is not None:
        marker = repo_path / "evaluation" / _PACKAGE_MARKERS["evaluation"]
        if not marker.is_file():
            raise FileNotFoundError(
                f"repo_path {repo_path} is not a TrackCraft3R checkout (no {marker}); clone "
                f"{TRACKCRAFT3R_REPO_URL}"
            )
    guard_upstream_packages(repo_path)
    if repo_path is not None and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    module = import_research_module(
        PREDICTOR_MODULE,
        extra="trackcraft3r",
        repo_url=TRACKCRAFT3R_REPO_URL,
        what="TrackCraft3R",
    )
    resolved = _module_dir(module)
    if repo_path is not None and resolved is not None and resolved.parent != repo_path:
        raise ImportError(
            f"{PREDICTOR_MODULE!r} resolved to {resolved}, not repo_path {repo_path}; another "
            "'evaluation' package shadows the checkout"
        )

    checkpoint = resolve_checkpoint(
        cfg.checkpoint_path,
        cfg.hf_repo_id,
        filename=cfg.checkpoint_filename,
        cache_dir=cfg.cache_dir,
        allow_download=cfg.allow_download,
        extra="trackcraft3r",
        what="TrackCraft3R checkpoint",
    )
    if checkpoint is None:
        raise ValueError(
            "TrackCraft3R has no meaningful random initialisation; set checkpoint_path or keep "
            f"hf_repo_id pointing at {TRACKCRAFT3R_HF_REPO}/{TRACKCRAFT3R_CHECKPOINT}"
        )

    env = {}
    if cfg.base_model_cache_dir:
        env["MODELSCOPE_CACHE"] = str(Path(cfg.base_model_cache_dir).expanduser())
    if not cfg.allow_download:
        # The ModelConfig used by wan_video_new lives in diffsynth/utils/__init__.py.
        # It forwards this flag as local_files_only to ModelScope (including tokenizer assets).
        env["MODELSCOPE_OFFLINE"] = "1"
    with _environment(env), offline_guard(cfg.allow_download):
        return module.WanSceneFlowPredictor(
            checkpoint_path=checkpoint,
            model_id=cfg.model_id,
            lora_rank=cfg.lora_rank,
            lora_target_modules=cfg.lora_target_modules,
            height=cfg.height,
            width=cfg.width,
            device=cfg.device,
            regression_timestep=cfg.regression_timestep,
            track_latent_length=cfg.track_latent_length,
            # Inputs are already on the native grid, so upstream's resize is the identity.
            resize_mode="stretch",
            diag_max_depth=cfg.diag_max_depth,
            pj_norm_percentile_lo=cfg.pj_norm_percentile_lo,
            pj_norm_percentile_hi=cfg.pj_norm_percentile_hi,
            apply_speed_opts=False,  # do not change process-wide cuDNN/TF32 settings
            parallel_vae_decode=torch.device(cfg.device).type == "cuda",
        )
