"""MVTracker adapter: synchronized multi-view RGB-D point tracking (ICCV 2025).

Upstream https://github.com/ethz-vlg/mvtracker, inspected at revision
``ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072`` (2025-09-06). The wrapper drives
``mvtracker.models.evaluation_predictor_3dpt.EvaluationPredictor`` around
``mvtracker.models.core.mvtracker.mvtracker.MVTracker``, built with the same keyword
arguments ``hubconf._build_model`` / ``hubconf.mvtracker_predictor`` use for the released
``mvtracker_main`` checkpoint, so no ``torch.hub`` repo download or code execution is needed.

The native predictor differs from the Ontic contract in every axis, and each conversion is
done here explicitly:

* layout ``(B,V,T,...)`` with depth ``(B,V,T,1,H,W)``, not ``(B,T,V,...)``;
* RGB in ``0..255`` (the encoder divides by 255 itself — see :attr:`MVTrackerConfig.rgb_scale`);
* depth ``0`` as the invalid sentinel, not a separate mask;
* pixel intrinsics on *integer* pixel centres, not Ontic's normalised ``(x+0.5)/W``;
* world-to-camera ``3x4`` extrinsics, not camera-to-world ``4x4``;
* queries ``(B,N,4)`` ordered ``(t,x,y,z)`` in world space;
* batch size 1, so this wrapper loops over ``B``;
* aggregate ``any_view`` visibility, returned both thresholded and as a probability.

Trajectories come back in the geometry's own world frame and units: any scene
normalization applied on the way in (see :attr:`MVTrackerConfig.scene_normalization`) is
inverted on the way out and recorded in the output metadata.
"""

from __future__ import annotations

import contextlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ontic_lib.transforms.rigid import invert_rigid_transform

from ..wrappers.common import amp_dtype, import_research_module, resolve_checkpoint
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
    resize_tracker_inputs,
    validate_tracker_inputs,
)
from .registry import register_tracker

MVTRACKER_REPO_URL = "https://github.com/ethz-vlg/mvtracker"
MVTRACKER_REVISION = "ceea8ad2af77ed9b44148ef8e9eeba4ea3c3f072"  # inspected 2025-09-06
MVTRACKER_HF_REPO = "ethz-vlg/mvtracker"
MVTRACKER_CHECKPOINT = "mvtracker_200000_june2025.pth"  # hubconf `mvtracker_main`
CORE_MODULE = "mvtracker.models.core.mvtracker.mvtracker"
PREDICTOR_MODULE = "mvtracker.models.evaluation_predictor_3dpt"
EXTRA = "mvtracker"

SIZE_MULTIPLE = 16  # the upstream feature encoder downsamples by 16 before upsampling to H/stride
VISIBILITY_SCOPE = "any_view"


@register_tracker("mvtracker")
@dataclass(kw_only=True)
class MVTrackerConfig(TrackerConfig):
    """Released ``mvtracker_main`` settings; every default mirrors upstream ``hubconf``.

    ``long_side`` (512) reproduces the released ``interp_shape=(384, 512)`` for 4:3 input;
    set ``image_size`` to pin an exact ``(height, width)`` instead. The wrapper resizes the
    clip itself and passes ``interp_shape=None`` upstream, because the upstream resize scales
    the intrinsics by ``W_new/W_old`` without the matching half-pixel shift.

    ``scene_normalization`` decides what the model actually sees. The released checkpoint was
    trained and evaluated on scenes normalized to a working scale, and upstream's own
    entry point for arbitrary scenes (``GenericSceneDataset``) defaults to doing so;
    ``"camera_radius"`` reproduces that, ``"none"`` passes the geometry through untouched and
    ``"manual"`` applies ``scene_scale`` / ``scene_translation``. Whatever is applied is
    inverted on the returned trajectories, so the public frame and units never change.
    """

    CAPABILITIES: ClassVar[TrackerCapabilities] = TrackerCapabilities(
        multiview=True,
        dense=False,
        arbitrary_query_times=True,
        execution_mode="offline",
        visibility_scope=VISIBILITY_SCOPE,
    )

    model_dir: str = MVTRACKER_HF_REPO
    checkpoint_file: str = MVTRACKER_CHECKPOINT
    long_side: int = 512
    image_size: Optional[Tuple[int, int]] = None

    # --- mvtracker.models.core.mvtracker.mvtracker.MVTracker(...) ---
    sliding_window_len: int = 12
    stride: int = 4
    fmaps_dim: int = 128
    add_space_attn: bool = True
    num_heads: int = 6
    hidden_size: int = 256
    space_depth: int = 6
    time_depth: int = 6
    num_virtual_tracks: int = 64
    corr_n_groups: int = 1
    corr_n_levels: int = 4
    corr_neighbors: int = 16
    corr_add_neighbor_offset: bool = True
    corr_add_neighbor_xyz: bool = False
    corr_filter_invalid_depth: bool = False
    use_flash_attention: Optional[bool] = None  # None → SDPA when a CUDA device is present

    # --- mvtracker.models.evaluation_predictor_3dpt.EvaluationPredictor(...) ---
    visibility_threshold: float = 0.5
    grid_size: int = 4
    n_grids_per_view: int = 1
    local_grid_size: int = 18
    local_extent: int = 50
    single_point: bool = False  # one forward pass per query; slow, used by some eval settings
    n_iters: int = 6

    # --- adapter ---
    scene_normalization: Literal["camera_radius", "manual", "none"] = "camera_radius"
    scene_target_radius: float = 6.3
    scene_scale: Optional[float] = None
    scene_translation: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    rgb_scale: float = 255.0
    autocast: bool = True
    bidirectional: bool = True  # a second pass fills frames before each query time

    def resolved_use_flash_attention(self) -> bool:
        """``use_flash_attention`` with ``None`` resolved against the installed accelerators.

        This only selects the attention kernel (``F.scaled_dot_product_attention`` versus the
        explicit softmax): both upstream classes declare the same ``to_q``/``to_kv``/``to_out``
        parameters, so the released checkpoint loads either way.
        """
        if self.use_flash_attention is not None:
            return self.use_flash_attention
        return torch.cuda.is_available() and hasattr(F, "scaled_dot_product_attention")

    def build(self) -> MVTracker:
        return MVTracker(self)


class MVTracker(TrackerBase):
    """Ontic wrapper around the released MVTracker ``EvaluationPredictor``.

    ``model`` may be supplied directly (a module with the predictor's call signature) to
    bypass the upstream import; ``build()`` always constructs the real one.
    """

    CAPABILITIES: ClassVar[TrackerCapabilities] = MVTrackerConfig.CAPABILITIES

    def __init__(self, cfg: MVTrackerConfig, *, model: Optional[nn.Module] = None) -> None:
        super().__init__()
        _validate_config(cfg)
        self.checkpoint_info: Dict[str, Any] = {}
        self.upstream_info: Dict[str, Any] = {"api_revision": MVTRACKER_REVISION}
        if model is None:
            model = self._build_predictor(cfg)
        self._configure_model(model, cfg)

    # ------------------------------------------------------------------ build
    def _build_predictor(self, cfg: MVTrackerConfig) -> nn.Module:
        core = _import_upstream(CORE_MODULE, "MVTracker")
        predictor_module = _import_upstream(PREDICTOR_MODULE, "EvaluationPredictor")
        model = core.MVTracker(
            sliding_window_len=cfg.sliding_window_len,
            stride=cfg.stride,
            # Upstream's in-forward normalization calls transform_scene() with an undefined
            # name `T` at this revision, and never transforms the depth maps; scene handling
            # lives in `scene_normalization` instead.
            normalize_scene_in_fwd_pass=False,
            fmaps_dim=cfg.fmaps_dim,
            add_space_attn=cfg.add_space_attn,
            num_heads=cfg.num_heads,
            hidden_size=cfg.hidden_size,
            space_depth=cfg.space_depth,
            time_depth=cfg.time_depth,
            num_virtual_tracks=cfg.num_virtual_tracks,
            use_flash_attention=cfg.resolved_use_flash_attention(),
            corr_n_groups=cfg.corr_n_groups,
            corr_n_levels=cfg.corr_n_levels,
            corr_neighbors=cfg.corr_neighbors,
            corr_add_neighbor_offset=cfg.corr_add_neighbor_offset,
            corr_add_neighbor_xyz=cfg.corr_add_neighbor_xyz,
            corr_filter_invalid_depth=cfg.corr_filter_invalid_depth,
        )
        self.checkpoint_info = _load_checkpoint(model, cfg)
        knn = getattr(core, "knn", None)
        self.upstream_info["knn_backend"] = getattr(knn, "__name__", "unknown")
        self.upstream_info["use_flash_attention"] = cfg.resolved_use_flash_attention()
        return predictor_module.EvaluationPredictor(
            multiview_model=model,
            interp_shape=None,  # this wrapper resizes; see the class docstring
            visibility_threshold=cfg.visibility_threshold,
            grid_size=cfg.grid_size,
            n_grids_per_view=cfg.n_grids_per_view,
            local_grid_size=cfg.local_grid_size,
            local_extent=cfg.local_extent,
            single_point=cfg.single_point,
            sift_size=0,
            num_uniformly_sampled_pts=0,
            n_iters=cfg.n_iters,
        )

    # ---------------------------------------------------------------- forward
    def forward(
        self, images: Tensor, queries: PointQueries, *, geometry: GeometrySequence
    ) -> TrackerOutput:
        cfg: MVTrackerConfig = self.cfg
        batch, frames, _views, points = validate_tracker_inputs(
            images, queries, geometry, multiview=True
        )
        if points == 0:
            return empty_tracker_output(
                images, queries, geometry, visibility_scope=VISIBILITY_SCOPE
            )
        self._check_window_coverage(queries, frames)

        size = self._target_size(images.shape[-2], images.shape[-1])
        rgb, depth, depth_valid = resize_tracker_inputs(images, geometry, size=size)
        rgb = rgb * cfg.rgb_scale
        depth = torch.where(depth_valid, depth, depth.new_zeros(()))  # upstream invalid sentinel
        intrinsics = pixel_intrinsics(geometry.intrinsics, size[0], size[1])
        world_to_camera = invert_rigid_transform(geometry.extrinsics.float())[..., :3, :]

        tracks, visibility, transforms, backward_used = [], [], [], []
        for index in range(batch):
            scale, translation = self._scene_transform(
                geometry, depth, depth_valid, intrinsics, index
            )
            transforms.append((scale, tuple(round(float(x), 6) for x in translation)))
            extrinsics_i = _transform_world_to_camera(world_to_camera[index], scale, translation)
            query_xyz = queries.xyz_world[index].float() * scale + translation
            query_points = torch.cat(
                [queries.time[index, :, None].to(query_xyz), query_xyz], dim=-1
            )
            inputs = {
                "rgbs": _to_upstream(rgb[index]),
                # Camera-z scales with the scene; the zero invalid sentinel survives it.
                "depths": _to_upstream(depth[index] * scale).unsqueeze(3),
                "intrs": _to_upstream(intrinsics[index]),
                "extrs": _to_upstream(extrinsics_i),
            }
            before = (
                torch.arange(frames, device=images.device)[None, :, None]
                < queries.time[index : index + 1, None]
            )
            backward = cfg.bidirectional and bool(before.any())
            backward_used.append(backward)
            with self._grad_context(), self._autocast(images.device):
                traj, vis = _unpack_result(
                    self.model(**inputs, query_points_3d=query_points[None]), frames, points
                )
                if backward:
                    reverse_query = query_points.clone()
                    reverse_query[:, 0] = frames - 1 - reverse_query[:, 0]
                    reverse_start = 0 if cfg.grid_size > 0 else int(reverse_query[:, 0].min())
                    if reverse_start >= frames - cfg.sliding_window_len // 2:
                        raise ValueError(
                            "MVTracker's reverse sliding window would not run; enable the "
                            "support grid, use a longer clip, or set bidirectional=False"
                        )
                    reverse_traj, reverse_vis = _unpack_result(
                        self.model(
                            **{key: value.flip(2) for key, value in inputs.items()},
                            query_points_3d=reverse_query[None],
                        ),
                        frames,
                        points,
                    )
                    traj = torch.where(before[..., None], reverse_traj.flip(1), traj)
                    vis = torch.where(before, reverse_vis.flip(1), vis)
            device = queries.ids.device
            tracks.append((traj.float().to(device) - translation.to(device)) / scale)
            visibility.append(vis.float().to(device))

        arange = torch.arange(frames, device=queries.ids.device)
        return make_tracker_output(
            queries,
            geometry,
            torch.cat(tracks, dim=0),
            torch.cat(visibility, dim=0),
            visibility_scope=VISIBILITY_SCOPE,
            # A one-way native pass has no supported estimates before a query.
            valid=None if cfg.bidirectional else arange[None, :, None] >= queries.time[:, None, :],
            metadata={
                "tracker": "mvtracker",
                "upstream_repo": MVTRACKER_REPO_URL,
                "upstream": dict(self.upstream_info),
                "checkpoint": dict(self.checkpoint_info),
                "input_resolution": size,
                "rgb_scale": cfg.rgb_scale,
                "n_iters": cfg.n_iters,
                "sliding_window_len": cfg.sliding_window_len,
                "support_grid": {
                    "grid_size": cfg.grid_size,
                    "n_grids_per_view": cfg.n_grids_per_view,
                    "single_point": cfg.single_point,
                },
                "scene_normalization": {
                    "mode": cfg.scene_normalization,
                    "target_radius": cfg.scene_target_radius,
                    "per_batch_scale_translation": transforms,
                    "inverted_on_output": True,
                },
                "visibility": {
                    "scope": VISIBILITY_SCOPE,
                    "source": "vis_e_as_prob",
                    "kind": "uncalibrated probability",
                    "upstream_threshold": cfg.visibility_threshold,
                },
                "backward_pass": tuple(backward_used),
                "valid_rule": "finite bidirectional estimates"
                if cfg.bidirectional
                else "frame index >= query time",
            },
        )

    # ---------------------------------------------------------------- helpers
    def _target_size(self, height: int, width: int) -> Tuple[int, int]:
        cfg: MVTrackerConfig = self.cfg
        if cfg.image_size is not None:
            return (int(cfg.image_size[0]), int(cfg.image_size[1]))
        scale = cfg.long_side / max(height, width)
        return (_snap(height * scale), _snap(width * scale))

    def _check_window_coverage(self, queries: PointQueries, frames: int) -> None:
        """Reject clips where the upstream sliding-window loop would never run.

        Upstream starts at the earliest query time and iterates while
        ``start < T - sliding_window_len // 2``; if that is false on entry every trajectory
        stays at its zero placeholder. The support grid contributes queries at ``t = 0``.
        """
        cfg: MVTrackerConfig = self.cfg
        half = cfg.sliding_window_len // 2
        starts = (
            torch.zeros_like(queries.time[:, 0])
            if cfg.grid_size > 0
            else queries.time.min(dim=1).values
        )
        if bool((starts >= frames - half).any()):
            raise ValueError(
                f"MVTracker's sliding window starts at t={int(starts.max())} and runs while "
                f"start < T - sliding_window_len//2 = {frames - half}, so no window would run "
                f"and every trajectory would stay zero. Supply a clip longer than {half} "
                "frames, or query an earlier time."
            )

    def _autocast(self, device: torch.device):
        if not self.cfg.autocast or device.type != "cuda":
            return contextlib.nullcontext()
        return torch.autocast("cuda", dtype=amp_dtype())

    def _scene_transform(
        self,
        geometry: GeometrySequence,
        depth: Tensor,
        depth_valid: Tensor,
        intrinsics: Tensor,
        index: int,
    ) -> Tuple[float, Tensor]:
        """Similarity ``X' = scale * X + translation`` taking one sequence to the model's scale.

        ``"camera_radius"`` follows upstream ``compute_auto_scene_normalization`` with
        ``rescale_by_camera_radius=True``: centre on the first-frame depth centroid, then scale
        so the median camera distance from it is ``scene_target_radius``. It deliberately drops
        upstream's floor lift along ``+z`` (that assumes a z-up world with a visible floor,
        which the Ontic geometry contract does not promise) and uses true camera centres rather
        than upstream's world-to-camera translation column.
        """
        cfg: MVTrackerConfig = self.cfg
        device = depth.device
        if cfg.scene_normalization == "none":
            return 1.0, torch.zeros(3, device=device)
        if cfg.scene_normalization == "manual":
            return float(cfg.scene_scale), torch.tensor(
                cfg.scene_translation, dtype=torch.float32, device=device
            )

        centroid = _depth_centroid(
            depth[index, 0],
            depth_valid[index, 0],
            intrinsics[index, 0],
            geometry.extrinsics[index, 0].float(),
        )
        centers = geometry.extrinsics[index, 0, :, :3, 3].float()
        median = (centers - centroid).norm(dim=-1).median()
        if not bool(torch.isfinite(median)) or float(median) <= 0:
            raise ValueError(
                "scene_normalization='camera_radius' needs cameras at a positive median "
                f"distance from the depth centroid, got {float(median)!r}. Use "
                "scene_normalization='manual' with an explicit scale, or 'none'."
            )
        scale = cfg.scene_target_radius / float(median)
        return scale, -scale * centroid


# --------------------------------------------------------------------- utils
def _snap(value: float) -> int:
    return max(SIZE_MULTIPLE, int(round(value / SIZE_MULTIPLE)) * SIZE_MULTIPLE)


def _to_upstream(sequence: Tensor) -> Tensor:
    """One sequence ``(T,V,...)`` → the upstream ``(1,V,T,...)`` layout."""
    return sequence.transpose(0, 1).unsqueeze(0).contiguous()


def _transform_world_to_camera(
    world_to_camera: Tensor, scale: float, translation: Tensor
) -> Tensor:
    """Re-express ``3x4`` w2c poses in the scene ``X' = scale * X + translation``.

    The normalized camera frame is the original one scaled by ``scale``, matching the depth
    maps scaled by the same factor: ``[R | scale * t - R @ translation]``.
    """
    rotation = world_to_camera[..., :3, :3]
    shifted = scale * world_to_camera[..., :3, 3] - (
        rotation @ translation.to(rotation).unsqueeze(-1)
    ).squeeze(-1)
    return torch.cat([rotation, shifted.unsqueeze(-1)], dim=-1)


def _depth_centroid(
    depth: Tensor, valid: Tensor, intrinsics: Tensor, camera_to_world: Tensor
) -> Tensor:
    """Mean world position of the valid depth samples of one frame, ``(V,H,W)`` → ``(3,)``."""
    height, width = depth.shape[-2:]
    ys, xs = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=torch.float32),
        torch.arange(width, device=depth.device, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)  # integer pixel centres
    rays = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(intrinsics.float()), pixels)
    camera = rays * depth.float().unsqueeze(-1)
    world = torch.einsum("vij,vhwj->vhwi", camera_to_world[:, :3, :3], camera)
    world = world + camera_to_world[:, None, None, :3, 3]
    points = world[valid]
    if points.numel() == 0:
        raise ValueError(
            "scene_normalization='camera_radius' found no valid depth in the first frame; "
            "supply usable depth or set scene_normalization='none'/'manual'"
        )
    centroid = points.mean(dim=0)
    if not bool(torch.isfinite(centroid).all()):
        raise ValueError(
            "the first-frame depth centroid is not finite; mask out invalid depth or set "
            "scene_normalization='none'/'manual'"
        )
    return centroid


def _unpack_result(result: Any, frames: int, points: int) -> Tuple[Tensor, Tensor]:
    """``(traj_e, probability visibility)`` for one sequence, checked against ``(1,T,N,..)``."""
    if not isinstance(result, dict) or "traj_e" not in result:
        raise RuntimeError("MVTracker predictor must return a dict containing 'traj_e'")
    traj = result["traj_e"]
    if "vis_e_as_prob" in result:
        vis = result["vis_e_as_prob"]
    elif torch.is_floating_point(result.get("vis_e", torch.zeros(0, dtype=torch.bool))):
        vis = result["vis_e"]  # the bare core model returns sigmoid probabilities as vis_e
    else:
        raise RuntimeError(
            "MVTracker predictor returned only thresholded visibility; 'vis_e_as_prob' is "
            "required to preserve the native aggregate score"
        )
    if tuple(traj.shape) != (1, frames, points, 3):
        raise RuntimeError(
            f"MVTracker returned trajectories of shape {tuple(traj.shape)}, "
            f"expected {(1, frames, points, 3)}"
        )
    if tuple(vis.shape) != (1, frames, points):
        raise RuntimeError(
            f"MVTracker returned visibility of shape {tuple(vis.shape)}, "
            f"expected {(1, frames, points)}"
        )
    return traj, vis


def _validate_config(cfg: MVTrackerConfig) -> None:
    if cfg.image_size is not None and (len(cfg.image_size) != 2 or min(cfg.image_size) <= 0):
        raise ValueError("image_size must be a positive (height, width) pair")
    if cfg.sliding_window_len < 2 or cfg.stride < 1:
        raise ValueError("sliding_window_len must be >= 2 and stride >= 1")
    if cfg.n_iters < 1:
        raise ValueError("n_iters must be >= 1")
    if cfg.grid_size < 0 or cfg.n_grids_per_view < 1:
        raise ValueError("grid_size must be >= 0 and n_grids_per_view >= 1")
    if not 0.0 <= cfg.visibility_threshold <= 1.0:
        raise ValueError("visibility_threshold must lie in [0, 1]")
    if cfg.rgb_scale <= 0:
        raise ValueError("rgb_scale must be positive")
    if cfg.scene_normalization not in ("camera_radius", "manual", "none"):
        raise ValueError("scene_normalization must be 'camera_radius', 'manual' or 'none'")
    if cfg.scene_normalization == "manual" and not (
        cfg.scene_scale is not None and cfg.scene_scale > 0
    ):
        raise ValueError("scene_normalization='manual' needs a positive scene_scale")
    if cfg.scene_normalization == "camera_radius" and cfg.scene_target_radius <= 0:
        raise ValueError("scene_target_radius must be positive")


def _import_upstream(module: str, attribute: str):
    """Import an upstream submodule, rejecting a shadowed top-level ``mvtracker``."""
    existing = sys.modules.get("mvtracker")
    existing_file = getattr(existing, "__file__", None)
    if existing_file and Path(existing_file).resolve() == Path(__file__).resolve():
        raise ImportError(
            f"the top-level name `mvtracker` resolves to this adapter ({__file__}), so the "
            f"upstream package from {MVTRACKER_REPO_URL} cannot be imported. Import this "
            "module as `ontic_nn.trackers.mvtracker` and keep the package directory off "
            "sys.path."
        )
    try:
        imported = import_research_module(
            module, extra=EXTRA, repo_url=MVTRACKER_REPO_URL, what="MVTracker"
        )
    except ImportError as e:
        # The upstream predictor imports its debug mp4 visualiser eagerly, so a missing
        # `moviepy.editor` (gone in moviepy 2.x), `cv2` or `flow_vis` looks like a missing
        # `mvtracker`. Name what is actually absent.
        missing = getattr(e.__cause__, "name", None) or ""
        if missing and missing.split(".")[0] != "mvtracker":
            raise ImportError(
                f"the `mvtracker` package is installed but `{module}` needs `{missing}`. "
                "MVTracker imports pandas, pypng, torchvision and easydict for the model and "
                "cv2, flow_vis, matplotlib and moviepy.editor (moviepy<2 only) for the "
                f"predictor's debug visualiser; install ontic-nn[{EXTRA}]"
            ) from e
        raise
    if not hasattr(imported, attribute):
        raise ImportError(
            f"{module}.{attribute} is missing from the installed `mvtracker` package "
            f"({getattr(imported, '__file__', 'unknown location')}); this adapter targets "
            f"{MVTRACKER_REPO_URL} at revision {MVTRACKER_REVISION}"
        )
    return imported


def _load_checkpoint(model: nn.Module, cfg: MVTrackerConfig) -> Dict[str, Any]:
    """Resolve and load the released weights, following upstream ``hubconf._load_into``."""
    path = resolve_checkpoint(
        cfg.checkpoint_path,
        cfg.model_dir,
        filename=cfg.checkpoint_file,
        cache_dir=cfg.cache_dir,
        allow_download=cfg.allow_download,
        extra=EXTRA,
        what="MVTracker checkpoint",
    )
    if path is None:
        return {"path": None, "state": "randomly initialised"}
    raw = torch.load(path, map_location="cpu", weights_only=True)
    state = raw
    if isinstance(raw, dict):
        for key in ("state_dict", "model"):
            if isinstance(raw.get(key), dict):
                state = raw[key]
                break
    prefix = "model."
    state = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(
            f"MVTracker checkpoint {path} has keys the model does not define: "
            f"{sorted(unexpected)[:8]}; check the model knobs against the checkpoint"
        )
    if missing:
        raise RuntimeError(
            f"MVTracker checkpoint {path} is missing {len(missing)} model keys "
            f"(e.g. {sorted(missing)[:4]}); refusing partially random initialization"
        )
    return {
        "path": path,
        "repo_id": None if cfg.checkpoint_path else cfg.model_dir,
        "filename": cfg.checkpoint_file,
        "loaded_keys": len(state),
        "missing_keys": len(missing),
    }


__all__ = ["MVTracker", "MVTrackerConfig", "MVTRACKER_REPO_URL", "MVTRACKER_REVISION"]
