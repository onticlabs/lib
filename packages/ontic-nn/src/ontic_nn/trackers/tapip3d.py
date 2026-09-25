"""TAPIP3D tracker wrapper: monocular RGB-D tracking in a persistent world frame.

Drives the released ``PointTracker3D`` of https://github.com/zbw001/TAPIP3D the way
``inference.py`` / ``utils/inference_utils.py`` do, but takes depth and cameras from the
supplied :class:`GeometrySequence` instead of estimating them: MegaSAM, its depth models and
``third_party/megasam`` are never imported. Inspected upstream revision
``4cb7e69a1687f67d56ec3e506768f51f2c581b46`` (2025-12-29).

The upstream conventions below were read off the code, not the README (which only documents
array shapes):

* ``extrinsics`` are **world-to-camera** ``4x4``: ``utils.common_utils.batch_unproject`` lifts
  ``K^-1 [u, v, 1] * depth`` with ``inv(extrinsic)``, and ``PointTracker3D._project`` maps a
  world point to the camera with ``extrinsic`` itself. So this wrapper inverts the public
  camera-to-world matrices.
* ``intrinsics`` are pixel intrinsics over **integer-centred** indices ``u in [0, W-1]``: the
  same ``batch_unproject`` builds its grid with ``arange(w)``. Hence
  ``pixel_intrinsics(..., integer_centers=True)``. Upstream rescales intrinsics for a resize
  with ``(W' - 1) / (W - 1)`` while resizing images with half-pixel-centred interpolation; this
  wrapper instead derives the pixel matrix from the normalised intrinsics at the target size,
  which is consistent with the ``align_corners=False`` resampling in ``resize_tracker_inputs``.
  The two differ by well under a pixel.
* ``query_point`` is ``(B, N, 4)`` ordered ``(t, x, y, z)`` with ``xyz`` in the world frame and
  ``t`` an index into the clip; ``depth_obs`` uses ``0`` for invalid depth.
* ``eval_mode="raw"`` (what ``load_model`` selects) keeps the model reasoning in the supplied
  world frame. ``"local"`` rewrites the queries into the camera frame at the query time and
  replaces every extrinsic with the identity, i.e. it cancels camera motion; it is available
  but is not the default here because it defeats world-space tracking with a moving camera.

Native inference is single-view (``V=1``) and single-clip (``B=1``; the upstream window loop
raises ``NotImplementedError`` for larger batches), so ``V > 1`` is rejected and ``B`` is
looped. Upstream detaches its ``Prediction``, so no gradient reaches the tracker's inputs
whatever ``freeze_tracker`` says.

Queries are consumed as world XYZ, so ``source_view`` / ``source_uv`` are provenance only and
are not read here. Depth arrives as ``GeometrySequence`` depth masked by ``valid_depth``;
the flying-pixel edge filter ``inference.py`` runs over MegaSAM depth (``_filter_one_depth``,
scipy plus MoGe utilities) is *not* applied — mask such pixels in the geometry instead.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import importlib
import math
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from ..wrappers.common import amp_dtype, resolve_checkpoint
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
from .sources import installed_repo

TAPIP3D_REPO_URL = "https://github.com/zbw001/TAPIP3D"
TAPIP3D_REVISION = "4cb7e69a1687f67d56ec3e506768f51f2c581b46"
TAPIP3D_HF_REPO = "zbww/tapip3d"
TAPIP3D_CHECKPOINT = "tapip3d_final.pth"
REPO_PATH_ENV = "ONTIC_TAPIP3D_REPO"
EXTRA = "tapip3d"
VISIBILITY_SCOPE = "query_view"

#: Generic top-level package names the upstream checkout claims once it is on ``sys.path``.
UPSTREAM_TOP_LEVEL: Tuple[str, ...] = ("models", "utils", "datasets", "training", "third_party")

_INSTALL_HINT = (
    f"clone {TAPIP3D_REPO_URL} (revision {TAPIP3D_REVISION[:7]}), build its CUDA extension "
    "(`cd third_party/pointops2 && python setup.py install`) and point "
    "`TAPIP3DConfig.repo_path` (or $" + REPO_PATH_ENV + ") at the checkout"
)


# ---------------------------------------------------------------------------
# Config / wrapper
# ---------------------------------------------------------------------------
@register_tracker("tapip3d")
@dataclass(kw_only=True)
class TAPIP3DConfig(TrackerConfig):
    """TAPIP3D with externally supplied geometry.

    ``repo_path`` is the upstream checkout (falls back to ``$ONTIC_TAPIP3D_REPO``, then the
    workspace installer's source index). It is only read by :meth:`build`, never on import,
    and it is put on ``sys.path`` for the duration of
    that import only. ``checkpoint_path`` wins over ``model_dir`` / ``checkpoint_file``, the
    released ``zbww/tapip3d`` hub weights.

    Inference resolution, in precedence order: ``inference_resolution`` (explicit ``(H, W)``),
    else ``resolution_factor`` applied to the checkpoint's training resolution the way
    ``inference.py`` does (``int(side * sqrt(factor))``, the default reproducing its
    ``--resolution_factor 2``), else the inherited ``long_side`` scaling that training
    resolution. ``num_iters`` / ``support_grid_size`` mirror the upstream demo arguments;
    ``bidirectional`` adds its reverse-time pass so that queries at ``t > 0`` also get frames
    before the query time (without it those frames are returned with ``valid=False``).
    """

    repo_path: Optional[str] = None
    model_dir: str = TAPIP3D_HF_REPO
    checkpoint_file: str = TAPIP3D_CHECKPOINT
    inference_resolution: Optional[Tuple[int, int]] = None
    resolution_factor: Optional[float] = 2.0
    num_iters: int = 6
    support_grid_size: int = 16
    eval_mode: str = "raw"
    bidirectional: bool = True
    use_depth_roi: bool = True
    amp: bool = True

    CAPABILITIES: ClassVar[TrackerCapabilities] = TrackerCapabilities(
        multiview=False,
        dense=False,
        arbitrary_query_times=True,
        execution_mode="offline",
        visibility_scope=VISIBILITY_SCOPE,
    )

    def build(self) -> TAPIP3D:
        return TAPIP3D(self)


class TAPIP3D(TrackerBase):
    """``PointTracker3D`` driven with Ontic geometry; ``model`` bypasses weight loading."""

    CAPABILITIES: ClassVar[TrackerCapabilities] = TAPIP3DConfig.CAPABILITIES

    def __init__(self, cfg: TAPIP3DConfig, model: Optional[nn.Module] = None) -> None:
        super().__init__()
        if cfg.eval_mode not in ("raw", "local"):
            raise ValueError(f"tapip3d: eval_mode must be 'raw' or 'local', got {cfg.eval_mode!r}")
        self._checkpoint: Optional[str] = None
        if model is None:
            model, self._checkpoint = load_tapip3d(cfg)
        self._native_image_size = _native_image_size(model)
        seq_len = getattr(model, "seq_len", None)
        self._seq_len: Optional[int] = int(seq_len) if seq_len else None
        if hasattr(model, "set_eval_mode"):
            model.set_eval_mode(cfg.eval_mode)
        if not cfg.freeze_tracker:
            warnings.warn(
                "tapip3d: freeze_tracker=False still yields no gradients — upstream returns a "
                "detached Prediction (models/point_tracker_3d.py), so its trajectories are "
                "always cut off from the autograd graph",
                stacklevel=2,
            )
        self._configure_model(model, cfg)

    # -- resolution -------------------------------------------------------------------
    @property
    def native_image_size(self) -> Tuple[int, int]:
        """``(H, W)`` the checkpoint was trained at, captured before any ``set_image_size``."""
        return self._native_image_size

    def inference_size(self) -> Tuple[int, int]:
        """``(H, W)`` this wrapper resizes RGB and depth to; see :class:`TAPIP3DConfig`."""
        cfg: TAPIP3DConfig = self.cfg
        native = self._native_image_size
        if cfg.inference_resolution is not None:
            h, w = (int(cfg.inference_resolution[0]), int(cfg.inference_resolution[1]))
        elif cfg.resolution_factor is not None:
            if cfg.resolution_factor <= 0:
                raise ValueError(
                    f"tapip3d: resolution_factor must be > 0, got {cfg.resolution_factor}"
                )
            scale = math.sqrt(float(cfg.resolution_factor))
            h, w = int(native[0] * scale), int(native[1] * scale)  # upstream truncates
        else:
            scale = float(cfg.long_side) / max(native)
            h, w = round(native[0] * scale), round(native[1] * scale)
        if h < 1 or w < 1:
            raise ValueError(f"tapip3d: degenerate inference resolution {(h, w)}")
        return h, w

    # -- forward ----------------------------------------------------------------------
    def forward(
        self,
        images: Tensor,
        queries: PointQueries,
        *,
        geometry: GeometrySequence,
    ) -> TrackerOutput:
        if images.ndim == 6 and images.shape[2] != 1:
            raise ValueError(
                f"tapip3d is a monocular tracker (native V=1) but got V={images.shape[2]}; its "
                "correspondence reasoning is single-view, so run a multi-view tracker or call it "
                "per view with that view's geometry and queries"
            )
        b, t, _v, n = validate_tracker_inputs(images, queries, geometry, multiview=False)
        cfg: TAPIP3DConfig = self.cfg
        if n == 0:
            return empty_tracker_output(
                images, queries, geometry, visibility_scope=VISIBILITY_SCOPE
            )

        size = self.inference_size()
        rgb, depth, depth_valid = resize_tracker_inputs(images, geometry, size=size)
        rgb = rgb[:, :, 0].float()  # (B, T, 3, H, W)
        depth = torch.where(depth_valid, depth, torch.zeros_like(depth))[:, :, 0].float()
        k_pix = pixel_intrinsics(geometry.intrinsics, size[0], size[1], integer_centers=True)
        k_pix = k_pix[:, :, 0].float()  # (B, T, 3, 3)
        c2w = geometry.extrinsics[:, :, 0].float()  # (B, T, 4, 4)
        w2c = _invert_poses(c2w)
        query_point = torch.cat(
            [queries.time.to(torch.float32)[..., None], queries.xyz_world.float()], dim=-1
        )  # (B, N, 4)

        coords: List[Tensor] = []
        logits: List[Tensor] = []
        supported: List[Tensor] = []
        backward_used: List[bool] = []
        with self._grad_context():
            for i in range(b):
                item = self._track_clip(
                    rgb[i : i + 1],
                    depth[i : i + 1],
                    k_pix[i : i + 1],
                    w2c[i : i + 1],
                    c2w[i : i + 1],
                    query_point[i : i + 1],
                    queries.time[i : i + 1],
                )
                coords.append(item[0])
                logits.append(item[1])
                supported.append(item[2])
                backward_used.append(item[3])

        tracks_world = torch.cat(coords, dim=0)  # (B, T, N, 3)
        visibility = torch.sigmoid(torch.cat(logits, dim=0))  # (B, T, N)
        valid = torch.cat(supported, dim=0) & torch.isfinite(tracks_world).all(dim=-1)

        pad = max(0, (self._seq_len or 0) - t)
        metadata: Dict[str, Any] = {
            "tracker": "tapip3d",
            "upstream_repo": TAPIP3D_REPO_URL,
            "upstream_api_revision": TAPIP3D_REVISION,
            "checkpoint": self._checkpoint,
            "eval_mode": cfg.eval_mode,
            "num_iters": cfg.num_iters,
            "support_grid_size": cfg.support_grid_size,
            "native_image_size": self._native_image_size,
            "inference_resolution": size,
            "padded_frames": pad,
            "backward_pass": tuple(backward_used),
            "extrinsics_convention": "world_to_camera 4x4 (upstream batch_unproject / _project)",
            "pixel_convention": "integer-centred pixel indices",
            "visibility": "sigmoid of the native logits; upstream's demo thresholds at 0.9",
            "gradients": "unavailable: upstream returns a detached Prediction",
        }
        output = make_tracker_output(
            queries,
            geometry,
            tracks_world,
            visibility,
            visibility_scope=VISIBILITY_SCOPE,
            valid=valid,
            metadata=metadata,
        )
        # V == 1, so the native per-frame score *is* the per-view score; nothing is derived here.
        return dataclasses.replace(output, visibility_per_view=output.visibility[:, :, None])

    # -- one clip ---------------------------------------------------------------------
    def _track_clip(
        self,
        rgb: Tensor,
        depth: Tensor,
        k_pix: Tensor,
        w2c: Tensor,
        c2w: Tensor,
        query_point: Tensor,
        query_time: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, bool]:
        """``(coords, visibility logits, supported, backward_used)`` for one ``B=1`` clip."""
        cfg: TAPIP3DConfig = self.cfg
        t = rgb.shape[1]
        depth_roi = _depth_roi(depth) if cfg.use_depth_roi else None
        coords, logits = self._run_native(rgb, depth, k_pix, w2c, c2w, query_point, depth_roi)

        steps = torch.arange(t, device=query_time.device, dtype=query_time.dtype)
        before = steps[None, :, None] < query_time[:, None, :]  # (1, T, N)
        native_bidirectional = bool(getattr(self.model, "bidirectional", False))
        backward = bool(cfg.bidirectional and not native_bidirectional and bool(before.any()))
        if backward:
            # Upstream runs the reverse-time clip and keeps its estimate strictly before the
            # query time (utils/inference_utils.inference).
            back_query = torch.cat([(t - 1) - query_point[..., :1], query_point[..., 1:]], dim=-1)
            back_coords, back_logits = self._run_native(
                rgb.flip(dims=(1,)),
                depth.flip(dims=(1,)),
                k_pix.flip(dims=(1,)),
                w2c.flip(dims=(1,)),
                c2w.flip(dims=(1,)),
                back_query,
                depth_roi,
            )
            coords = torch.where(before[..., None], back_coords.flip(dims=(1,)), coords)
            logits = torch.where(before, back_logits.flip(dims=(1,)), logits)

        if backward or native_bidirectional:
            supported = torch.ones_like(before)
        else:
            # Frames before the query time are only the window initialisation copied forward.
            supported = ~before
        return coords, logits, supported, backward

    def _run_native(
        self,
        rgb: Tensor,
        depth: Tensor,
        k_pix: Tensor,
        w2c: Tensor,
        c2w: Tensor,
        query_point: Tensor,
        depth_roi: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """One native pass; returns ``(coords (1, T, N, 3), logits (1, T, N))``."""
        cfg: TAPIP3DConfig = self.cfg
        t = rgb.shape[1]
        n = query_point.shape[1]
        if self._seq_len and t < self._seq_len:
            # Shorter than one window: upstream's window loop would not run at all and would
            # return the untouched initialisation, so extend with the last frame (what its own
            # partial-window padding does) and drop the extra frames afterwards.
            pad = self._seq_len - t
            rgb = torch.cat([rgb, rgb[:, -1:].expand(-1, pad, -1, -1, -1)], dim=1)
            depth = torch.cat([depth, depth[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            k_pix = torch.cat([k_pix, k_pix[:, -1:].expand(-1, pad, -1, -1)], dim=1)
            w2c = torch.cat([w2c, w2c[:, -1:].expand(-1, pad, -1, -1)], dim=1)

        support = self._support_queries(depth, k_pix, c2w)
        if support is not None:
            query_point = torch.cat([query_point, support], dim=1)

        height, width = rgb.shape[-2:]
        self.model.set_image_size((height, width))
        kwargs: Dict[str, Any] = {
            "rgb_obs": rgb,
            "depth_obs": depth,
            "num_iters": cfg.num_iters,
            "query_point": query_point,
            "intrinsics": k_pix,
            "extrinsics": w2c,
            "mode": "inference",
        }
        if depth_roi is not None:
            kwargs["depth_roi"] = depth_roi
        with self._autocast(rgb.device):
            out = self.model(**kwargs)
        preds = out[0] if isinstance(out, tuple) else out  # upstream: (Prediction, [TrainData])
        return preds.coords[:, :t, :n].float(), preds.visibs[:, :t, :n].float()

    def _support_queries(self, depth: Tensor, k_pix: Tensor, c2w: Tensor) -> Optional[Tensor]:
        """Upstream's ``get_grid_queries``: a first-frame pixel grid lifted to world ``(1, M, 4)``.

        These extra queries only condition the correlation context; they are sliced off again.
        """
        size = int(self.cfg.support_grid_size)
        if size <= 0:
            return None
        height, width = depth.shape[-2:]
        xy = _points_on_a_grid(size, height, width, device=depth.device, dtype=depth.dtype)
        ji = xy.round().long()
        cols = ji[..., 0].clamp(0, width - 1)
        rows = ji[..., 1].clamp(0, height - 1)
        d = depth[0, 0][rows[0], cols[0]]  # (M,)
        keep = d > 0
        if not bool(keep.any()):
            return None
        xy, d = xy[:, keep], d[keep]
        rays = torch.cat([xy, torch.ones_like(xy[..., :1])], dim=-1)  # (1, M, 3)
        inv_k = torch.linalg.inv(k_pix[0, 0].double()).to(rays.dtype)
        local = torch.einsum("ij,bnj->bni", inv_k, rays) * d[None, :, None]
        local = torch.cat([local, torch.ones_like(local[..., :1])], dim=-1)
        world = torch.einsum("ij,bnj->bni", c2w[0, 0], local)[..., :3]
        return torch.cat([torch.zeros_like(world[..., :1]), world], dim=-1)

    def _autocast(self, device: torch.device):
        if self.cfg.amp and device.type == "cuda":
            return torch.autocast("cuda", dtype=amp_dtype())
        return contextlib.nullcontext()


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def _invert_poses(c2w: Tensor) -> Tensor:
    """Camera-to-world ``(..., 4, 4)`` → the world-to-camera matrices upstream expects."""
    try:
        return torch.linalg.inv(c2w.double()).to(c2w.dtype)
    except RuntimeError as e:
        raise ValueError(f"tapip3d: geometry.extrinsics are not invertible ({e})") from e


def _points_on_a_grid(size: int, height: int, width: int, *, device, dtype) -> Tensor:
    """CoTracker's ``get_points_on_a_grid``: ``(1, size * size, 2)`` pixel ``(x, y)``.

    Both margins are ``width / 64``, as upstream (``third_party/cotracker/model_utils.py``).
    """
    if size == 1:
        return torch.tensor([width / 2, height / 2], device=device, dtype=dtype)[None, None]
    margin = width / 64
    ys = torch.linspace(margin, height - margin, size, device=device, dtype=dtype)
    xs = torch.linspace(margin, width - margin, size, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).reshape(1, -1, 2)


def _depth_roi(depth: Tensor) -> Tensor:
    """Upstream's inference-time depth range ``[1e-7, q75 + 1.5 * IQR]`` over positive depths."""
    values = depth[depth > 0].reshape(-1)
    if values.numel() == 0:
        raise ValueError(
            "tapip3d: geometry has no valid positive depth in this clip; the model normalises "
            "the scene by its depth statistics and cannot run"
        )
    count = values.numel()
    q25 = torch.kthvalue(values, min(max(int(0.25 * count), 1), count)).values
    q75 = torch.kthvalue(values, min(max(int(0.75 * count), 1), count)).values
    high = q75 + 1.5 * (q75 - q25)
    return torch.stack([torch.full_like(high, 1e-7), high]).to(torch.float32)


def _native_image_size(model: nn.Module) -> Tuple[int, int]:
    size = getattr(model, "image_size", None)
    try:
        return int(size[0]), int(size[1])  # type: ignore[index]
    except (TypeError, IndexError, ValueError) as e:
        raise TypeError(
            f"tapip3d: the model's `image_size` is {size!r}, not an (H, W) pair; expected "
            "TAPIP3D's PointTracker3D, which carries the resolution it was trained at"
        ) from e


# ---------------------------------------------------------------------------
# Upstream import and checkpoint loading
# ---------------------------------------------------------------------------
def load_tapip3d(cfg: TAPIP3DConfig) -> Tuple[nn.Module, str]:
    """Build ``PointTracker3D`` from the checkpoint's own config and load its weights.

    Returns the model and the checkpoint path. The upstream checkout is on ``sys.path`` only
    while its packages are imported; the imported modules stay in ``sys.modules`` afterwards
    (that is what keeps the model working) and nothing already imported is replaced.
    """
    repo = _resolve_repo_path(cfg)
    checkpoint = resolve_checkpoint(
        cfg.checkpoint_path,
        cfg.model_dir or None,
        filename=cfg.checkpoint_file,
        cache_dir=cfg.cache_dir,
        allow_download=cfg.allow_download,
        extra=EXTRA,
        what="TAPIP3D checkpoint",
    )
    if checkpoint is None:
        raise ValueError(
            "tapip3d: no checkpoint — set checkpoint_path, or keep model_dir "
            f"{TAPIP3D_HF_REPO!r} to fetch {TAPIP3D_CHECKPOINT!r} from the hub"
        )

    with _upstream_on_path(repo):
        models = _import_upstream(repo)
        # weights_only=False: the checkpoint stores its OmegaConf training config next to the
        # weights, exactly as upstream's models.from_pretrained loads it. Only point this at a
        # checkpoint you trust.
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = _build_from_checkpoint(models, state, checkpoint)
    return model, checkpoint


def _resolve_repo_path(cfg: TAPIP3DConfig) -> Path:
    raw = cfg.repo_path or os.environ.get(REPO_PATH_ENV) or installed_repo("tapip3d")
    if not raw:
        raise ImportError(
            f"TAPIP3D is not an installable package: {_INSTALL_HINT}. "
            "Install ontic-nn[tapip3d] for the pure-Python dependencies "
            "(hydra-core, omegaconf, einops, python-box, rich)."
        )
    repo = Path(raw).expanduser().resolve()
    if not (repo / "models" / "point_tracker_3d.py").is_file():
        raise ImportError(
            f"tapip3d: {repo} does not look like a TAPIP3D checkout "
            "(models/point_tracker_3d.py is missing); " + _INSTALL_HINT
        )
    return repo


@contextlib.contextmanager
def _upstream_on_path(repo: Path) -> Iterator[None]:
    """Prepend ``repo`` to ``sys.path`` for the block, refusing to shadow foreign modules."""
    _check_module_collisions(repo)
    known = set(sys.modules)
    entry = str(repo)
    sys.path.insert(0, entry)
    try:
        yield
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(entry)
        claimed = [n for n in UPSTREAM_TOP_LEVEL if n in sys.modules and n not in known]
        if claimed:
            warnings.warn(
                f"tapip3d: the checkout at {repo} now owns the top-level module name(s) "
                f"{', '.join(claimed)} for the rest of this process — import anything else that "
                "uses those names (for instance HuggingFace `datasets`) in a separate process",
                stacklevel=3,
            )


def _check_module_collisions(repo: Path) -> None:
    """Raise when a generic upstream top-level name is already taken by something else."""
    clashes: Dict[str, str] = {}
    for name in UPSTREAM_TOP_LEVEL:
        if name not in sys.modules:
            continue
        module = sys.modules[name]
        if module is None:
            clashes[name] = "blocked (sys.modules entry is None)"
            continue
        paths = _module_paths(module)
        if not paths:
            clashes[name] = "a module without a file"
        elif all(not p.is_relative_to(repo) for p in paths):
            clashes[name] = str(paths[0])
    if clashes:
        listing = "; ".join(f"{name} <- {owner}" for name, owner in sorted(clashes.items()))
        raise ImportError(
            f"tapip3d: its checkout at {repo} imports the generic top-level package name(s) "
            f"{', '.join(sorted(clashes))}, which this process already uses: {listing}. "
            "Nothing is removed or replaced here — load TAPIP3D in a process that does not "
            "import those names, or import it before the conflicting package."
        )


def _module_paths(module: Any) -> List[Path]:
    paths: List[Path] = []
    file = getattr(module, "__file__", None)
    if file:
        paths.append(Path(file).resolve())
    for entry in getattr(module, "__path__", None) or ():
        if isinstance(entry, (str, os.PathLike)):
            paths.append(Path(entry).resolve())
    return paths


def _import_upstream(repo: Path):
    try:
        models = importlib.import_module("models")
    except ImportError as e:
        raise ImportError(
            f"tapip3d: importing `models` from {repo} failed ({type(e).__name__}: {e}). It needs "
            "the upstream dependencies — hydra-core, omegaconf, einops, python-box, rich, "
            "torchvision — and the compiled `pointops2_cuda` extension imported at the top of "
            "models/corr_features/knn_feature_4d_optimized.py "
            "(`cd third_party/pointops2 && python setup.py install`, CUDA toolchain required)."
        ) from e
    if not any(p.is_relative_to(repo) for p in _module_paths(models)):
        raise ImportError(
            f"tapip3d: `models` resolved to {_module_paths(models) or '<unknown>'} instead of the "
            f"checkout at {repo}; that name is taken in this process"
        )
    return models


def _build_from_checkpoint(models: Any, state: Any, checkpoint: str) -> nn.Module:
    if not isinstance(state, dict) or "cfg" not in state or "weight" not in state:
        keys = sorted(state)[:8] if isinstance(state, dict) else type(state).__name__
        raise RuntimeError(
            f"tapip3d: {checkpoint} is not a TAPIP3D checkpoint — expected a dict with 'cfg' and "
            f"'weight' (as written by upstream training), found {keys}. Foreign checkpoints that "
            "upstream's models.smart_load handles (SpaTracker) are not supported here."
        )
    cfg = state["cfg"]
    try:
        model_cfg = copy.deepcopy(cfg["model"])
        resolution = cfg["train_dataset"]["resolution"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(
            f"tapip3d: {checkpoint} has no cfg['model'] / cfg['train_dataset']['resolution'] "
            f"({type(e).__name__}: {e}); it was not written by this upstream revision"
        ) from e
    # The full checkpoint already includes the encoder weights. Upstream's training
    # config otherwise downloads CoTracker via torch.hub during construction, even
    # when allow_download=False, then immediately overwrites those weights below.
    encoder = model_cfg.get("encoder", {})
    if encoder.get("name") == "cotracker_cnn":
        encoder["pretrained"] = False
    model = models.from_config(model_cfg, image_size=tuple(int(x) for x in resolution))
    _load_weights(model, state["weight"], checkpoint)
    model.eval()
    return model


def _load_weights(model: nn.Module, weights: Any, checkpoint: str) -> None:
    """Strict load with a diagnostic instead of PyTorch's bare key dump."""
    if not isinstance(weights, dict):
        raise RuntimeError(
            f"tapip3d: the 'weight' entry of {checkpoint} is {type(weights).__name__}, "
            "not a state dict"
        )
    target = model.state_dict()
    missing = sorted(set(target) - set(weights))
    unexpected = sorted(set(weights) - set(target))
    mismatched = sorted(
        f"{k}: checkpoint {tuple(weights[k].shape)} vs model {tuple(target[k].shape)}"
        for k in set(target) & set(weights)
        if getattr(weights[k], "shape", None) is not None
        and tuple(weights[k].shape) != tuple(target[k].shape)
    )
    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"tapip3d: {checkpoint} does not match the model built from its own config — "
            f"{len(missing)} missing, {len(unexpected)} unexpected, {len(mismatched)} mismatched "
            f"shapes.{_sample('missing', missing)}{_sample('unexpected', unexpected)}"
            f"{_sample('mismatched', mismatched)} The checkout must be at the revision the "
            f"checkpoint was written with ({TAPIP3D_REVISION[:7]} was inspected here)."
        )
    model.load_state_dict(weights, strict=True)


def _sample(label: str, names: Sequence[str], limit: int = 5) -> str:
    if not names:
        return ""
    shown = ", ".join(names[:limit])
    more = f", ... (+{len(names) - limit})" if len(names) > limit else ""
    return f" {label}: {shown}{more}."
