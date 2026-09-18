"""PointWorld-style CoTracker3 image tracks lifted with supplied depth and cameras.

PointWorld data revision 3872ec6 uses CoTrackerPredictor(offline=True, v2=False,
window_len=16) with scaled_online.pth, followed by nearest-pixel depth lifting.
Views run independently; identities are never matched or fused across cameras.
Ontic's normalized, edge-origin intrinsics require a half-pixel conversion to
CoTracker's integer-centered image coordinates, including across resolutions.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch
from torch import Tensor, nn

from ..wrappers.common import import_research_module, resolve_checkpoint
from .common import (
    GeometrySequence,
    PointQueries,
    TrackerBase,
    TrackerCapabilities,
    TrackerConfig,
    TrackerOutput,
    empty_tracker_output,
    make_tracker_output,
    project_queries,
    validate_tracker_inputs,
)
from .registry import register_tracker

COTRACKER3_REPO = "https://github.com/facebookresearch/co-tracker"
COTRACKER3_REVISION = "82e02e8029753ad4ef13cf06be7f4fc5facdda4d"
POINTWORLD_REVISION = "3872ec6ee73146aa671192ef79b5dfbedc0246e3"


@register_tracker("cotracker3")
@dataclass(kw_only=True)
class CoTracker3Config(TrackerConfig):
    """PointWorld's offline CoTracker3 setup with explicit surface queries.

    ``repo_path`` accepts a CoTracker checkout or PointWorld's data checkout with
    its co-tracker submodule initialized (or use ``ONTIC_COTRACKER3_REPO``).
    Without a path, use the installed ``cotracker`` package. All weights load
    through an explicit local checkpoint, respecting ``allow_download``.

    The predictor uses its native 384x512 grid, independently of ``long_side``.
    Supplied depth is sampled at its original resolution. Upstream inference is
    detached; training via ``freeze_tracker=False`` is unsupported.
    """

    repo_path: str | None = None
    model_dir: str = "facebook/cotracker3"
    checkpoint_file: str = "scaled_online.pth"
    window_len: int = 16
    bidirectional: bool = True
    query_chunk_size: int = 8192
    amp: bool = True

    CAPABILITIES: ClassVar[TrackerCapabilities] = TrackerCapabilities(
        multiview=True, visibility_scope="query_view", execution_mode="offline_per_view"
    )

    def build(self) -> CoTracker3:
        return CoTracker3(self)


class CoTracker3(TrackerBase):
    CAPABILITIES = CoTracker3Config.CAPABILITIES

    def __init__(self, cfg: CoTracker3Config, model: nn.Module | None = None):
        super().__init__()
        if not cfg.freeze_tracker:
            raise ValueError(
                "cotracker3: upstream predictor is inference-only; freeze_tracker must be True"
            )
        if cfg.window_len < 2 or cfg.query_chunk_size < 1:
            raise ValueError("cotracker3: window_len must be >=2 and query_chunk_size positive")
        self._checkpoint = None
        if model is None:
            model, self._checkpoint = load_cotracker3(cfg)
        self._configure_model(model, cfg)

    @torch.no_grad()
    def forward(
        self, images: Tensor, queries: PointQueries, *, geometry: GeometrySequence
    ) -> TrackerOutput:
        b, t, v, n = validate_tracker_inputs(images, queries, geometry)
        if n == 0:
            return empty_tracker_output(images, queries, geometry, visibility_scope="query_view")
        h, w = images.shape[-2:]
        if min(h, w) < 2:
            raise ValueError("cotracker3: images must be at least 2x2")
        uv = project_queries(queries, geometry)
        pixels = uv * uv.new_tensor([w, h]) - 0.5
        if ((pixels < -1e-4) | (pixels > pixels.new_tensor([w - 1, h - 1]) + 1e-4)).any():
            raise ValueError("cotracker3: queries must lie between image pixel centers")
        pixels = pixels.clamp(min=0).minimum(pixels.new_tensor([w - 1, h - 1]))
        views = queries.source_view
        if views is None:
            views = torch.zeros_like(queries.time)
        tracks = images.new_full((b, t, n, 3), torch.nan, dtype=torch.float32)
        valid = torch.zeros((b, t, n), device=images.device, dtype=torch.bool)
        visibility = images.new_zeros((b, t, n), dtype=torch.float32)
        for bi in range(b):
            for vi in range(v):
                indices = torch.where(views[bi] == vi)[0]
                if not len(indices):
                    continue
                video = images[bi : bi + 1, :, vi].float() * 255.0
                for chunk in indices.split(self.cfg.query_chunk_size):
                    q = torch.cat(
                        [queries.time[bi, chunk, None].float(), pixels[bi, chunk]], dim=-1
                    )[None]
                    with torch.autocast(
                        "cuda", dtype=torch.float16, enabled=images.is_cuda and self.cfg.amp
                    ):
                        xy, vis = self.model(
                            video, queries=q, backward_tracking=self.cfg.bidirectional
                        )
                    if xy.shape != (1, t, len(chunk), 2) or vis.shape != (1, t, len(chunk)):
                        raise ValueError("cotracker3: unexpected upstream track/visibility shapes")
                    if not torch.isfinite(vis).all() or ((vis < 0) | (vis > 1)).any():
                        raise ValueError("cotracker3: visibility must be finite in [0,1]")
                    world, supported = _lift_tracks(xy[0].float(), geometry, bi, vi, (h, w))
                    # Occluded image positions sample the occluder's surface, so they
                    # cannot provide a valid 3D location for the tracked point.
                    supported &= vis[0] > 0.5
                    if not self.cfg.bidirectional:
                        supported &= (
                            torch.arange(t, device=images.device)[:, None]
                            >= queries.time[bi, chunk]
                        )
                    tracks[bi, :, chunk] = world.masked_fill(~supported[..., None], torch.nan)
                    valid[bi, :, chunk] = supported
                    visibility[bi, :, chunk] = vis[0].float()
        return make_tracker_output(
            queries,
            geometry,
            tracks,
            visibility,
            valid=valid,
            metadata={
                "tracker": "cotracker3",
                "upstream_repo": COTRACKER3_REPO,
                "upstream_api_revision": COTRACKER3_REVISION,
                "reference_pipeline": "https://github.com/NVlabs/PointWorld/tree/data",
                "reference_revision": POINTWORLD_REVISION,
                "checkpoint": self._checkpoint,
                "predictor": {"offline": True, "v2": False, "window_len": self.cfg.window_len},
                "backward_tracking": self.cfg.bidirectional,
                "query_chunk_size": self.cfg.query_chunk_size,
                "inference_resolution": list(getattr(self.model, "interp_shape", (384, 512))),
                "view_processing": "independent source cameras; no cross-view identity fusion",
                "lifting": "nearest depth pixel with normalized pixel-center camera geometry",
                "visibility": "binary upstream decision (threshold 0.9); not calibrated confidence",
                "valid_rule": "visible, in-image, finite positive unmasked depth",
            },
        )


def _lift_tracks(xy, geometry, batch, view, image_size):
    """Nearest surface sample, never interpolate holes or accept out-of-image tracks."""
    h, w = image_size
    depth = geometry.depth[batch, :, view].float()
    t, hd, wd = depth.shape
    finite = torch.isfinite(xy).all(-1)
    inside = finite & ((xy >= 0) & (xy < xy.new_tensor([w, h]))).all(-1)
    uv = (torch.nan_to_num(xy, nan=0, posinf=0, neginf=0) + 0.5) / xy.new_tensor([w, h])
    depth_xy = (uv * uv.new_tensor([wd, hd]) - 0.5).round().long()
    x, y = depth_xy[..., 0].clamp(0, wd - 1), depth_xy[..., 1].clamp(0, hd - 1)
    time = torch.arange(t, device=xy.device)[:, None]
    z = depth[time, y, x]
    valid = inside & geometry.valid_depth[batch, :, view][time, y, x]
    # Lift the sampled pixel itself, as PointWorld does, while preserving Ontic's
    # half-pixel convention and supporting per-frame poses and skewed intrinsics.
    pixel = torch.stack([(x + 0.5) / wd, (y + 0.5) / hd, torch.ones_like(z)], dim=-1)
    k = geometry.intrinsics[batch, :, view].float()
    camera = torch.einsum("tij,tnj->tni", torch.linalg.inv(k), pixel) * z[..., None]
    c2w = geometry.extrinsics[batch, :, view].float()
    world = torch.einsum("tij,tnj->tni", c2w[:, :3, :3], camera) + c2w[:, None, :3, 3]
    return world, valid & torch.isfinite(world).all(-1)


def load_cotracker3(cfg: CoTracker3Config):
    raw = cfg.repo_path or os.environ.get("ONTIC_COTRACKER3_REPO")
    repo = None
    if raw:
        repo = Path(raw).expanduser().resolve()
        if not (repo / "cotracker/predictor.py").is_file():
            repo = repo / "third_party/co-tracker"
        if not (repo / "cotracker/predictor.py").is_file():
            raise ImportError(
                "cotracker3: repo_path must be a CoTracker checkout or PointWorld data checkout "
                f"with third_party/co-tracker initialized; clone {COTRACKER3_REPO} "
                f"at {COTRACKER3_REVISION[:7]}"
            )
        loaded = sys.modules.get("cotracker")
        if loaded is not None and not Path(
            getattr(loaded, "__file__", "") or ""
        ).resolve().is_relative_to(repo):
            raise ImportError(
                "cotracker3: another checkout is loaded; restart to change repositories"
            )
    checkpoint = resolve_checkpoint(
        cfg.checkpoint_path,
        cfg.model_dir or None,
        filename=cfg.checkpoint_file,
        cache_dir=cfg.cache_dir,
        allow_download=cfg.allow_download,
        extra="cotracker3",
        what="CoTracker3 checkpoint",
    )
    if checkpoint is None:
        raise ValueError("cotracker3: a pretrained checkpoint is required")
    if repo is not None:
        sys.path.insert(0, str(repo))
    try:
        module = import_research_module(
            "cotracker.predictor", extra="cotracker3", repo_url=COTRACKER3_REPO, what="CoTracker3"
        )
        model = module.CoTrackerPredictor(
            checkpoint=checkpoint, offline=True, v2=False, window_len=cfg.window_len
        )
    finally:
        if repo is not None:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(repo))
    return model, checkpoint
