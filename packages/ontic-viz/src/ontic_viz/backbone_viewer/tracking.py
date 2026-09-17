"""Clip preparation, query selection, inference and export, independent of viser.

Tracking always uses dataset calibration as its common world frame. Geometry
calibration happens before tracking and is recorded separately from trajectories.
"""

from __future__ import annotations

import dataclasses
import io
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from ontic_lib.camera import project_world_points
from ontic_lib.depth.alignment import fit_depth_scale
from ontic_lib.pointops import align_camera_poses_sim3, furthest_point_indices
from ontic_nn.trackers import TRACKERS, GeometrySequence, PointQueries, TrackerOutput

from .data_source import Frame
from .render import PointFilters, build_point_cloud, confidence_threshold
from .runner import BackboneResult, BackboneRunner

SENSOR = "Sensor / GT depth"
SENSOR_SCALE = "Backbone depth · sensor scale"
RIG_SCALE = "Backbone depth · camera-rig scale"
GEOMETRY_SOURCES = [SENSOR, SENSOR_SCALE, RIG_SCALE]
TRACKER_LABELS = {
    "Analytic demo (no model)": "demo",
    "MVTracker": "mvtracker",
    "TAPIP3D": "tapip3d",
    "TrackCraft3R": "trackcraft3r",
}


class TrackingCancelled(Exception):
    pass


def check_cancel(cancel: Event | None):
    if cancel is not None and cancel.is_set():
        raise TrackingCancelled("Tracking cancelled")


@dataclass(frozen=True)
class ClipSpec:
    start: int = 0
    length: int = 12
    stride: int = 1
    image_long_side: int = 512

    def indices(self, available: int) -> tuple[int, ...]:
        if self.start < 0 or self.length < 2 or self.stride < 1 or self.image_long_side < 16:
            raise ValueError(
                "Choose a nonnegative start, at least 2 frames, positive stride and resolution ≥16"
            )
        indices = tuple(self.start + i * self.stride for i in range(self.length))
        if indices[-1] >= available:
            raise ValueError(
                f"Clip ends at frame {indices[-1]}, but the sequence ends at {available - 1}"
            )
        return indices


@dataclass
class PreparedClip:
    key: tuple
    source_name: str
    trajectory: int
    indices: tuple[int, ...]
    frames: tuple[Frame, ...]
    images: torch.Tensor  # CPU (1,T,V,3,H,W)
    geometry: GeometrySequence  # CPU, dataset world frame
    camera_names: tuple[str, ...]
    confidence: torch.Tensor | None = None  # CPU (1,T,V,H,W), distinct from validity

    def display_result(self, index: int) -> BackboneResult:
        g = self.geometry
        return BackboneResult(
            torch.where(g.valid_depth[0, index], g.depth[0, index], 0),
            None if self.confidence is None else self.confidence[0, index],
            None,
            None,
            g.extrinsics[0, index],
            g.intrinsics[0, index],
        )

    def visible_source_indices(self, filters: PointFilters, index: int = 0) -> torch.Tensor:
        result = self.display_result(index)
        return build_point_cloud(
            result,
            self.images[0, index],
            **filters.cloud_options(result.conf),
            return_source_indices=True,
        )[2]


@dataclass
class TrackingRun:
    clip: PreparedClip
    queries: PointQueries
    output: TrackerOutput
    colors: np.ndarray  # persistent ID colours (N,3), uint8
    elapsed: float


def resize_frame(frame: Frame, long_side: int) -> Frame:
    h, w = frame.images.shape[-2:]
    scale = min(1.0, long_side / max(h, w))
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    rgb = F.interpolate(frame.images.float(), size=size, mode="bilinear", align_corners=False)
    depth = frame.depth
    if depth is not None:
        if depth.ndim == 3:
            depth = depth[:, None]
        valid = torch.isfinite(depth) & (depth > 0)
        depth = F.interpolate(torch.where(valid, depth, 0).float(), size=size, mode="nearest-exact")
    return dataclasses.replace(
        frame, images=rgb.cpu(), depth=None if depth is None else depth.cpu()
    )


def _subset_geometry(g: GeometrySequence, views: list[int]) -> GeometrySequence:
    return GeometrySequence(
        g.depth[:, :, views],
        g.extrinsics[:, :, views],
        g.intrinsics[:, :, views],
        g.valid_depth[:, :, views],
        g.frame_ids,
        g.units,
        dict(g.provenance),
    )


def region_mask(points: torch.Tensor, regions) -> torch.Tensor:
    """Union of 3D query boxes; the caller intersects this with display eligibility."""
    keep = torch.zeros(points.shape[:-1], dtype=torch.bool)
    for lo, hi in regions:
        lo, hi = torch.as_tensor(lo), torch.as_tensor(hi)
        if not bool((hi > lo).all()):
            raise ValueError("Query box maximum must exceed minimum on every axis")
        keep |= ((points >= lo) & (points <= hi)).all(-1)
    return keep


def seed_queries(
    geometry: GeometrySequence,
    count: int,
    *,
    sampling="Even coverage",
    crop=None,
    source_indices=None,
    regions=None,
) -> PointQueries:
    """Choose valid first-frame surface samples once, retaining source view/pixel provenance."""
    if count < 1:
        raise ValueError("Query count must be positive")
    valid = geometry.valid_depth[0, 0]
    v, h, w = valid.shape
    # Bound lifting/FPS cost even for many high-resolution cameras.
    if source_indices is None:
        step = max(1, int(np.ceil(np.sqrt(v * h * w / 16000))))
        coarse = torch.zeros_like(valid)
        coarse[:, step // 2 :: step, step // 2 :: step] = True
        source_indices = torch.where((valid & coarse).flatten())[0]
    else:
        source_indices = source_indices[valid.flatten()[source_indices]]
    vi = source_indices // (h * w)
    y, x = (source_indices % (h * w)) // w, source_indices % w
    if len(x) == 0:
        raise ValueError(
            "No visible depth samples in the reference frame; relax the display filters"
        )
    uv = torch.stack([(x + 0.5) / w, (y + 0.5) / h], -1)[None]
    queries = PointQueries.from_pixels(
        geometry, torch.zeros(1, len(x), dtype=torch.long), vi[None], uv
    )
    keep = torch.ones(len(x), dtype=torch.bool)
    if crop is not None:
        lo, hi = (torch.as_tensor(a) for a in crop)
        if not (hi > lo).all():
            raise ValueError("Workspace maximum must exceed minimum on every axis")
        keep = ((queries.xyz_world[0] >= lo) & (queries.xyz_world[0] <= hi)).all(-1)
    if regions is not None:
        keep &= region_mask(queries.xyz_world[0], regions)
    candidates = torch.where(keep)[0]
    if len(candidates) == 0:
        raise ValueError(
            "The workspace crop or query boxes excludes all visible reference points; enlarge the boxes or relax the filters"
        )
    if len(candidates) > 16000:
        candidates = candidates[torch.linspace(0, len(candidates) - 1, 16000).round().long()]
    n = min(count, len(candidates))
    if sampling == "Farthest points" and n < len(candidates):
        xyz = queries.xyz_world[0, candidates]
        candidates = candidates[furthest_point_indices(xyz, n)]
    else:
        candidates = candidates[torch.linspace(0, len(candidates) - 1, n).round().long()]
    # IDs encode the source pixel, so increasing the budget doesn't recolour existing IDs.
    ids = (vi * h * w + y * w + x)[candidates][None].long()
    return PointQueries(
        ids,
        queries.time[:, candidates],
        queries.xyz_world[:, candidates],
        queries.source_view[:, candidates],
        queries.source_uv[:, candidates],
    )


def id_colors(ids: torch.Tensor) -> np.ndarray:
    import colorsys

    return (
        np.asarray(
            [colorsys.hsv_to_rgb((int(i) * 0.61803398875) % 1.0, 0.65, 1.0) for i in ids[0]],
            dtype=np.float32,
        )
        .__mul__(255)
        .astype(np.uint8)
    )


@dataclass(frozen=True)
class TrackerSettings:
    name: str = "mvtracker"
    checkpoint_path: str = ""
    repo_path: str = ""
    base_model_cache_dir: str = ""
    allow_download: bool = False
    long_side: int = 512


def build_tracker(settings: TrackerSettings, device: str):
    if settings.name not in TRACKERS:
        raise ValueError(f"Unknown tracker: {settings.name}")
    cfg = TRACKERS[settings.name](
        checkpoint_path=settings.checkpoint_path or None,
        allow_download=settings.allow_download,
        long_side=settings.long_side,
    )
    if settings.name == "tapip3d":
        cfg.repo_path = settings.repo_path or None
        cfg.resolution_factor = None
    elif settings.name == "trackcraft3r":
        cfg.repo_path = settings.repo_path or None
        cfg.base_model_cache_dir = settings.base_model_cache_dir or None
        cfg.device = device
    path = None
    if settings.name == "mvtracker" and settings.repo_path:
        path = str(Path(settings.repo_path).expanduser().resolve())
        if not (Path(path) / "mvtracker/models/core/mvtracker/mvtracker.py").is_file():
            raise ValueError("Research checkout must point to the MVTracker repository root")
        loaded = sys.modules.get("mvtracker")
        if loaded is not None and not Path(
            getattr(loaded, "__file__", "")
        ).resolve().is_relative_to(path):
            raise ImportError(
                "Another MVTracker checkout is loaded; restart the viewer to switch repositories"
            )
        sys.path.insert(0, path)
    try:
        return cfg.build().to(device).eval()
    finally:
        if path is not None:
            sys.path.remove(path)


class TrackingRunner:
    def __init__(self, device="cpu", *, builder: Callable = build_tracker):
        self.device, self.builder = device, builder
        self.cached_clip: PreparedClip | None = None

    def prepare(
        self,
        source,
        traj: int,
        spec: ClipSpec,
        camera_names: tuple[str, ...],
        *,
        geometry_source=SENSOR,
        backbone="",
        backbone_runner: BackboneRunner | None = None,
        conditioned=False,
        progress=lambda msg: None,
        cancel: Event | None = None,
    ) -> PreparedClip:
        check_cancel(cancel)
        indices = spec.indices(source.num_timesteps(traj))
        key = (
            id(source),
            traj,
            spec,
            camera_names,
            geometry_source,
            (
                backbone_runner.model_key(backbone)
                if hasattr(backbone_runner, "model_key")
                else backbone
            )
            if geometry_source != SENSOR
            else "",
            conditioned if geometry_source != SENSOR else False,
        )
        if self.cached_clip is not None and self.cached_clip.key == key:
            progress("Reusing cached clip geometry")
            if backbone_runner is not None:
                backbone_runner.release()
            return self.cached_clip
        if not camera_names:
            raise ValueError("Select at least one input camera in 1. Dataset")
        if geometry_source not in GEOMETRY_SOURCES:
            raise ValueError("Unknown geometry source")
        if geometry_source == RIG_SCALE and len(camera_names) < 2:
            raise ValueError(
                "Camera-rig scale needs at least two geometry cameras with a nonzero baseline"
            )
        if geometry_source != SENSOR and backbone_runner is None:
            raise ValueError("A backbone runner is required for predicted geometry")
        frames, display_frames, depths, scales, confidences = [], [], [], [], []
        try:
            for i, t in enumerate(indices):
                check_cancel(cancel)
                progress(f"Preparing frame {i + 1}/{len(indices)} · dataset frame {t}")
                frame = resize_frame(
                    source.get_frame(traj, t, with_depth=True), spec.image_long_side
                )
                display_frames.append(frame)
                missing = set(camera_names) - set(frame.cam_names)
                if missing:
                    raise ValueError(f"Frame {t} is missing selected cameras: {sorted(missing)}")
                ix = [frame.cam_names.index(name) for name in camera_names]
                frame = dataclasses.replace(
                    frame,
                    images=frame.images[ix],
                    intrinsics=frame.intrinsics[ix],
                    extrinsics=frame.extrinsics[ix],
                    cam_names=list(camera_names),
                    depth=None if frame.depth is None else frame.depth[ix],
                )
                if frames and frame.images.shape != frames[0].images.shape:
                    raise ValueError(
                        "Selected cameras must have consistent image dimensions through the clip"
                    )
                if geometry_source == SENSOR:
                    if frame.depth is None:
                        raise ValueError(
                            "This clip has no sensor/GT depth; select a backbone geometry source"
                        )
                    depth = frame.depth[:, 0]
                    scale = 1.0
                    confidence = None
                else:
                    if i == 0:
                        check_cancel(cancel)
                        backbone_runner.set_long_side(spec.image_long_side)
                        backbone_runner.load(backbone)
                    result = backbone_runner.run(
                        frame, list(range(len(ix))), condition_on_gt_cameras=conditioned
                    )
                    depth = result.depth
                    confidence = result.conf
                    if geometry_source == SENSOR_SCALE:
                        if frame.depth is None:
                            raise ValueError(
                                "Sensor scale requires recorded depth for the selected cameras"
                            )
                        reference = F.interpolate(
                            frame.depth, size=depth.shape[-2:], mode="nearest-exact"
                        )[:, 0]
                        valid = (
                            torch.isfinite(depth)
                            & (depth > 0)
                            & torch.isfinite(reference)
                            & (reference > 0)
                        )
                        if valid.sum() < 16:
                            raise ValueError(
                                f"Frame {t} has too few valid depth samples for scale calibration"
                            )
                        scale = float(fit_depth_scale(depth, reference, mask=valid))
                    else:
                        if result.pred_extrinsics is None:
                            raise ValueError(
                                "This backbone does not predict cameras; choose sensor scale instead"
                            )
                        for poses in (result.pred_extrinsics, result.gt_extrinsics):
                            centers = poses[:, :3, 3]
                            if (centers - centers.mean(0)).norm(dim=-1).max() < 1e-5:
                                raise ValueError(
                                    "Camera-rig scale is undefined for coincident camera centres"
                                )
                        _, _, fitted = align_camera_poses_sim3(
                            result.pred_extrinsics[None], result.gt_extrinsics[None]
                        )
                        scale = float(fitted[0])
                    if not np.isfinite(scale) or scale <= 0:
                        raise ValueError(
                            f"Invalid geometry calibration scale at frame {t}: {scale}"
                        )
                    depth = depth * scale
                    depth = torch.where(torch.isfinite(depth) & (depth > 0), depth, 0)
                    depth = F.interpolate(
                        depth[:, None], size=frame.images.shape[-2:], mode="nearest-exact"
                    )[:, 0]
                    if confidence is not None:
                        confidence = (
                            F.interpolate(
                                confidence[:, None],
                                size=frame.images.shape[-2:],
                                mode="nearest-exact",
                            )[:, 0]
                            .detach()
                            .cpu()
                        )
                frames.append(frame)
                depths.append(depth)
                scales.append(scale)
                confidences.append(confidence)
        finally:
            if backbone_runner is not None:
                backbone_runner.release()
        check_cancel(cancel)
        geometry = GeometrySequence(
            torch.stack(depths)[None],
            torch.stack([f.extrinsics for f in frames])[None],
            torch.stack([f.intrinsics for f in frames])[None],
            frame_ids=(f"{source.name}:{traj}:dataset-world",),
            units=("meters",),
            provenance={
                "source": geometry_source,
                "backbone": backbone if geometry_source != SENSOR else None,
                "backbone_configuration": str(backbone_runner.model_key(backbone))
                if geometry_source != SENSOR and hasattr(backbone_runner, "model_key")
                else None,
                "camera_source": "dataset calibration",
                "per_frame_depth_scale": scales,
                "alignment_stage": "geometry preparation; trajectories are never fitted",
            },
        )
        geometry.validate()
        clip = PreparedClip(
            key,
            source.name,
            traj,
            indices,
            tuple(display_frames),
            torch.stack([f.images for f in frames])[None],
            geometry,
            camera_names,
            torch.stack(confidences)[None] if all(c is not None for c in confidences) else None,
        )
        self.cached_clip = clip
        return clip

    def run(
        self,
        clip: PreparedClip,
        settings: TrackerSettings,
        *,
        source=None,
        query_view=None,
        count=512,
        sampling="Even coverage",
        crop=None,
        point_filters: PointFilters | None = None,
        regions=None,
        progress=lambda msg: None,
        cancel=None,
    ) -> TrackingRun:
        check_cancel(cancel)
        if settings.name != "demo" and settings.name not in TRACKERS:
            raise ValueError(f"Unknown tracker: {settings.name}")
        mono = settings.name != "demo" and not TRACKERS[settings.name].CAPABILITIES.multiview
        if settings.name == "trackcraft3r" and len(clip.indices) != 12:
            raise ValueError("TrackCraft3R requires a 12-frame clip in this viewer")
        if settings.name == "mvtracker" and len(clip.indices) < 7:
            raise ValueError("MVTracker requires at least 7 clip frames")
        views = list(range(len(clip.camera_names)))
        if mono:
            if query_view not in clip.camera_names:
                raise ValueError("Select a tracker camera that is enabled in 1. Dataset")
            views = [clip.camera_names.index(query_view)]
        geometry = _subset_geometry(clip.geometry, views)
        source_indices = None
        if point_filters is not None:
            source_indices = clip.visible_source_indices(point_filters)
            if mono:
                pixels = geometry.depth.shape[-2] * geometry.depth.shape[-1]
                source_indices = source_indices[source_indices // pixels == views[0]] % pixels
        queries = seed_queries(
            geometry,
            count,
            sampling=sampling,
            crop=crop,
            source_indices=source_indices,
            regions=regions,
        )
        started = time.perf_counter()
        if settings.name == "demo":
            if source is None or source.name != "demo":
                raise ValueError("Analytic demo is only available for the synthetic demo scene")
            output = source.trajectories(clip.indices, queries, geometry)
        else:
            progress(f"Loading {settings.name} · {queries.ids.shape[1]} queries")
            model = self.builder(settings, self.device)
            try:
                check_cancel(cancel)
                geo = dataclasses.replace(
                    geometry,
                    **{
                        field: getattr(geometry, field).to(self.device)
                        for field in ("depth", "extrinsics", "intrinsics", "depth_valid")
                    },
                )
                query = dataclasses.replace(
                    queries,
                    **{
                        field: getattr(queries, field).to(self.device)
                        for field in ("ids", "time", "xyz_world", "source_view", "source_uv")
                    },
                )
                progress(
                    f"Tracking with {settings.name} · {len(clip.indices)} frames · {len(views)} view(s)"
                )
                with torch.inference_mode():
                    output = model(clip.images[:, :, views].to(self.device), query, geometry=geo)
                output = dataclasses.replace(
                    output,
                    **{
                        f.name: getattr(output, f.name).detach().cpu()
                        for f in dataclasses.fields(output)
                        if isinstance(getattr(output, f.name), torch.Tensor)
                    },
                )
            finally:
                # Keep only CPU output after each run so stages do not compete for VRAM.
                geo = query = None
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        check_cancel(cancel)
        output.metadata.update(
            dataset_frames=clip.indices,
            tracking_cameras=[clip.camera_names[i] for i in views],
            query_display_filters=None
            if point_filters is None
            else dataclasses.asdict(point_filters),
            query_regions=regions,
        )
        return TrackingRun(
            clip, queries, output, id_colors(queries.ids), time.perf_counter() - started
        )


def visible_query_mask(run: TrackingRun, filters: PointFilters, regions=None) -> torch.Tensor:
    """Keep whole trajectories whose reference pixels still belong to the shown cloud."""
    clip, queries = run.clip, run.queries
    h, w = clip.geometry.depth.shape[-2:]
    views = torch.tensor(
        [clip.camera_names.index(name) for name in run.output.metadata["tracking_cameras"]]
    )
    view = views[queries.source_view[0]]
    x = (queries.source_uv[0, :, 0] * w - 0.5).round().long()
    y = (queries.source_uv[0, :, 1] * h - 0.5).round().long()
    ids = view * h * w + y * w + x
    keep = torch.isin(ids, clip.visible_source_indices(filters))
    if regions is not None:
        keep &= region_mask(queries.xyz_world[0], regions)
    return keep


def visible_track_samples(run: TrackingRun, filters: PointFilters) -> torch.Tensor:
    """Apply workspace and depth-confidence exclusion at every trajectory sample.

    Confidence is sampled at the current projected position, with the same
    per-frame percentile as the cloud. Multi-view tracks may be supported by any
    tracking camera; query-view tracks must pass in their own source camera.
    Point-count thinning and query boxes select identities at the reference frame
    only: they do not snap moving trajectories to a changing display grid.
    """
    clip, out = run.clip, run.output
    xyz = out.tracks_world[0]
    keep = torch.isfinite(xyz).all(-1)
    if filters.crop is not None:
        keep &= region_mask(xyz, (filters.crop,))
    if clip.confidence is None:
        return keep
    geometry = clip.geometry
    h, w = geometry.depth.shape[-2:]
    tracking_views = torch.tensor(
        [clip.camera_names.index(name) for name in out.metadata["tracking_cameras"]]
    )
    query_views = tracking_views[run.queries.source_view[0]]
    for t in range(len(clip.indices)):
        confidence = clip.confidence[0, t]
        threshold = confidence_threshold(confidence, filters.drop_conf_pct)
        uv, in_front = project_world_points(
            xyz[t].expand(len(clip.camera_names), -1, -1),
            geometry.extrinsics[0, t],
            geometry.intrinsics[0, t],
        )
        inside = in_front & torch.isfinite(uv).all(-1) & ((uv >= 0) & (uv < 1)).all(-1)
        uv = torch.nan_to_num(uv, nan=0.0, posinf=0.0, neginf=0.0).clamp(0, 1)
        x, y = (uv[..., 0] * w).long().clamp(max=w - 1), (uv[..., 1] * h).long().clamp(max=h - 1)
        view = torch.arange(len(clip.camera_names))[:, None]
        supported = inside & geometry.valid_depth[0, t, view, y, x]
        supported &= confidence[view, y, x] > threshold
        if out.visibility_scope == "query_view":
            keep[t] &= supported[query_views, torch.arange(xyz.shape[1])]
        else:
            keep[t] &= supported[tracking_views].any(0)
    return keep


def render_tracks(
    run: TrackingRun,
    index: int,
    *,
    trail_length=8,
    show_occluded=True,
    threshold=0.5,
    query_mask=None,
    sample_mask=None,
):
    """Points/colours and line segments; never bridge an invalid trajectory sample."""
    out = run.output
    xyz, valid = out.tracks_world[0].numpy(), out.valid[0].numpy().copy()
    valid &= np.isfinite(xyz).all(-1)
    if query_mask is not None:
        valid &= np.asarray(query_mask, dtype=bool)[None]
    if sample_mask is not None:
        valid &= np.asarray(sample_mask, dtype=bool)
    visibility = None if out.visibility is None else out.visibility[0].numpy()
    if not show_occluded and visibility is not None:
        valid &= visibility >= threshold
    colors = run.colors.copy()
    if visibility is not None:
        colors = (colors * np.where(visibility[index, :, None] >= threshold, 1.0, 0.3)).astype(
            np.uint8
        )
    start = max(0, index - trail_length)
    edges = valid[start:index] & valid[start + 1 : index + 1]
    segments = np.stack([xyz[start:index], xyz[start + 1 : index + 1]], axis=-2)[edges]
    line_colors = np.broadcast_to(run.colors[None, :, None], (index - start, len(colors), 2, 3))[
        edges
    ].copy()
    return xyz[index, valid[index]], colors[valid[index]], segments, line_colors


def export_tracks(run: TrackingRun) -> bytes:
    out = run.output
    payload = {name: getattr(out, name).numpy() for name in ("ids", "tracks_world", "valid")}
    for name in ("visibility", "visibility_per_view"):
        if getattr(out, name) is not None:
            payload[name] = getattr(out, name).numpy()
    for name in ("time", "xyz_world", "source_view", "source_uv"):
        payload[f"query_{name}"] = getattr(run.queries, name).numpy()
    views = [run.clip.camera_names.index(n) for n in out.metadata["tracking_cameras"]]
    payload.update(
        frame_indices=np.asarray(run.clip.indices),
        camera_to_world=run.clip.geometry.extrinsics[:, :, views].numpy(),
        intrinsics_normalized=run.clip.geometry.intrinsics[:, :, views].numpy(),
        metadata_json=np.asarray(
            json.dumps({**out.metadata, "visibility_scope": out.visibility_scope}, default=str)
        ),
    )
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **payload)
    return buffer.getvalue()
