"""Shared geometry and trajectory contract for optional pretrained point trackers.

Images use ``(B,T,V,3,H,W)``, c2w poses and normalized pixel-center intrinsics.
Depth, camera translations and queries must already share one frame and scale per
sequence. These containers never align predictions to ground truth implicitly.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from ontic_nn.wrappers.common import BackboneOutput


@dataclass(frozen=True)
class TrackerCapabilities:
    multiview: bool
    dense: bool = False
    arbitrary_query_times: bool = True
    execution_mode: str = "offline"
    visibility_scope: str = "query_view"


@dataclass(kw_only=True)
class TrackerConfig:
    freeze_tracker: bool = True
    long_side: int = 512
    checkpoint_path: str | None = None
    cache_dir: str | None = None
    allow_download: bool = True

    def build(self) -> TrackerBase:
        raise NotImplementedError


class TrackerBase(nn.Module):
    CAPABILITIES: ClassVar[TrackerCapabilities]
    cfg: TrackerConfig

    @property
    def capabilities(self) -> TrackerCapabilities:
        return self.CAPABILITIES

    def _configure_model(self, model: nn.Module, cfg: TrackerConfig) -> None:
        self.model = model
        self.cfg = cfg
        if cfg.long_side <= 0:
            raise ValueError("long_side must be positive")
        self.model.requires_grad_(not cfg.freeze_tracker)
        if cfg.freeze_tracker:
            self.model.eval()

    def train(self, mode: bool = True) -> TrackerBase:
        super().train(mode)
        if self.cfg.freeze_tracker:
            self.model.eval()
        return self

    def _grad_context(self):
        return torch.set_grad_enabled(torch.is_grad_enabled() and not self.cfg.freeze_tracker)

    @abstractmethod
    def forward(
        self, images: Tensor, queries: PointQueries, *, geometry: GeometrySequence
    ) -> TrackerOutput:
        raise NotImplementedError


def _require_float(name: str, value: Tensor) -> None:
    if not value.is_floating_point():
        raise ValueError(f"{name} must be floating point")


@dataclass
class GeometrySequence:
    depth: Tensor  # (B,T,V,Hd,Wd), camera-z, potentially invalid/masked
    extrinsics: Tensor  # (B,T,V,4,4), c2w
    intrinsics: Tensor  # (B,T,V,3,3), normalized
    depth_valid: Tensor | None = None
    frame_ids: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    provenance: dict = field(default_factory=dict)

    @property
    def valid_depth(self) -> Tensor:
        valid = torch.isfinite(self.depth) & (self.depth > 0)
        return valid if self.depth_valid is None else valid & self.depth_valid

    def validate(self) -> None:
        if self.depth.ndim != 5 or min(self.depth.shape) <= 0:
            raise ValueError("depth must have nonempty shape (B,T,V,Hd,Wd)")
        _require_float("depth", self.depth)
        b, t, v = self.depth.shape[:3]
        for name, value, shape in (
            ("extrinsics", self.extrinsics, (b, t, v, 4, 4)),
            ("intrinsics", self.intrinsics, (b, t, v, 3, 3)),
        ):
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
            _require_float(name, value)
            if value.device != self.depth.device:
                raise ValueError(f"{name} and depth must be on the same device")
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
        bottom = self.extrinsics.new_tensor([0, 0, 0, 1])
        if not torch.allclose(self.extrinsics[..., 3, :], bottom.expand(b, t, v, 4)):
            raise ValueError("extrinsics must be homogeneous camera-to-world matrices")
        if (self.intrinsics[..., 0, 0] <= 0).any() or (self.intrinsics[..., 1, 1] <= 0).any():
            raise ValueError("intrinsics must have positive focal lengths")
        kbottom = self.intrinsics.new_tensor([0, 0, 1])
        if not torch.allclose(self.intrinsics[..., 2, :], kbottom.expand(b, t, v, 3)):
            raise ValueError("intrinsics must be pinhole matrices with bottom row (0,0,1)")
        rotation = self.extrinsics[..., :3, :3].float()
        identity = torch.eye(3, device=rotation.device).expand_as(rotation)
        if not torch.allclose(rotation.mT @ rotation, identity, atol=2e-3, rtol=2e-3):
            raise ValueError(
                "extrinsics must contain rigid rotations (scale depth/centers separately)"
            )
        if not torch.allclose(
            torch.linalg.det(rotation), torch.ones_like(rotation[..., 0, 0]), atol=2e-3, rtol=2e-3
        ):
            raise ValueError("extrinsics rotations must have determinant +1")
        if self.depth_valid is not None:
            if self.depth_valid.shape != self.depth.shape or self.depth_valid.dtype != torch.bool:
                raise ValueError("depth_valid must be bool with the depth shape")
            if self.depth_valid.device != self.depth.device:
                raise ValueError("depth_valid and depth must be on the same device")
        if self.frame_ids and (len(self.frame_ids) != b or not all(self.frame_ids)):
            raise ValueError("frame_ids must contain one nonempty world-frame identifier per batch")
        if self.units and (
            len(self.units) != b or any(u not in ("meters", "arbitrary") for u in self.units)
        ):
            raise ValueError("units must contain 'meters' or 'arbitrary' per batch item")

    @classmethod
    def from_backbone_outputs(
        cls,
        outputs: Sequence[BackboneOutput],
        *,
        camera_source: Literal["provided", "predicted"],
        shared_world_frame: bool,
        frame_ids: tuple[str, ...] = (),
        units: tuple[str, ...] = (),
    ) -> GeometrySequence:
        """Stack time-ordered backbone outputs after the caller establishes a shared frame.

        The camera source is explicit: no automatic GT preference or fallback.
        This method does not solve per-frame scale/pose drift. If depth was
        predicted in another gauge, align depth and cameras before calling it.
        """
        if not outputs:
            raise ValueError("at least one BackboneOutput is required")
        if not shared_world_frame:
            raise ValueError("establish a shared world frame and scale before stacking geometry")
        if camera_source not in ("provided", "predicted"):
            raise ValueError("camera_source must be 'provided' or 'predicted'")
        suffix = "_pred" if camera_source == "predicted" else ""
        keys = ("depth", f"extrinsics{suffix}", f"intrinsics{suffix}")
        for out in outputs:
            missing = [key for key in keys if key not in out.data]
            if missing:
                raise ValueError(f"BackboneOutput lacks selected geometry: {', '.join(missing)}")
        depth = torch.stack([out.data["depth"] for out in outputs], dim=1)
        valid = torch.stack(
            [
                torch.isfinite(out.data["depth"])
                & (out.data["depth"] > 0)
                & ~out.data.get(
                    "sky_mask", torch.zeros_like(out.data["depth"], dtype=torch.bool)
                ).bool()
                for out in outputs
            ],
            dim=1,
        )
        result = cls(
            depth=depth,
            extrinsics=torch.stack([out.data[keys[1]] for out in outputs], dim=1),
            intrinsics=torch.stack([out.data[keys[2]] for out in outputs], dim=1),
            depth_valid=valid,
            frame_ids=frame_ids,
            units=units,
            provenance={"camera_source": camera_source, "shared_world_frame": "caller_asserted"},
        )
        result.validate()
        return result


@dataclass
class PointQueries:
    ids: Tensor  # int64 (B,N)
    time: Tensor  # int64 (B,N), clip indices
    xyz_world: Tensor  # (B,N,3)
    source_view: Tensor | None = None
    source_uv: Tensor | None = None  # normalized edge-origin pixel centers

    @classmethod
    def from_pixels(
        cls,
        geometry: GeometrySequence,
        time: Tensor,
        source_view: Tensor,
        source_uv: Tensor,
        *,
        ids: Tensor | None = None,
    ) -> PointQueries:
        """Lift normalized image queries using nearest valid camera-z depth.

        IDs default to query order. No interpolation across invalid depth holes.
        """
        geometry.validate()
        b = geometry.depth.shape[0]
        if time.ndim != 2 or time.shape[0] != b:
            raise ValueError("time must have shape (B,N)")
        if ids is None:
            ids = torch.arange(time.shape[1], device=time.device).expand(b, -1).clone()
        result = cls(ids, time, source_uv.new_zeros((*time.shape, 3)), source_view, source_uv)
        _validate_queries(result, geometry)
        batch = torch.arange(b, device=time.device)[:, None]
        depth, valid = _query_depth(result, geometry, source_uv)
        if not valid.all():
            raise ValueError("queries must sample valid finite positive depth")
        k = geometry.intrinsics[batch, time, source_view].float()
        c2w = geometry.extrinsics[batch, time, source_view].float()
        uv1 = torch.cat([source_uv.float(), torch.ones_like(source_uv[..., :1])], dim=-1)
        xyz_cam = (torch.linalg.inv(k) @ uv1.unsqueeze(-1)).squeeze(-1) * depth[..., None]
        result.xyz_world = (c2w[..., :3, :3] @ xyz_cam.unsqueeze(-1)).squeeze(-1) + c2w[..., :3, 3]
        return result


@dataclass
class TrackerOutput:
    ids: Tensor
    tracks_world: Tensor
    valid: Tensor
    visibility: Tensor | None = None
    visibility_scope: str = "unavailable"
    visibility_per_view: Tensor | None = None
    metadata: dict = field(default_factory=dict)


def _validate_queries(queries: PointQueries, geometry: GeometrySequence) -> None:
    b, t, v = geometry.depth.shape[:3]
    if queries.ids.ndim != 2 or queries.ids.shape[0] != b:
        raise ValueError("query ids must have shape (B,N)")
    shape = queries.ids.shape
    if queries.ids.dtype != torch.int64 or queries.time.dtype != torch.int64:
        raise ValueError("query ids and time must be int64")
    if queries.time.shape != shape or queries.xyz_world.shape != (*shape, 3):
        raise ValueError("query time/xyz_world shapes must match ids (B,N)/(B,N,3)")
    _require_float("xyz_world", queries.xyz_world)
    paired = (queries.source_view is None, queries.source_uv is None)
    if paired[0] != paired[1]:
        raise ValueError("source_view and source_uv must be supplied together")
    tensors = [queries.ids, queries.time, queries.xyz_world]
    if queries.source_view is not None:
        if queries.source_view.shape != shape or queries.source_view.dtype != torch.int64:
            raise ValueError("source_view must be int64 (B,N)")
        if queries.source_uv.shape != (*shape, 2):
            raise ValueError("source_uv must have shape (B,N,2)")
        _require_float("source_uv", queries.source_uv)
        tensors += [queries.source_view, queries.source_uv]
    if any(x.device != geometry.depth.device for x in tensors):
        raise ValueError("queries and geometry must be on the same device")
    if not torch.isfinite(queries.xyz_world).all():
        raise ValueError("xyz_world must be finite")
    if ((queries.time < 0) | (queries.time >= t)).any():
        raise ValueError("query time is outside the supplied clip")
    if any(torch.unique(row).numel() != row.numel() for row in queries.ids):
        raise ValueError("query IDs must be unique within each sequence")
    if queries.source_view is not None:
        if ((queries.source_view < 0) | (queries.source_view >= v)).any():
            raise ValueError("source_view is outside the supplied cameras")
        if (
            not torch.isfinite(queries.source_uv).all()
            or ((queries.source_uv < 0) | (queries.source_uv >= 1)).any()
        ):
            raise ValueError("source_uv must be finite normalized coordinates in [0,1)")


def validate_tracker_inputs(
    images: Tensor, queries: PointQueries, geometry: GeometrySequence, *, multiview: bool = True
) -> tuple[int, int, int, int]:
    geometry.validate()
    if images.ndim != 6 or images.shape[3] != 3 or min(images.shape) <= 0:
        raise ValueError("images must have nonempty shape (B,T,V,3,H,W)")
    if images.shape[:3] != geometry.depth.shape[:3]:
        raise ValueError("images and geometry batch/time/view axes must match")
    _require_float("images", images)
    if images.device != geometry.depth.device:
        raise ValueError("images and geometry must be on the same device")
    if not torch.isfinite(images).all() or ((images < 0) | (images > 1)).any():
        raise ValueError("images must be finite RGB values in [0,1]")
    b, t, v = images.shape[:3]
    if not multiview and v != 1:
        raise ValueError("this tracker is monocular and requires V=1")
    _validate_queries(queries, geometry)
    return b, t, v, queries.ids.shape[1]


def resize_tracker_inputs(
    images: Tensor, geometry: GeometrySequence, *, size: tuple[int, int]
) -> tuple[Tensor, Tensor, Tensor]:
    """Resize images and masked z-depth; never change depth units or the source geometry."""
    if len(size) != 2 or min(size) <= 0:
        raise ValueError("size must be positive (height,width)")
    lead = images.shape[:3]
    rgb = F.interpolate(images.flatten(0, 2), size=size, mode="bilinear", align_corners=False)
    valid = geometry.valid_depth
    safe_depth = torch.where(valid, geometry.depth, 0)
    depth = F.interpolate(safe_depth.flatten(0, 2)[:, None], size=size, mode="nearest-exact")
    mask = F.interpolate(valid.flatten(0, 2)[:, None].float(), size=size, mode="nearest-exact")
    return (
        rgb.reshape(*lead, 3, *size),
        depth.reshape(*lead, *size),
        mask.reshape(*lead, *size).bool(),
    )


def pixel_intrinsics(
    intrinsics: Tensor, height: int, width: int, *, integer_centers: bool = True
) -> Tensor:
    """Convert normalized edge-origin K to pixels, optionally centered on integer indices."""
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    k = intrinsics.float().clone()
    k[..., 0, :] *= width
    k[..., 1, :] *= height
    if integer_centers:
        k[..., 0, 2] -= 0.5
        k[..., 1, 2] -= 0.5
    return k


def _query_depth(queries: PointQueries, geometry: GeometrySequence, uv: Tensor):
    b, _, v, h, w = geometry.depth.shape
    views = queries.source_view
    if views is None:
        if v != 1:
            raise ValueError("source_view is required to select a camera for multi-view queries")
        views = torch.zeros_like(queries.time)
    batch = torch.arange(b, device=uv.device)[:, None]
    x = (uv[..., 0] * w).floor().long().clamp(0, w - 1)
    y = (uv[..., 1] * h).floor().long().clamp(0, h - 1)
    index = (batch, queries.time, views, y, x)
    return geometry.depth[index].float(), geometry.valid_depth[index]


def project_queries(
    queries: PointQueries, geometry: GeometrySequence, *, depth_tolerance: float | None = 0.05
) -> Tensor:
    """Project queries in their source view and check source geometry agreement.

    XYZ-only queries require V=1. By default reject points inconsistent with
    observed depth by more than 5% (e.g. behind an occluder). This is a query
    initialization check, not a visibility estimate for subsequent trajectories.
    """
    _validate_queries(queries, geometry)
    b, _, v, h, w = geometry.depth.shape
    views = queries.source_view
    if views is None:
        if v != 1:
            raise ValueError("source_view is required for multi-view image projection")
        views = torch.zeros_like(queries.time)
    batch = torch.arange(b, device=queries.time.device)[:, None]
    c2w = geometry.extrinsics[batch, queries.time, views].float()
    k = geometry.intrinsics[batch, queries.time, views].float()
    cam = (
        c2w[..., :3, :3].mT @ (queries.xyz_world.float() - c2w[..., :3, 3]).unsqueeze(-1)
    ).squeeze(-1)
    if (cam[..., 2] <= 0).any():
        raise ValueError("queries must lie in front of their source camera")
    xyh = (k @ cam.unsqueeze(-1)).squeeze(-1)
    uv = xyh[..., :2] / xyh[..., 2:]
    if ((uv < 0) | (uv >= 1)).any() or not torch.isfinite(uv).all():
        raise ValueError("queries project outside their source image")
    if queries.source_uv is not None:
        pixel_error = (uv - queries.source_uv).abs() * uv.new_tensor([w, h])
        if (pixel_error > 0.51).any():
            raise ValueError("source_uv and xyz_world disagree in the supplied geometry")
        uv = queries.source_uv.float()
    if depth_tolerance is not None:
        if depth_tolerance < 0:
            raise ValueError("depth_tolerance must be nonnegative or None")
        depth, valid = _query_depth(queries, geometry, uv)
        if (
            not valid.all()
            or ((cam[..., 2] - depth).abs() > depth.abs() * depth_tolerance + 1e-5).any()
        ):
            raise ValueError("query does not agree with visible source depth")
    return uv


def make_tracker_output(
    queries: PointQueries,
    geometry: GeometrySequence,
    tracks_world: Tensor,
    visibility: Tensor | None = None,
    *,
    visibility_scope: str = "query_view",
    valid: Tensor | None = None,
    metadata: dict | None = None,
) -> TrackerOutput:
    b, t = geometry.depth.shape[:2]
    shape = (b, t, queries.ids.shape[1])
    if tracks_world.shape != (*shape, 3):
        raise ValueError(f"upstream tracks must have shape {(*shape, 3)}")
    if tracks_world.device != queries.ids.device:
        raise ValueError("upstream tracks and query IDs must share a device")
    tracks_world = tracks_world.float()
    finite = torch.isfinite(tracks_world).all(dim=-1)
    if valid is not None:
        if valid.shape != shape or valid.dtype != torch.bool or valid.device != tracks_world.device:
            raise ValueError("valid must be bool (B,T,N) on the tracks device")
        finite = finite & valid
    if visibility is not None:
        if visibility.shape != shape or visibility.device != tracks_world.device:
            raise ValueError("upstream visibility must be (B,T,N) on the tracks device")
        visibility = visibility.float()
        if not torch.isfinite(visibility).all() or ((visibility < 0) | (visibility > 1)).any():
            raise ValueError("upstream visibility must contain finite probabilities in [0,1]")
    if visibility_scope not in ("any_view", "query_view", "unavailable"):
        raise ValueError("unknown visibility_scope")
    meta = dict(metadata or {})
    meta.update(
        frame_ids=geometry.frame_ids,
        units=geometry.units or ("arbitrary",) * b,
        geometry_provenance=dict(geometry.provenance),
    )
    return TrackerOutput(
        queries.ids,
        tracks_world,
        finite,
        visibility,
        visibility_scope if visibility is not None else "unavailable",
        metadata=meta,
    )


def empty_tracker_output(
    images: Tensor, queries: PointQueries, geometry: GeometrySequence, *, visibility_scope: str
) -> TrackerOutput:
    b, t = images.shape[:2]
    return make_tracker_output(
        queries,
        geometry,
        images.new_empty((b, t, 0, 3)),
        images.new_empty((b, t, 0)),
        visibility_scope=visibility_scope,
    )
