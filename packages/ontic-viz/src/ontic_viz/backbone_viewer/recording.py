"""Portable viewer recordings: calibrated clips with optional trajectories, without pickle."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch

from ontic_nn.trackers import GeometrySequence, PointQueries, TrackerOutput

from .data_source import Frame
from .tracking import PreparedClip, TrackingRun, export_tracks, id_colors


def save_recording(value: TrackingRun | PreparedClip, path: str | Path) -> Path:
    clip = value.clip if isinstance(value, TrackingRun) else value
    payload = {}
    if isinstance(value, TrackingRun):
        with np.load(io.BytesIO(export_tracks(value)), allow_pickle=False) as data:
            payload.update({k: data[k] for k in data.files})
    frames = clip.frames
    payload.update(
        viewer_version=np.asarray(1),
        viewer_metadata=np.asarray(
            json.dumps(
                {
                    "source": clip.source_name,
                    "trajectory": clip.trajectory,
                    "camera_names": clip.camera_names,
                    "frame_camera_names": [f.cam_names for f in frames],
                    "frame_ids": clip.geometry.frame_ids,
                    "units": clip.geometry.units,
                    "geometry_provenance": clip.geometry.provenance,
                    "tracking_seconds": value.elapsed if isinstance(value, TrackingRun) else None,
                },
                default=str,
            )
        ),
        frame_indices=np.asarray(clip.indices),
        images=clip.images.numpy(),
        geometry_depth=clip.geometry.depth.numpy(),
        geometry_valid=clip.geometry.valid_depth.numpy(),
        geometry_c2w=clip.geometry.extrinsics.numpy(),
        geometry_intrinsics=clip.geometry.intrinsics.numpy(),
    )
    if clip.confidence is not None:
        payload["geometry_confidence"] = clip.confidence.numpy()
    # Keep frame-level optional overlays separate; hand/robot shapes can change over time.
    for i, frame in enumerate(frames):
        for field in ("images", "extrinsics", "intrinsics", "depth", "hands"):
            tensor = getattr(frame, field)
            if tensor is not None:
                payload[f"frame_{i}_{field}"] = tensor.numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_recording(path: str | Path) -> TrackingRun | PreparedClip:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if int(data["viewer_version"]) != 1:
            raise ValueError("Unsupported viewer recording version")
        meta = json.loads(str(data["viewer_metadata"]))

        def tensor(key):
            return torch.from_numpy(data[key].copy())

        geometry = GeometrySequence(
            tensor("geometry_depth"),
            tensor("geometry_c2w"),
            tensor("geometry_intrinsics"),
            tensor("geometry_valid"),
            tuple(meta["frame_ids"]),
            tuple(meta["units"]),
            meta["geometry_provenance"],
        )
        geometry.validate()
        indices = tuple(int(i) for i in data["frame_indices"])
        frames = tuple(
            Frame(
                **{f: tensor(f"frame_{i}_{f}") for f in ("images", "extrinsics", "intrinsics")},
                cam_names=meta["frame_camera_names"][i],
                depth=tensor(f"frame_{i}_depth") if f"frame_{i}_depth" in data else None,
                hands=tensor(f"frame_{i}_hands") if f"frame_{i}_hands" in data else None,
            )
            for i in range(len(indices))
        )
        clip = PreparedClip(
            ("recording", str(path.resolve())),
            meta["source"],
            meta["trajectory"],
            indices,
            frames,
            tensor("images"),
            geometry,
            tuple(meta["camera_names"]),
            tensor("geometry_confidence") if "geometry_confidence" in data else None,
        )
        if clip.confidence is not None and clip.confidence.shape != geometry.depth.shape:
            raise ValueError("Recording confidence does not match its depth dimensions")
        if "tracks_world" not in data:
            return clip
        output_meta = json.loads(str(data["metadata_json"]))
        output = TrackerOutput(
            tensor("ids"),
            tensor("tracks_world"),
            tensor("valid"),
            tensor("visibility") if "visibility" in data else None,
            output_meta.pop("visibility_scope"),
            tensor("visibility_per_view") if "visibility_per_view" in data else None,
            output_meta,
        )
        queries = PointQueries(
            output.ids,
            tensor("query_time"),
            tensor("query_xyz_world"),
            tensor("query_source_view"),
            tensor("query_source_uv"),
        )
        if output.tracks_world.shape != (1, len(indices), output.ids.shape[1], 3):
            raise ValueError("Recording tracks do not match its clip and query dimensions")
        return TrackingRun(clip, queries, output, id_colors(output.ids), meta["tracking_seconds"])


def recording_label(path: str | Path) -> str:
    """Read only the small metadata entries to name a saved result in the GUI."""
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["viewer_metadata"]))
        source = {"hocap": "HOCAP", "synthrobot": "SynthRobot", "physinone": "PhysInOne"}.get(
            meta["source"], meta["source"]
        )
        backbone = meta["geometry_provenance"].get("backbone")
        depth = {None: "Sensor depth", "vggt": "VGGT depth", "moge3": "MoGe-3 depth"}.get(
            backbone, backbone
        )
        if "metadata_json" in data:
            tracker = json.loads(str(data["metadata_json"])).get("tracker", "tracks")
            tracker = {
                "mvtracker": "MVTracker",
                "tapip3d": "TAPIP3D",
                "trackcraft3r": "TrackCraft3R",
                "analytic_demo": "Analytic demo",
            }.get(tracker, tracker)
        else:
            tracker = "Preview"
        return f"{source} · {depth} → {tracker}"
