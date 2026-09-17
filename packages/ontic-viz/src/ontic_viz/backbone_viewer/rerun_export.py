"""Optional headless Rerun recording of a completed tracking run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .render import build_point_cloud
from .tracking import TrackingRun, render_tracks


def save_rerun(
    run: TrackingRun, path: str | Path, *, trail_length: int = 10, cloud_stride: int = 4
) -> Path:
    """Save calibrated clouds, images/cameras and persistent-ID tracks to ``.rrd``.

    No browser, desktop viewer, GPU, or Rerun server is started. Install the
    optional ``ontic-viz[rerun]`` extra to use this exporter.
    """
    try:
        import rerun as rr
    except ImportError as e:
        raise ImportError("Rerun export needs the optional ontic-viz[rerun] dependencies") from e
    if trail_length < 0 or cloud_stride < 1:
        raise ValueError("Use a nonnegative trail length and positive cloud stride")
    path = Path(path)
    recording = rr.RecordingStream("ontic-point-tracking")
    recording.save(path)
    try:
        metadata = {
            **run.output.metadata,
            "visibility_scope": run.output.visibility_scope,
            "point_ids": run.output.ids[0].tolist(),
        }
        recording.log(
            "metadata", rr.TextDocument(json.dumps(metadata, indent=2, default=str)), static=True
        )
        geometry = run.clip.geometry
        for i, t in enumerate(run.clip.indices):
            recording.set_time("dataset_frame", sequence=t)
            points, colors, lines, line_colors = render_tracks(run, i, trail_length=trail_length)
            keep = run.output.valid[0, i].numpy() & np.isfinite(
                run.output.tracks_world[0, i].numpy()
            ).all(-1)
            labels = [str(int(n)) for n in run.output.ids[0].numpy()[keep]]
            recording.log(
                "world/tracks/points",
                rr.Points3D(points, colors=colors, radii=0.009, labels=labels, show_labels=False),
            )
            recording.log(
                "world/tracks/trails", rr.LineStrips3D(lines, colors=line_colors[:, 0], radii=0.002)
            )
            cloud, rgb = build_point_cloud(
                run.clip.display_result(i),
                run.clip.images[0, i],
                stride=cloud_stride,
                align_mode="none",
            )
            recording.log("world/cloud", rr.Points3D(cloud, colors=rgb, radii=0.003))
            for v, name in enumerate(run.clip.camera_names):
                entity = f"world/cameras/{v}_{name.replace('/', '_')}"
                pose = geometry.extrinsics[0, i, v].numpy()
                recording.log(entity, rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]))
                image = run.clip.images[0, i, v].permute(1, 2, 0).numpy()
                h, w = image.shape[:2]
                intr = geometry.intrinsics[0, i, v].numpy().copy()
                intr[0] *= w
                intr[1] *= h
                recording.log(
                    entity + "/image",
                    rr.Pinhole(
                        image_from_camera=intr,
                        resolution=(w, h),
                        camera_xyz=rr.ViewCoordinates.RDF,
                        image_plane_distance=0.1,
                    ),
                )
                recording.log(
                    entity + "/image", rr.Image((image.clip(0, 1) * 255).astype(np.uint8))
                )
        recording.flush()
    finally:
        recording.disconnect()
    return path
