"""Record real RGB-D/backbone tracking comparisons using local checkpoints.

Run with a compatible GPU environment and Ontic packages on PYTHONPATH. The
upstream source and weights are explicit; shared environments are not modified.
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch

from ontic_viz.backbone_viewer.data_source import DEFAULT_ROOTS, build_source
from ontic_viz.backbone_viewer.recording import save_recording
from ontic_viz.backbone_viewer.runner import BackboneRunner
from ontic_viz.backbone_viewer.render import PointFilters
from ontic_viz.backbone_viewer.tracking import (
    SENSOR,
    SENSOR_SCALE,
    ClipSpec,
    TrackerSettings,
    TrackingRunner,
    export_tracks,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="hocap")
    p.add_argument("--root")
    p.add_argument("--trajectory", type=int, default=0)
    p.add_argument("--start", type=int, default=40)
    p.add_argument("--frames", type=int, default=12)
    p.add_argument("--step", type=int, default=3)
    p.add_argument("--views", type=int, nargs="+", default=[0, 1, 3])
    p.add_argument("--long-side", type=int, default=256)
    p.add_argument("--queries", type=int, default=128)
    p.add_argument("--backbone", default="sensor")
    p.add_argument("--backbone-checkpoint")
    p.add_argument("--tracker", choices=["mvtracker", "cotracker3"], default="mvtracker")
    p.add_argument("--tracker-checkpoint", required=True)
    p.add_argument("--tracker-repo", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--display-presets", type=Path, help="Use the dataset's viewer filters for query selection"
    )
    args = p.parse_args()
    torch.set_num_threads(4)
    source = build_source(args.dataset, args.root or DEFAULT_ROOTS[args.dataset])
    frame = source.get_frame(args.trajectory, args.start, with_depth=True)
    cameras = tuple(frame.cam_names[i] for i in args.views)
    backbone = BackboneRunner(args.device)
    if args.backbone != "sensor":
        if not args.backbone_checkpoint:
            p.error("--backbone-checkpoint is required for predicted depth")
        backbone.configure(
            args.backbone, checkpoint_path=args.backbone_checkpoint, allow_download=False
        )
    runner = TrackingRunner(args.device)
    point_filters = None
    if args.display_presets:
        cfg = json.loads(args.display_presets.read_text())[f"{args.dataset}_default"]
        point_filters = PointFilters(
            stride=cfg["stride"],
            drop_conf_pct=cfg["drop_conf_pct"],
            crop=(tuple(cfg["ws_min"]), tuple(cfg["ws_max"])) if cfg["crop_enabled"] else None,
            voxel_size=cfg["voxel_size"],
            sfc_stride=cfg["sfc_stride"],
            max_points=cfg["fps_max_points"],
        )
    started = time.perf_counter()
    clip = runner.prepare(
        source,
        args.trajectory,
        ClipSpec(args.start, args.frames, args.step, args.long_side),
        cameras,
        geometry_source=SENSOR if args.backbone == "sensor" else SENSOR_SCALE,
        backbone=args.backbone,
        backbone_runner=backbone,
        progress=lambda message: print(message, flush=True),
    )
    crop = None
    if frame.hands is not None:
        crop = (frame.hands.amin((0, 1)) - 0.15, frame.hands.amax((0, 1)) + 0.15)
    result = runner.run(
        clip,
        TrackerSettings(
            name=args.tracker,
            repo_path=args.tracker_repo,
            checkpoint_path=args.tracker_checkpoint,
            long_side=args.long_side,
        ),
        source=source,
        count=args.queries,
        crop=crop,
        point_filters=point_filters,
        progress=lambda message: print(message, flush=True),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    name = f"{args.dataset}-{args.backbone}-{args.tracker}"
    save_recording(result, args.output / f"{name}.viewer.npz")
    (args.output / f"{name}.tracks.npz").write_bytes(export_tracks(result))
    displacement = (result.output.tracks_world[0, -1] - result.output.tracks_world[0, 0]).norm(
        dim=-1
    )
    endpoint_valid = result.output.valid[0, 0] & result.output.valid[0, -1]
    metadata = {
        "dataset": args.dataset,
        "trajectory": source.list_trajectories()[args.trajectory],
        "frames": clip.indices,
        "cameras": clip.camera_names,
        "backbone": args.backbone,
        "backbone_checkpoint": args.backbone_checkpoint,
        "geometry": clip.geometry.provenance,
        "tracker_metadata": result.output.metadata,
        "torch": torch.__version__,
        "device": args.device,
        "queries": result.output.ids.shape[-1],
        "valid_fraction": float(result.output.valid.float().mean()),
        "mean_visibility": float(result.output.visibility.mean()),
        "median_displacement_m": float(displacement[endpoint_valid].median())
        if endpoint_valid.any()
        else None,
        "tracking_seconds": result.elapsed,
        "total_seconds": time.perf_counter() - started,
        "note": "Real pretrained inference; these are diagnostics, not tracking accuracy metrics.",
    }
    if all(f.depth is not None for f in clip.frames):
        sensor = torch.stack(
            [f.depth[[f.cam_names.index(c) for c in cameras], 0] for f in clip.frames]
        )[None]
        valid = (sensor > 0) & torch.isfinite(sensor) & clip.geometry.valid_depth
        metadata["calibrated_depth_mae_m"] = float(
            (clip.geometry.depth - sensor).abs()[valid].mean()
        )
    (args.output / f"{name}.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    print(json.dumps(metadata, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
