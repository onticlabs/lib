#!/usr/bin/env python3
"""Render a saved Franka Duo placement through a DEXTRIS recording's cameras.

Requires ontic-viz[robot], OpenCV and the robotics checkout. Uses EGL by default;
set MUJOCO_GL=osmesa if EGL is unavailable. Images retain the recorded lens distortion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

from ontic_data.datasets.dextris import parse_calibration
from ontic_viz.backbone_viewer.robot_model import DuoRobotModel, pose7_to_matrix


def render_overlays(recording: Path, alignment_path: Path, output: Path) -> None:
    alignment = json.loads(alignment_path.read_text())
    calibration_path = recording / "calibration_result.json"
    if hashlib.sha256(calibration_path.read_bytes()).hexdigest() != alignment["calibration_sha256"]:
        raise ValueError("The alignment belongs to a different camera calibration")
    cameras = parse_calibration(calibration_path)
    robot = DuoRobotModel()
    robot.set_qpos(alignment["qpos"])
    model, data = robot._model, robot._data
    base = pose7_to_matrix(alignment["base_pose"])
    height, width = 540, 960
    model.vis.global_.offwidth = width
    model.vis.global_.offheight = height
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    output.mkdir(parents=True, exist_ok=True)
    tiles = []
    with mujoco.Renderer(model, height, width) as renderer:
        for name, camera in sorted(cameras.items()):
            video = cv2.VideoCapture(str(recording / f"{recording.name}_{name}.mp4"))
            try:
                video.set(cv2.CAP_PROP_POS_FRAMES, alignment["frame_index"])
                ok, bgr = video.read()
            finally:
                video.release()
            if not ok:
                raise ValueError(f"Could not decode {name}, frame {alignment['frame_index']}")
            bgr = cv2.resize(bgr, (width, height))
            intrinsics = camera["K_px"].copy()
            intrinsics[0] *= width / camera["W"]
            intrinsics[1] *= height / camera["H"]
            rotation = camera["R"] @ base[:3, :3]
            translation = camera["R"] @ base[:3, 3] + camera["t"]
            renderer.update_scene(data, scene_option=options)
            renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
            for geom in renderer.scene.geoms[: renderer.scene.ngeom]:
                # Match the viewer's mesh-only overlay, without the synthetic pillar.
                if model.geom_type[geom.objid] != mujoco.mjtGeom.mjGEOM_MESH:
                    geom.rgba[3] = 0
                geom.emission = 0.5
            near = 0.01
            for view in renderer.scene.camera:
                view.pos[:] = -rotation.T @ translation
                view.forward[:] = rotation.T @ [0, 0, 1]
                view.up[:] = rotation.T @ [0, -1, 0]
                view.frustum_near, view.frustum_far = near, 10.0
                view.frustum_top = intrinsics[1, 2] / intrinsics[1, 1] * near
                view.frustum_bottom = -(height - intrinsics[1, 2]) / intrinsics[1, 1] * near
                view.frustum_center = (width / 2 - intrinsics[0, 2]) / intrinsics[0, 0] * near
                view.frustum_width = width / (2 * intrinsics[0, 0]) * near
            rgb = renderer.render().copy()
            renderer.enable_segmentation_rendering()
            segmentation = renderer.render()
            mesh_ids = [geom_id for geom_id, _ in robot._drawn_geoms]
            mask = np.isin(segmentation[:, :, 0], mesh_ids).astype(np.uint8)
            renderer.disable_segmentation_rendering()
            yy, xx = np.mgrid[:height, :width]
            pixels = np.stack([xx, yy], -1).astype(np.float32)
            mapping = cv2.undistortPoints(
                pixels.reshape(-1, 1, 2), intrinsics, camera["dist"], P=intrinsics
            ).reshape(height, width, 2)
            mesh = cv2.remap(rgb[:, :, ::-1], mapping, None, cv2.INTER_LINEAR)
            mask = cv2.remap(mask, mapping, None, cv2.INTER_NEAREST)
            overlay = bgr.copy()
            selected = mask > 0
            overlay[selected] = (bgr[selected] * 0.55 + mesh[selected] * 0.45).astype(np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (255, 210, 30), 1)
            cv2.rectangle(overlay, (0, height - 30), (width, height), (30, 30, 30), -1)
            excluded = alignment.get("provenance", {}).get("excluded_cameras", {})
            note = " | excluded from fit" if name in excluded else ""
            cv2.putText(
                overlay,
                f"{name} | frame {alignment['frame_index']} | estimated Franka Duo{note}",
                (12, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.imwrite(str(output / f"{name}.jpg"), overlay)
            tiles.append(overlay)
    grid = np.concatenate([np.concatenate(tiles[i : i + 4], axis=1) for i in range(0, 8, 4)])
    cv2.imwrite(str(output / "all_cameras.jpg"), grid)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("alignment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_overlays(args.recording, args.alignment, args.output)


if __name__ == "__main__":
    main()
