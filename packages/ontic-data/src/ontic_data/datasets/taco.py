"""TACO: 12 allocentric cameras at 30 FPS, per-camera mp4s, optional hand/object poses.

Raw calibration is ``[R|t]`` world-to-camera with pixel-space ``K``; output follows the
pipeline convention (c2w 4x4, normalised intrinsics). ``taco_info.csv`` is read with
pandas (extra ``tables``); frames come through :class:`~ontic_data.video.VideoReader`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .._np import normalize_intrinsics_np, w2c_to_c2w_np
from ..config import DatasetCfg, HorizonFn, Stage, StepFn
from ..temporal import SceneView, TemporalSceneDataset, resize_frames
from ..video import VideoBackend, VideoReader, has_video_backend
from ..view_sampler import ViewSampler


def _pandas():
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("TACO metadata needs pandas; install ontic-data[tables]") from e
    return pd


@dataclass(kw_only=True)
class TacoDatasetCfg(DatasetCfg):
    root: str = "/fast/mzhobro/taco_dataset_resized"

    image_shape: list[int] = field(default_factory=lambda: [376, 512])
    near: float = 0.1
    far: float = 1000.0
    fps: int = 30
    camera_ixs_allowed: list[int] | None = field(default_factory=lambda: list(range(12)))

    workspace_min: list[float] | None = field(default_factory=lambda: [-0.354, -0.994, 0.401])
    workspace_max: list[float] | None = field(default_factory=lambda: [0.662, 0.323, 1.409])

    require_complete: bool = True
    require_good_calib: bool = True  # keep calib_status == "good" only
    load_hand_poses: bool = True
    load_object_poses: bool = False
    video_backend: VideoBackend = "torchcodec"

    def build(
        self, stage: Stage, *, step_fn: StepFn | None = None, horizon_fn: HorizonFn | None = None
    ):
        view_sampler = self.build_view_sampler(stage)
        return DatasetTaco(self, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)


class DatasetTaco(TemporalSceneDataset):
    """One record (a ``taco_info.csv`` row) per sequence."""

    cfg: TacoDatasetCfg

    def __init__(
        self,
        cfg: TacoDatasetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        *,
        step_fn: StepFn | None = None,
        horizon_fn: HorizonFn | None = None,
    ) -> None:
        if not has_video_backend(cfg.video_backend):
            raise ImportError(
                f"TACO needs the {cfg.video_backend!r} video backend; install ontic-data[video]"
            )
        self.root = Path(cfg.root).resolve()
        super().__init__(cfg, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)

    def _discover(self) -> list[dict]:
        cfg = self.cfg
        meta = _pandas().read_csv(self.root / "taco_info.csv")
        if cfg.require_complete:
            meta = meta[meta["all_modalities_complete"] == True].reset_index(drop=True)  # noqa: E712
        meta = meta[meta["n_allocentric_cameras"] >= 12].reset_index(drop=True)
        if cfg.require_good_calib and "calib_status" in meta.columns:
            meta = meta[meta["calib_status"] == "good"].reset_index(drop=True)
        return meta.to_dict("records")

    def _n_frames_of(self, rec: dict) -> int:
        return int(rec["n_frames"])

    def _scene_name(self, rec: dict, t_start: int) -> str:
        return str(rec["sequence_id"])  # the sequence id alone keys the sample

    def _record_label(self, rec: dict) -> str:
        return str(rec["sequence_id"])

    def _open_scene(self, rec: dict) -> TacoSceneView:
        return TacoSceneView(self, rec)

    # ------------------------------------------------------------------ #
    # Calibration
    # ------------------------------------------------------------------ #

    @lru_cache(maxsize=32)
    def _load_calibration_json(self, calib_path: str) -> dict:
        with open(calib_path) as f:
            return json.load(f)

    def load_calibration(self, rec_ix: int) -> dict:
        """Raw calibration dict for a sequence (``{}`` when it has none)."""
        calib_path = self.records[rec_ix].get("alloc_camera_params_path", "")
        if not calib_path:
            return {}
        return self._load_calibration_json(str(self.root / calib_path))

    def _get_camera_params(self, rec: dict, cam_ids: list[str]) -> tuple[Tensor, Tensor, int, int]:
        """``(extrinsics (V,4,4) c2w, intrinsics (V,3,3) normalised, width, height)``."""
        calib = self._load_calibration_json(
            str(self.root / rec.get("alloc_camera_params_path", ""))
        )
        extrinsics, intrinsics = [], []
        orig_w, orig_h = 640, 480
        for cam_id in cam_ids:
            cam_data = calib.get(cam_id, {})
            K_raw = cam_data.get("K", None)
            K = (
                np.array(K_raw, dtype=np.float32).reshape(3, 3)
                if K_raw is not None
                else np.eye(3, dtype=np.float32)
            )
            img_size = cam_data.get("imgSize", None)
            if img_size is not None:
                orig_w, orig_h = int(img_size[0]), int(img_size[1])
            intrinsics.append(torch.from_numpy(normalize_intrinsics_np(K, orig_w, orig_h)))

            R_raw, T_raw = cam_data.get("R", None), cam_data.get("T", None)
            if R_raw is not None and T_raw is not None:
                R = np.array(R_raw, dtype=np.float32).reshape(3, 3)
                T = np.array(T_raw, dtype=np.float32).reshape(3)
                extrinsics.append(torch.from_numpy(w2c_to_c2w_np(R, T)))
            else:
                extrinsics.append(torch.eye(4, dtype=torch.float32))
        return torch.stack(extrinsics), torch.stack(intrinsics), orig_w, orig_h

    # ------------------------------------------------------------------ #
    # Poses
    # ------------------------------------------------------------------ #

    def _load_hand_poses(self, rec: dict, frame_indices: list[int]) -> dict[str, Tensor]:
        """Precomputed 3-D joints as ``(T, 1, 21, 4)`` per hand (presence constant 1)."""
        joints_path = self.root / "Hand_Poses_3D" / rec["sequence_id"] / "hand_joints.npy"
        if not joints_path.exists():
            return {}
        all_joints = np.load(str(joints_path))  # (N_frames, 2, 21, 3)
        valid = [min(i, len(all_joints) - 1) for i in frame_indices]
        joints = torch.from_numpy(all_joints[valid].copy()).float()
        joints = torch.cat([joints, torch.ones(*joints.shape[:-1], 1)], dim=-1)
        return {"left_hand": joints[:, 0:1], "right_hand": joints[:, 1:2]}

    def _load_object_poses(self, rec: dict, frame_indices: list[int]) -> dict[str, Tensor]:
        """Object 6DoF poses ``(T, 4, 4)`` object-to-world, keyed by file stem."""
        obj_dir = rec.get("object_poses_dir", "")
        if not obj_dir:
            return {}
        obj_path = self.root / obj_dir
        if not obj_path.exists():
            return {}
        result = {}
        for npy_file in obj_path.glob("*.npy"):
            poses = np.load(str(npy_file), mmap_mode="r")
            valid = [min(i, len(poses) - 1) for i in frame_indices]
            result[npy_file.stem] = torch.from_numpy(poses[valid].copy()).float()
        return result


class TacoSceneView(SceneView):
    """Static calibrated rig; pixels from per-camera mp4s opened per sample."""

    def __init__(self, ds: DatasetTaco, rec: dict) -> None:
        self.ds, self.rec = ds, rec
        cam_ids = str(rec["camera_ids"]).split(";")
        if ds.cfg.camera_ixs_allowed is not None:
            cam_ids = [cam_ids[i] for i in ds.cfg.camera_ixs_allowed]
        self.cam_names = cam_ids
        self._extr, self.intrinsics, _, _ = ds._get_camera_params(rec, cam_ids)
        self._readers: dict[str, VideoReader] = {}

    def extrinsics(self, frame_idx: int) -> Tensor:
        return self._extr

    def load_views(self, cam_ixs, frame_idx, out_hw, *, depth=False, side="target"):
        mr_dir = self.rec.get("marker_removed_dir", "")
        views = []
        for i in cam_ixs:
            path = str(self.ds.root / mr_dir / f"{self.cam_names[int(i)]}.mp4")
            vr = self._readers.get(path)
            if vr is None:
                vr = self._readers[path] = VideoReader(path, self.ds.cfg.video_backend)
            views.append(vr.get_frames([min(frame_idx, len(vr) - 1)])[0])  # (H, W, 3) uint8
        image = torch.stack(views).permute(0, 3, 1, 2).float() / 255.0
        return {"image": resize_frames(image, *out_hw, "bilinear")}

    def load_actions(self, frame_indices: list[int]) -> dict[str, Tensor] | None:
        cfg = self.ds.cfg
        actions: dict[str, Tensor] = {}
        if cfg.load_hand_poses:
            actions.update(self.ds._load_hand_poses(self.rec, frame_indices))
        if cfg.load_object_poses:
            actions.update(self.ds._load_object_poses(self.rec, frame_indices))
        return actions or None
