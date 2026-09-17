"""DEXTRIS: 8 synchronised cameras (``P00..P07``) at 1080x1920 / 60 FPS, short clips.

On-disk layout::

    <root>/<task>/<task>_<id>/
        calibration_result.json           # cameras{cam: camera_matrix (px), R, t (w2c),
                                          #         image_size [H, W], dist_coeffs}
        <task>_<id>_hand_tracking.json    # per-frame triangulated world-frame keypoints
        <task>_<id>_P{00..07}.mp4

Actions are ``left_hand`` / ``right_hand`` ``(T, 1, 21, 4)`` world-frame keypoints with a
per-frame presence bit (missing frames zero-filled). A ``dextris_info.csv`` is read with
pandas when present (extra ``tables``); otherwise samples are discovered on disk.
``RobotDextrisDatasetCfg`` uses the same calibration and videos in a flat
``<root>/<sample>/`` layout, without hand annotations or a train/validation split.
"""

from __future__ import annotations

import csv
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .._np import normalize_intrinsics_np, w2c_to_c2w_np
from ..config import DatasetCfg, HorizonFn, Stage, StepFn
from ..temporal import SceneView, TemporalSceneDataset, resize_frames
from ..video import VideoBackend, VideoReader, has_video_backend
from ..view_sampler import ViewSampler

CAMERA_NAMES = [f"P{i:02d}" for i in range(8)]
META_CSV_NAME = "dextris_info.csv"


@dataclass(kw_only=True)
class DextrisDatasetCfg(DatasetCfg):
    root: str = "/fast/mzhobro/dextris_dataset"

    image_shape: list[int] = field(default_factory=lambda: [540, 960])
    near: float = 0.1
    far: float = 5.0
    fps: int = 60
    camera_ixs_allowed: list[int] | None = field(default_factory=lambda: list(range(8)))

    workspace_min: list[float] | None = field(default_factory=lambda: [-0.95, -0.73, -0.239])
    workspace_max: list[float] | None = field(default_factory=lambda: [0.5, 0.69, 0.544])

    tasks: list[str] = field(default_factory=lambda: ["pickUp"])
    flat_layout: bool = False
    split_mode: Literal["scene", "none"] = "scene"
    load_hand_poses: bool = True
    require_both_hands: bool = False
    video_backend: VideoBackend = "torchcodec"

    def build(
        self, stage: Stage, *, step_fn: StepFn | None = None, horizon_fn: HorizonFn | None = None
    ):
        view_sampler = self.build_view_sampler(stage)
        return DatasetDextris(self, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)


@dataclass(kw_only=True)
class RobotDextrisDatasetCfg(DextrisDatasetCfg):
    root: str = "/mnt/fast/mzhobro/trailer-demo"
    tasks: list[str] = field(default_factory=list)
    flat_layout: bool = True
    split_mode: Literal["scene", "none"] = "none"
    load_hand_poses: bool = False
    video_backend: VideoBackend = "opencv"
    workspace_min: list[float] | None = None
    workspace_max: list[float] | None = None


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #


def parse_calibration(calib_path: str | Path) -> dict[str, dict]:
    """``calibration_result.json`` -> ``{cam_id: {K_px, R, t, dist, H, W}}`` (raw units)."""
    with open(calib_path) as f:
        calib = json.load(f)
    out: dict[str, dict] = {}
    for cam_id, cd in calib["cameras"].items():
        out[cam_id] = {
            "K_px": np.array(cd["camera_matrix"], dtype=np.float64),
            "R": np.array(cd["R"], dtype=np.float64),
            "t": np.array(cd["t"], dtype=np.float64),
            "dist": np.array(cd["dist_coeffs"], dtype=np.float64).ravel(),
            "H": int(cd["image_size"][0]),
            "W": int(cd["image_size"][1]),
        }
    return out


def calib_to_pipeline_arrays(
    calib: dict[str, dict], cam_ids: list[str]
) -> tuple[Tensor, Tensor, int, int]:
    """``(extrinsics (V,4,4) c2w, intrinsics (V,3,3) normalised, H, W)``."""
    Hs = {calib[c]["H"] for c in cam_ids}
    Ws = {calib[c]["W"] for c in cam_ids}
    if len(Hs) != 1 or len(Ws) != 1:
        raise ValueError(f"non-uniform image sizes across cameras: H={Hs} W={Ws}")
    H, W = Hs.pop(), Ws.pop()
    extr = [torch.from_numpy(w2c_to_c2w_np(calib[c]["R"], calib[c]["t"])) for c in cam_ids]
    intr = [torch.from_numpy(normalize_intrinsics_np(calib[c]["K_px"], W, H)) for c in cam_ids]
    return torch.stack(extr), torch.stack(intr), H, W


def build_hand_action_tensor(
    kp: np.ndarray, present: np.ndarray, frame_indices: list[int]
) -> Tensor:
    """``(F, 21, 3)`` keypoints + ``(F,)`` presence -> ``(T, 1, 21, 4)`` (NaN -> 0)."""
    T_full = kp.shape[0]
    idx = [min(i, T_full - 1) for i in frame_indices]
    kp_sub = np.where(np.isnan(kp[idx]), 0.0, kp[idx]).astype(np.float32)
    present_sub = present[idx].astype(np.float32)
    presence = np.broadcast_to(present_sub[:, None], kp_sub.shape[:-1])
    out = np.concatenate([kp_sub, presence[..., None]], axis=-1)
    return torch.from_numpy(out[:, None].copy()).float()


def parse_hand_tracking(ht_path: str | Path) -> dict:
    """Per-sample hand-tracking JSON -> ``{cameras, n_frames, left_kp (T,21,3) NaN where
    missing, right_kp, left_present (T,), right_present, left_reproj_err, right_reproj_err}``."""
    with open(ht_path) as f:
        ht = json.load(f)

    cams = list(ht["cameras_used"])
    timesteps = list(ht["timesteps"])
    T = len(timesteps)

    left_kp = np.full((T, 21, 3), np.nan, dtype=np.float64)
    right_kp = np.full((T, 21, 3), np.nan, dtype=np.float64)
    left_present = np.zeros(T, dtype=bool)
    right_present = np.zeros(T, dtype=bool)
    left_err = np.full(T, np.nan, dtype=np.float64)
    right_err = np.full(T, np.nan, dtype=np.float64)

    tri = ht["triangulated"]
    for i, ts in enumerate(timesteps):
        entry = tri.get(str(ts))
        if entry is None:
            continue
        if entry.get("left") is not None:
            left_kp[i] = np.asarray(entry["left"]["keypoints_3d"], dtype=np.float64)
            left_present[i] = True
            left_err[i] = float(entry["left"].get("reprojection_error_px", np.nan))
        if entry.get("right") is not None:
            right_kp[i] = np.asarray(entry["right"]["keypoints_3d"], dtype=np.float64)
            right_present[i] = True
            right_err[i] = float(entry["right"].get("reprojection_error_px", np.nan))

    return {
        "cameras": cams,
        "n_frames": T,
        "left_kp": left_kp,
        "right_kp": right_kp,
        "left_present": left_present,
        "right_present": right_present,
        "left_reproj_err": left_err,
        "right_reproj_err": right_err,
    }


def discover_samples(
    root: str | Path,
    tasks: list[str] | None = None,
    *,
    flat_layout: bool = False,
    require_hand_tracking: bool = True,
) -> list[Path]:
    """Sample directories with calibration, all 8 videos and optional hand tracking."""
    root = Path(root)
    out: list[Path] = []
    if flat_layout:
        task_dirs = [root]
    else:
        task_dirs = (
            [root / t for t in tasks] if tasks else sorted(d for d in root.iterdir() if d.is_dir())
        )
    for td in task_dirs:
        if not td.is_dir():
            continue
        for sample in sorted(td.iterdir()):
            if not sample.is_dir():
                continue
            sid = sample.name
            if flat_layout and tasks and sid.rsplit("_", 1)[0] not in tasks:
                continue
            calib_ok = (sample / "calibration_result.json").exists()
            ht_ok = not require_hand_tracking or (sample / f"{sid}_hand_tracking.json").exists()
            videos_ok = all((sample / f"{sid}_{cam}.mp4").exists() for cam in CAMERA_NAMES)
            if calib_ok and ht_ok and videos_ok:
                out.append(sample)
    return out


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def write_metadata(cfg: DextrisDatasetCfg, path: str | Path) -> list[dict]:
    """Index complete recordings using ffprobe headers, without decoding videos.

    The resulting dextris_info.csv is reused by the existing loader. Rebuild it
    explicitly after adding or replacing recordings. Unreadable/incomplete scenes
    are reported and excluded; the original index survives a failed rebuild.
    """
    root = Path(cfg.root)
    rows = []
    samples = discover_samples(
        root, cfg.tasks, flat_layout=cfg.flat_layout, require_hand_tracking=cfg.load_hand_poses
    )
    for sample in samples:
        try:
            counts = []
            for cam in CAMERA_NAMES:
                video = sample / f"{sample.name}_{cam}.mp4"
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_frames",
                        "-of",
                        "json",
                        str(video),
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=30,
                )
                counts.append(int(json.loads(result.stdout)["streams"][0]["nb_frames"]))
            if len(set(counts)) != 1 or counts[0] <= 0:
                raise ValueError(f"camera frame counts disagree or are empty: {counts}")
        except (subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
            print(f"Skipping {sample.name}: {exc}", flush=True)
            continue
        rows.append(
            {
                "task": sample.name.rsplit("_", 1)[0] if cfg.flat_layout else sample.parent.name,
                "sample_id": sample.name,
                "rel_dir": str(sample.relative_to(root)),
                "n_frames": counts[0],
            }
        )
        print(f"Indexed {sample.name}: {counts[0]} frames, {len(counts)} cameras", flush=True)
    if not rows:
        raise ValueError(f"No complete recordings found in {root}; index was not written")
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        try:
            writer = csv.DictWriter(f, fieldnames=["task", "sample_id", "rel_dir", "n_frames"])
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    return rows


class DatasetDextris(TemporalSceneDataset):
    """One record per sample directory."""

    cfg: DextrisDatasetCfg

    def __init__(
        self,
        cfg: DextrisDatasetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        *,
        step_fn: StepFn | None = None,
        horizon_fn: HorizonFn | None = None,
    ) -> None:
        if not has_video_backend(cfg.video_backend):
            extra = "video" if cfg.video_backend == "torchcodec" else cfg.video_backend
            raise ImportError(
                f"DEXTRIS needs the {cfg.video_backend!r} video backend; install ontic-data[{extra}]"
            )
        self.root = Path(cfg.root).resolve()
        super().__init__(cfg, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)

    def _discover(self) -> list[dict]:
        cfg = self.cfg
        meta_csv = self.root / META_CSV_NAME
        if meta_csv.exists():
            try:
                import pandas as pd
            except ImportError as e:
                raise ImportError(
                    f"{META_CSV_NAME} needs pandas; install ontic-data[tables]"
                ) from e
            meta = pd.read_csv(meta_csv)
            if cfg.tasks:
                meta = meta[meta["task"].isin(cfg.tasks)].reset_index(drop=True)
            return meta.to_dict("records")
        sample_dirs = discover_samples(
            self.root,
            cfg.tasks,
            flat_layout=cfg.flat_layout,
            require_hand_tracking=cfg.load_hand_poses,
        )
        if not sample_dirs:
            raise FileNotFoundError(f"no DEXTRIS samples in {self.root} for tasks={cfg.tasks}")
        return [
            {
                "task": sd.name.rsplit("_", 1)[0] if cfg.flat_layout else sd.parent.name,
                "sample_id": sd.name,
                "rel_dir": str(sd.relative_to(self.root)),
            }
            for sd in sample_dirs
        ]

    def _n_frames_of(self, rec: dict) -> int:
        n = rec.get("n_frames")
        if n is None or (isinstance(n, float) and np.isnan(n)):
            n = rec["n_frames"] = self._probe_n_frames(rec)
        return int(n)

    def _scene_name(self, rec: dict, t_start: int) -> str:
        sp = self.cfg.speedup
        end = t_start + self.n_full_steps * sp
        return f"{self._record_label(rec)}_{t_start}:{end}:{sp}"

    def _record_label(self, rec: dict) -> str:
        if self.cfg.flat_layout:
            return rec["sample_id"]
        return f"{rec['task']}/{rec['sample_id']}"

    def _open_scene(self, rec: dict) -> DextrisSceneView:
        return DextrisSceneView(self, rec)

    def _sample_dir(self, rec: dict) -> Path:
        rel = rec.get("rel_dir")
        if isinstance(rel, str) and rel:
            return self.root / rel
        return self.root / rec["task"] / rec["sample_id"]

    @lru_cache(maxsize=64)
    def _calib_for(self, sample_dir: str) -> dict:
        return parse_calibration(Path(sample_dir) / "calibration_result.json")

    def _probe_n_frames(self, rec: dict) -> int:
        sd = self._sample_dir(rec)
        for cam in CAMERA_NAMES:
            vp = sd / f"{rec['sample_id']}_{cam}.mp4"
            if vp.exists():
                return len(VideoReader(vp, self.cfg.video_backend))
        return 0

    @lru_cache(maxsize=64)
    def _hand_for(self, sample_dir: str, sample_id: str) -> dict:
        return parse_hand_tracking(Path(sample_dir) / f"{sample_id}_hand_tracking.json")

    def _load_hand_poses(
        self, sample_dir: Path, sample_id: str, frame_indices: list[int]
    ) -> dict[str, Tensor]:
        ht = self._hand_for(str(sample_dir), sample_id)
        return {
            "left_hand": build_hand_action_tensor(ht["left_kp"], ht["left_present"], frame_indices),
            "right_hand": build_hand_action_tensor(
                ht["right_kp"], ht["right_present"], frame_indices
            ),
        }


class DextrisSceneView(SceneView):
    """Static calibrated 8-camera rig; pixels from mp4s opened per sample."""

    def __init__(self, ds: DatasetDextris, rec: dict) -> None:
        self.ds, self.rec = ds, rec
        self.sample_dir = ds._sample_dir(rec)
        self.sid = rec["sample_id"]
        calib = ds._calib_for(str(self.sample_dir))
        cam_ids = [c for c in CAMERA_NAMES if c in calib]
        if ds.cfg.camera_ixs_allowed is not None:
            cam_ids = [cam_ids[i] for i in ds.cfg.camera_ixs_allowed if i < len(cam_ids)]
        self.cam_names = cam_ids
        self._extr, self.intrinsics, _, _ = calib_to_pipeline_arrays(calib, cam_ids)
        self._readers: dict[str, VideoReader] = {}

    def extrinsics(self, frame_idx: int) -> Tensor:
        return self._extr

    def load_views(self, cam_ixs, frame_idx, out_hw, *, depth=False, side="target"):
        views = []
        for i in cam_ixs:
            path = str(self.sample_dir / f"{self.sid}_{self.cam_names[int(i)]}.mp4")
            vr = self._readers.get(path)
            if vr is None:
                vr = self._readers[path] = VideoReader(path, self.ds.cfg.video_backend)
            views.append(vr.get_frames([min(frame_idx, len(vr) - 1)])[0])
        image = torch.stack(views).permute(0, 3, 1, 2).float() / 255.0
        return {"image": resize_frames(image, *out_hw, "bilinear")}

    def load_actions(self, frame_indices: list[int]) -> dict[str, Tensor] | None:
        if not self.ds.cfg.load_hand_poses:
            return None
        return self.ds._load_hand_poses(self.sample_dir, self.sid, frame_indices)
