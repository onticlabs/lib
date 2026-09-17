"""PhysInOne: synthetic physics scenes from static ``CineCamera_<N>`` plus one moving camera.

On-disk layout::

    <root>/<split>/<scene>/
        <scene>.json                     # trajectory: {sequence_info, actors{...}}
        blender_CineCamera_<N>.json      # per-camera metadata (N non-contiguous)
        blender_CineCamera_Moving.json
        CineCamera_<N>/rgb/0000.jpg  depth/0000.npz ("depth", metres)  seg/0000.npz ("seg")

Camera JSONs carry ``camera_angle_x`` (horizontal FOV), ``img_h``/``img_w`` and per-frame
``transform_matrix`` c2w in the Blender/OpenGL frame; output is OpenCV c2w
(``c2w @ diag(1,-1,-1,1)``) with normalised pinhole intrinsics. Optional actions are per
dynamic object world translations ``(T, 1, 1, 4)`` (Unreal cm rescaled by
``object_traj_scale``). PIL and pandas (for ``physinone_info.csv``) are lazy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from ..config import DatasetCfg, HorizonFn, Stage, StepFn
from ..temporal import SceneView, TemporalSceneDataset, resize_frames
from ..view_sampler import ViewSampler

#: Blender (OpenGL-style) camera -> OpenCV camera: flip the Y and Z basis axes.
BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float64)

META_CSV_NAME = "physinone_info.csv"


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def blender_c2w_to_opencv(mat: np.ndarray) -> np.ndarray:
    """Blender c2w ``(4, 4)`` -> OpenCV c2w float32 (world frame untouched)."""
    return (np.asarray(mat, dtype=np.float64) @ BLENDER_TO_OPENCV).astype(np.float32)


def intrinsics_from_fov(camera_angle_x: float, width: int, height: int) -> np.ndarray:
    """Normalised pinhole ``K`` from a horizontal FOV (radians), square pixels, centred."""
    fx_px = (width / 2.0) / np.tan(camera_angle_x / 2.0)
    K = np.eye(3, dtype=np.float32)
    K[0, 0] = fx_px / width
    K[1, 1] = fx_px / height
    K[0, 2] = 0.5
    K[1, 2] = 0.5
    return K


# --------------------------------------------------------------------------- #
# Scene discovery + parsing
# --------------------------------------------------------------------------- #


def scene_camera_names(scene_dir: str | Path, include_moving: bool = True) -> list[str]:
    """Sorted camera list: numeric ``CineCamera_<N>`` (by int) then ``CineCamera_Moving``."""
    scene_dir = Path(scene_dir)
    numeric: list[int] = []
    has_moving = False
    for p in scene_dir.glob("blender_CineCamera_*.json"):
        tag = p.stem[len("blender_CineCamera_") :]
        if tag == "Moving":
            has_moving = True
        elif tag.isdigit():
            numeric.append(int(tag))
    names = [f"CineCamera_{n}" for n in sorted(numeric)]
    if has_moving and include_moving:
        names.append("CineCamera_Moving")
    return names


def discover_scenes(root: str | Path) -> list[dict]:
    """``[{split, scene, rel_dir}]`` for every dir holding a ``blender_CineCamera_*.json``."""
    root = Path(root)
    out: list[dict] = []
    seen: set[Path] = set()
    for cam_json in sorted(root.rglob("blender_CineCamera_*.json")):
        scene_dir = cam_json.parent
        if scene_dir in seen:
            continue
        seen.add(scene_dir)
        rel = scene_dir.relative_to(root)
        split = rel.parts[0] if len(rel.parts) > 1 else ""
        out.append({"split": split, "scene": scene_dir.name, "rel_dir": str(rel)})
    return out


@lru_cache(maxsize=128)
def _parse_camera_json(path: str) -> dict:
    """``{c2w: (F,4,4) float32 OpenCV, K: (3,3) normalised, H, W, n_frames}``."""
    with open(path) as f:
        j = json.load(f)
    frames = j["frames"]
    c2w = np.stack([blender_c2w_to_opencv(fr["transform_matrix"]) for fr in frames], 0)
    K = intrinsics_from_fov(float(j["camera_angle_x"]), int(j["img_w"]), int(j["img_h"]))
    return {
        "c2w": c2w,
        "K": K,
        "H": int(j["img_h"]),
        "W": int(j["img_w"]),
        "n_frames": int(j.get("total_frames", len(frames))),
    }


def parse_object_trajectories(traj_path: str | Path) -> dict[str, np.ndarray]:
    """``<scene>.json`` -> ``{name: (F, 3)}`` raw-cm world translations, NaN-filled gaps.

    Solid actors contribute one trajectory keyed by actor name; interactable actors one
    per component keyed ``<actor>/<component>``.
    """
    with open(traj_path) as f:
        tj = json.load(f)
    n_frames = int(tj.get("sequence_info", {}).get("total_frames", 0))
    out: dict[str, np.ndarray] = {}

    def _series(transform_data: list) -> np.ndarray | None:
        if not transform_data:
            return None
        F_ = n_frames if n_frames > 0 else (max(int(d["frame"]) for d in transform_data) + 1)
        arr = np.full((F_, 3), np.nan, dtype=np.float64)
        for d in transform_data:
            fi = int(d["frame"])
            if 0 <= fi < F_:
                loc = d["transform"]["location"]
                arr[fi] = [loc["x"], loc["y"], loc["z"]]
        return arr

    for name, actor in tj.get("actors", {}).items():
        s = _series(actor.get("transform_data", []))
        if s is not None:
            out[name] = s
        for cname, comp in actor.get("components", {}).items():
            cs = _series(comp.get("transform_data", []))
            if cs is not None:
                out[f"{name}/{cname}"] = cs
    return out


def build_object_action_tensor(
    series: np.ndarray, frame_indices: list[int], scale: float
) -> Tensor:
    """``(F, 3)`` series -> ``(T, 1, 1, 4)`` xyz * scale + presence (0 where NaN)."""
    F_ = series.shape[0]
    idx = [min(i, F_ - 1) for i in frame_indices]
    sub = series[idx]
    present = (~np.isnan(sub).any(axis=-1)).astype(np.float32)
    xyz = np.where(np.isnan(sub), 0.0, sub).astype(np.float32) * scale
    out = np.concatenate([xyz, present[:, None]], axis=-1)
    return torch.from_numpy(out[:, None, None].copy()).float()


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(kw_only=True)
class PhysInOneDatasetCfg(DatasetCfg):
    root: str = "/mnt/fast/mzhobro/physinone_dataset"

    image_shape: list[int] = field(default_factory=lambda: [560, 560])  # native 1120x1120
    near: float = 0.1
    far: float = 20.0
    fps: int = 30
    camera_ixs_allowed: list[int] | None = None  # positional into the sorted camera list

    # "scene" = deterministic 90/10 train/val scene split; "none" = every scene in both.
    split_mode: Literal["scene", "none"] = "scene"

    include_moving_camera: bool = True
    load_object_trajectories: bool = False
    object_traj_scale: float = 0.01  # Unreal cm -> m
    with_depth: bool = False
    with_seg: bool = False
    # ``state_mask`` from the seg map: "foreground" = seg > 0, "dynamic" = seg >= 128;
    # ``mask_views`` = "both" (context + target) or "target".
    with_mask: bool = False
    mask_mode: Literal["foreground", "dynamic"] = "foreground"
    mask_views: Literal["both", "target"] = "both"

    workspace_min: list[float] | None = None
    workspace_max: list[float] | None = None

    def build(
        self, stage: Stage, *, step_fn: StepFn | None = None, horizon_fn: HorizonFn | None = None
    ):
        view_sampler = self.build_view_sampler(stage)
        return DatasetPhysInOne(self, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class DatasetPhysInOne(TemporalSceneDataset):
    """One record per scene."""

    cfg: PhysInOneDatasetCfg
    DEPTH_IS_METRIC = True

    def __init__(
        self,
        cfg: PhysInOneDatasetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        *,
        step_fn: StepFn | None = None,
        horizon_fn: HorizonFn | None = None,
    ) -> None:
        self.root = Path(cfg.root).resolve()
        super().__init__(cfg, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)

    def _discover(self) -> list[dict]:
        meta_csv = self.root / META_CSV_NAME
        if meta_csv.exists():
            try:
                import pandas as pd
            except ImportError as e:
                raise ImportError(
                    f"{META_CSV_NAME} needs pandas; install ontic-data[tables]"
                ) from e
            return pd.read_csv(meta_csv).to_dict("records")
        rows = discover_scenes(self.root)
        if not rows:
            raise FileNotFoundError(f"no PhysInOne scenes under {self.root}")
        return rows

    def _n_frames_of(self, rec: dict) -> int:
        n = rec.get("n_frames")
        if n is None or (isinstance(n, float) and np.isnan(n)):
            n = rec["n_frames"] = self._probe_n_frames(rec)
        return int(n)

    def _scene_name(self, rec: dict, t_start: int) -> str:
        sp = self.cfg.speedup
        end = t_start + self.n_full_steps * sp
        return f"{rec.get('split', '')}/{rec['scene']}_{t_start}:{end}:{sp}"

    def _record_label(self, rec: dict) -> str:
        return f"{rec.get('split', '')}/{rec['scene']}".lstrip("/")

    def _open_scene(self, rec: dict) -> PhysInOneSceneView:
        return PhysInOneSceneView(self, rec)

    def _scene_dir(self, rec: dict) -> Path:
        rel = rec.get("rel_dir")
        if isinstance(rel, str) and rel:
            return self.root / rel
        return self.root / rec["scene"]

    def _camera_names(self, scene_dir: Path) -> list[str]:
        names = scene_camera_names(scene_dir, self.cfg.include_moving_camera)
        if self.cfg.camera_ixs_allowed is not None:
            names = [names[i] for i in self.cfg.camera_ixs_allowed if i < len(names)]
        return names

    def scene_cameras(self, rec_ix: int) -> list[str]:
        """Sorted camera names for a scene (after ``camera_ixs_allowed`` filtering)."""
        return self._camera_names(self._scene_dir(self.records[rec_ix]))

    def _probe_n_frames(self, rec: dict) -> int:
        scene_dir = self._scene_dir(rec)
        names = scene_camera_names(scene_dir, self.cfg.include_moving_camera)
        if not names:
            return 0
        return _parse_camera_json(str(scene_dir / f"blender_{names[0]}.json"))["n_frames"]


def _open_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("PhysInOne frames need pillow; install ontic-data[images]") from e
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB")).copy()


class PhysInOneSceneView(SceneView):
    """Cameras from the blender JSONs (per-frame c2w), pixels from jpg/npz files."""

    def __init__(self, ds: DatasetPhysInOne, rec: dict) -> None:
        self.ds, self.rec = ds, rec
        self.scene_dir = ds._scene_dir(rec)
        self.cam_names = ds._camera_names(self.scene_dir)
        c2w, intr = [], []
        for name in self.cam_names:
            info = _parse_camera_json(str(self.scene_dir / f"blender_{name}.json"))
            c2w.append(info["c2w"])
            intr.append(torch.from_numpy(info["K"]))
        self.cam_c2w = np.stack(c2w, 0)  # (V, F, 4, 4)
        self.intrinsics = torch.stack(intr, 0)
        self._n_avail: dict[Path, int] = {}

    def extrinsics(self, frame_idx: int) -> Tensor:
        fi = min(frame_idx, self.cam_c2w.shape[1] - 1)
        return torch.from_numpy(self.cam_c2w[:, fi]).float()

    def _clamped(self, d: Path, frame_idx: int, pattern: str) -> int:
        n = self._n_avail.get(d)
        if n is None:
            n = self._n_avail[d] = len(list(d.glob(pattern)))
        return min(frame_idx, n - 1) if n else frame_idx

    def _rgb(self, names: list[str], frame_idx: int, out_hw) -> Tensor:
        views = []
        for name in names:
            rgb_dir = self.scene_dir / name / "rgb"
            fi = self._clamped(rgb_dir, frame_idx, "*.jpg")
            views.append(torch.from_numpy(_open_rgb(rgb_dir / f"{fi:04d}.jpg")))
        image = torch.stack(views, 0).permute(0, 3, 1, 2).float() / 255.0
        return resize_frames(image, *out_hw, "bilinear")

    def _npz(self, names: list[str], frame_idx: int, sub: str, key: str, out_hw) -> Tensor:
        """depth/seg npz -> ``(V, 1, H, W)`` float. Depth is clamped to ``cfg.far`` because
        sky is stored as the float16 sentinel 65504."""
        clamp_far = sub == "depth"
        views = []
        for name in names:
            d = self.scene_dir / name / sub
            fi = self._clamped(d, frame_idx, "*.npz")
            arr = np.load(d / f"{fi:04d}.npz")[key].astype(np.float32)
            if clamp_far:
                arr = np.clip(arr, 0.0, float(self.ds.cfg.far))
            views.append(torch.from_numpy(arr))
        return resize_frames(torch.stack(views, 0).unsqueeze(1), *out_hw, "nearest")

    def load_views(self, cam_ixs, frame_idx, out_hw, *, depth=False, side="target"):
        cfg = self.ds.cfg
        names = [self.cam_names[int(i)] for i in cam_ixs]
        out = {"image": self._rgb(names, frame_idx, out_hw)}
        if depth:
            out["depth"] = self._npz(names, frame_idx, "depth", "depth", out_hw)
        if side != "viewer":
            if cfg.with_seg:
                out["seg"] = self._npz(names, frame_idx, "seg", "seg", out_hw)
            if cfg.with_mask and (side == "target" or cfg.mask_views == "both"):
                seg = self._npz(names, frame_idx, "seg", "seg", out_hw)
                thr = 128.0 if cfg.mask_mode == "dynamic" else 0.5
                out["state_mask"] = seg >= thr
        return out

    def load_actions(self, frame_indices: list[int]) -> dict[str, Tensor] | None:
        cfg = self.ds.cfg
        if not cfg.load_object_trajectories:
            return None
        traj = parse_object_trajectories(self.scene_dir / f"{self.rec['scene']}.json")
        return {
            name: build_object_action_tensor(series, frame_indices, cfg.object_traj_scale)
            for name, series in traj.items()
        }
