"""SynthRobot: synthetic bimanual manipulation (ManiSkill/SAPIEN, ``franka_duo`` robot).

Every *recording* holds *group* directories, each with one HDF5 trajectory store plus one
RGB and one depth mp4 per (episode, camera)::

    <root>/<...>/<recording>/
        recording_recipe.json                        # {"depth": {"min_m", "max_m"}}
        <group>/                                     # e.g. house_0
            trajectories_batch_1_of_1.h5             # traj_0 .. traj_{N-1}
            episode_00000000_<cam>_batch_1_of_1.mp4
            episode_00000000_<cam>_depth_batch_1_of_1.mp4

Facts the loader depends on:

* **Depth codec is recording-dependent.** Stores that declare
  ``obs_scene["depth_encoding"]`` use the hue-log codec in
  :mod:`ontic_data.depth_codec` (only ``range_m`` varies; other constants are checked
  against the vendored module). Undeclared stores are sniffed: packed-16
  (``z = (R*256 + G) / 65535 * (max - min) + min``, ``B`` constant) vs hue.
* **Macroblock padding.** The renderer emits 640x360 but streams are 640x368; the true
  content size is ``(2*cy, 2*cx)`` from the intrinsics and the frame is cropped to it.
* **Cameras.** ``obs/sensor_param/<cam>/extrinsic_cv`` is a per-timestep w2c 3x4 and
  ``intrinsic_cv`` a pixel-space 3x3; output is c2w 4x4 + normalised intrinsics.
* **Poses.** ``obs/extra/tcp_pose_{left,right}`` is ``xyz + wxyz`` in the robot base
  frame; ``obs/extra/robot_base_pose`` places that base in the world.
* **JSON blobs.** ``obs/agent/qpos`` rows are NUL-padded uint8 JSON.

Actions are world-frame point trajectories ``(T, 1, P, 4)``: the two TCPs
(``action_mode="tcp"`` P=1 or ``"frame"`` P=4 with axis tips) or ``"robot_points"``
sampled along the links from recorded joint angles (needs the robotics checkout and
mujoco, see :mod:`ontic_data.robot`). h5py is lazy (extra ``hdf5``).
"""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor

from .. import depth_codec
from .._np import normalize_intrinsics_np, w2c_to_c2w_np
from ..config import DatasetCfg, HorizonFn, Stage, StepFn
from ..temporal import SceneView, TemporalSceneDataset, resize_frames, split_records
from ..video import VideoBackend, VideoReader, has_video_backend
from ..view_sampler import ViewSampler

DEFAULT_DEPTH_MIN_M = 0.05
DEFAULT_DEPTH_MAX_M = 5.0

META_CSV_NAME = "synthrobot_info.csv"

H5_GLOB = "trajectories_batch_*.h5"
_H5_RE = re.compile(r"^trajectories_batch_(\d+)_of_(\d+)\.h5$")
# Camera names contain underscores AND hyphens (``wrist_camera_l``, ``1-1``), so anchor on
# the fixed prefix/suffix and take everything between as the camera, minus ``_depth``.
_VIDEO_RE = re.compile(
    r"^episode_(\d{8})_(?P<cam>.+?)(?P<depth>_depth)?_batch_(\d+)_of_(\d+)\.mp4$"
)


def _h5py():
    try:
        import h5py
    except ImportError as e:
        raise ImportError("SynthRobot trajectory stores need h5py; install ontic-data[hdf5]") from e
    return h5py


# --------------------------------------------------------------------------- #
# Depth codecs
# --------------------------------------------------------------------------- #


def decode_depth_rgb(rgb: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """Packed-16 ``(..., 3)`` uint8 -> metres ``(...)`` float32 (R high byte, G low)."""
    rgb = np.asarray(rgb)
    code = rgb[..., 0].astype(np.float32) * 256.0 + rgb[..., 1].astype(np.float32)
    return (code / 65535.0 * (max_m - min_m) + min_m).astype(np.float32)


def encode_depth_rgb(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """Inverse of :func:`decode_depth_rgb`."""
    unit = np.clip((np.asarray(depth_m, dtype=np.float64) - min_m) / (max_m - min_m), 0.0, 1.0)
    code = np.rint(unit * 65535.0).astype(np.int32)
    out = np.zeros(code.shape + (3,), dtype=np.uint8)
    out[..., 0] = (code >> 8) & 0xFF
    out[..., 1] = code & 0xFF
    return out


DEPTH_ENCODING_PACKED16 = "packed16"
DEPTH_ENCODING_HUE = "hue"
DEPTH_ENCODING_UNKNOWN = "unknown"


@dataclass(frozen=True)
class HueLogSpec:
    """What a store declares in ``obs_scene["depth_encoding"]``: only the metric range
    varies per recording; every other field is a codec constant and is checked against
    :mod:`ontic_data.depth_codec` rather than stored."""

    min_m: float
    max_m: float

    @property
    def relative_step(self) -> float:
        return depth_codec.relative_step(self.min_m, self.max_m)

    @classmethod
    def from_obs_scene(cls, scene: dict | None) -> HueLogSpec | None:
        """Parse the declaration; ``None`` when the store carries none (packed-16 stores).

        Raises when the declared codec constants disagree with the vendored module.
        """
        if not isinstance(scene, dict):
            return None
        enc = scene.get("depth_encoding")
        if not isinstance(enc, dict):
            return None
        if enc.get("name") != "hue-log":
            raise NotImplementedError(f"unknown depth encoding {enc.get('name')!r}")

        ours = {
            "hue_codes": depth_codec.HUE_CODES,
            "guard": depth_codec.GUARD,
            "levels": depth_codec.LEVELS,
            "sky_grey": depth_codec.SKY_GREY,
            "hole_grey": depth_codec.HOLE_GREY,
            "desaturation_cut": depth_codec.DESATURATION_CUT,
        }
        drift = {k: (enc[k], v) for k, v in ours.items() if k in enc and int(enc[k]) != v}
        if drift:
            raise ValueError(
                "depth_encoding disagrees with the vendored depth_codec (declared vs "
                f"vendored): {drift}"
            )
        lo, hi = enc.get("range_m", (DEFAULT_DEPTH_MIN_M, DEFAULT_DEPTH_MAX_M))
        return cls(min_m=float(lo), max_m=float(hi))


def detect_depth_encoding_rgb(frame_rgb: np.ndarray) -> str:
    """Classify one undeclared depth frame ``(H, W, 3)``: packed-16 leaves ``B`` constant."""
    a = np.asarray(frame_rgb)
    if len(np.unique(a[..., 2])) == 1:
        return DEPTH_ENCODING_PACKED16
    return DEPTH_ENCODING_HUE


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def quat_wxyz_to_matrix(q) -> np.ndarray:
    """Rotation matrix from a ``wxyz`` quaternion."""
    w, x, y, z = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose7_to_matrix(pose7) -> np.ndarray:
    """``[x, y, z, qw, qx, qy, qz]`` -> ``(4, 4)``."""
    pose7 = np.asarray(pose7, dtype=np.float64)
    M = np.eye(4)
    M[:3, :3] = quat_wxyz_to_matrix(pose7[3:7])
    M[:3, 3] = pose7[:3]
    return M


def compose_pose(base_pose7, local_pose7) -> np.ndarray:
    """``T_world_base @ T_base_local``."""
    return pose7_to_matrix(base_pose7) @ pose7_to_matrix(local_pose7)


def w2c34_to_c2w44(extrinsic_cv: np.ndarray) -> np.ndarray:
    """OpenCV world-to-camera ``(3, 4)`` -> camera-to-world ``(4, 4)``."""
    ext = np.asarray(extrinsic_cv, dtype=np.float64)
    return w2c_to_c2w_np(ext[:3, :3], ext[:3, 3])


def content_hw_from_intrinsics(K_px: np.ndarray, video_h: int, video_w: int) -> tuple[int, int]:
    """Rendered content size ``(2*cy, 2*cx)`` inside a macroblock-padded stream, clamped."""
    h = int(round(2.0 * float(K_px[1, 2])))
    w = int(round(2.0 * float(K_px[0, 2])))
    return min(max(h, 1), int(video_h)), min(max(w, 1), int(video_w))


def normalized_K_from_pixel_K(K_px: np.ndarray) -> np.ndarray:
    """Normalised ``K`` for a centred pinhole, using its own implied render size."""
    K_px = np.asarray(K_px, dtype=np.float32)
    h = max(int(round(2.0 * float(K_px[1, 2]))), 1)
    w = max(int(round(2.0 * float(K_px[0, 2]))), 1)
    return normalize_intrinsics_np(K_px, w, h)


def _matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> ``wxyz`` quaternion (branch on the largest diagonal term)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return np.asarray(q, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Actions
# --------------------------------------------------------------------------- #


def build_tcp_action_tensor(
    world_pose7: np.ndarray,
    frame_indices: list[int],
    mode: Literal["tcp", "frame"] = "frame",
    axis_len: float = 0.05,
) -> Tensor:
    """World-frame TCP poses ``(F, 7)`` -> ``(T, 1, P, 4)``; ``"tcp"`` keeps the origin
    (P=1), ``"frame"`` adds one tip per body axis at ``axis_len`` metres (P=4)."""
    world_pose7 = np.asarray(world_pose7, dtype=np.float64)
    n = world_pose7.shape[0]
    idx = [min(max(int(i), 0), n - 1) for i in frame_indices]

    if mode == "tcp":
        local = np.zeros((1, 3), dtype=np.float64)
    elif mode == "frame":
        local = np.concatenate([np.zeros((1, 3)), np.eye(3) * float(axis_len)], axis=0)
    else:
        raise ValueError(f"unknown action_mode {mode!r} (expected 'tcp' or 'frame')")

    pts = np.empty((len(idx), local.shape[0], 3), dtype=np.float32)
    for t, fi in enumerate(idx):
        R = quat_wxyz_to_matrix(world_pose7[fi, 3:7])
        pts[t] = (local @ R.T + world_pose7[fi, :3]).astype(np.float32)

    out = np.concatenate([pts, np.ones((*pts.shape[:2], 1), dtype=np.float32)], axis=-1)
    return torch.from_numpy(out[:, None]).float()


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def parse_video_name(name: str) -> dict | None:
    """Parse ``episode_<8d>_<cam>[_depth]_batch_<i>_of_<n>.mp4``; ``None`` otherwise."""
    m = _VIDEO_RE.match(name)
    if m is None:
        return None
    return {
        "episode": int(m.group(1)),
        "camera": m.group("cam"),
        "depth": m.group("depth") is not None,
        "batch": int(m.group(4)),
        "n_batches": int(m.group(5)),
    }


def _depth_range_for(group_dir: Path, root: Path) -> tuple[float, float]:
    """Nearest enclosing ``recording_recipe.json``'s depth range, else defaults."""
    root = root.resolve()
    d = group_dir.resolve()
    while True:
        recipe = d / "recording_recipe.json"
        if recipe.exists():
            try:
                depth = json.loads(recipe.read_text()).get("depth") or {}
                return (
                    float(depth.get("min_m", DEFAULT_DEPTH_MIN_M)),
                    float(depth.get("max_m", DEFAULT_DEPTH_MAX_M)),
                )
            except (json.JSONDecodeError, TypeError, ValueError):
                break
        if d == root or d.parent == d:
            break
        d = d.parent
    return DEFAULT_DEPTH_MIN_M, DEFAULT_DEPTH_MAX_M


def discover_recordings(root: str | Path) -> list[dict]:
    """One row per (group directory, batch) holding an HDF5 trajectory store.

    Row keys: ``rel_dir``, ``h5_name``, ``batch``, ``n_batches``, ``episode_offset``,
    ``n_traj``, ``n_frames`` (per trajectory), ``cameras`` (with both a sensor_param entry
    and an RGB video), ``depth_cameras``, ``depth_min_m``/``depth_max_m``,
    ``depth_encoding`` (declared spec or None), ``task_type``, ``success`` (per traj).
    """
    h5py = _h5py()
    root = Path(root)
    rows: list[dict] = []
    for h5_path in sorted(root.rglob(H5_GLOB)):
        m = _H5_RE.match(h5_path.name)
        if m is None:
            continue
        batch, n_batches = int(m.group(1)), int(m.group(2))
        group_dir = h5_path.parent

        rgb_cams: set[str] = set()
        depth_cams: set[str] = set()
        episodes: set[int] = set()
        for fn in group_dir.iterdir():
            info = parse_video_name(fn.name)
            if info is None or info["batch"] != batch or info["n_batches"] != n_batches:
                continue
            episodes.add(info["episode"])
            (depth_cams if info["depth"] else rgb_cams).add(info["camera"])

        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted(f.keys(), key=lambda k: int(k.split("_")[-1]))
            if not traj_keys:
                continue
            n_frames = [int(f[k]["rewards"].shape[0]) for k in traj_keys]
            sensor_cams = list(f[traj_keys[0]]["obs/sensor_param"].keys())
            success = [bool(np.asarray(f[k]["success"])[-1]) for k in traj_keys]
            try:
                scene = json.loads(np.asarray(f[traj_keys[0]]["obs_scene"]).item())
            except (json.JSONDecodeError, ValueError, AttributeError):
                scene = {}
            task_type = scene.get("task_type", "")
            hue_spec = HueLogSpec.from_obs_scene(scene)

        if len(episodes) != len(traj_keys):
            continue  # cannot pair episode videos with trajectories

        cameras = [c for c in sensor_cams if c in rgb_cams]
        if not cameras:
            continue

        depth_range = _depth_range_for(group_dir, root)
        rows.append(
            {
                "rel_dir": str(group_dir.relative_to(root)),
                "h5_name": h5_path.name,
                "batch": batch,
                "n_batches": n_batches,
                "episode_offset": min(episodes),
                "n_traj": len(traj_keys),
                "n_frames": n_frames,
                "cameras": cameras,
                "depth_cameras": [c for c in cameras if c in depth_cams],
                "depth_min_m": depth_range[0],
                "depth_max_m": depth_range[1],
                "depth_encoding": asdict(hue_spec) if hue_spec is not None else None,
                "task_type": task_type,
                "success": success,
            }
        )
    return rows


_CSV_COLUMNS = [
    "rel_dir", "h5_name", "batch", "n_batches", "episode_offset", "n_traj",
    "depth_min_m", "depth_max_m", "task_type", "cameras", "depth_cameras",
    "n_frames", "success", "depth_encoding",
]  # fmt: skip


def write_recordings_csv(rows: list[dict], path: str | Path) -> None:
    """Serialise :func:`discover_recordings` rows to a flat CSV."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    **{k: r.get(k, "") for k in _CSV_COLUMNS},
                    "cameras": ";".join(r["cameras"]),
                    "depth_cameras": ";".join(r["depth_cameras"]),
                    "n_frames": ";".join(str(n) for n in r["n_frames"]),
                    "success": ";".join("1" if s else "0" for s in r["success"]),
                    # JSON, and an empty cell round-trips to None (undeclared), not {}.
                    "depth_encoding": json.dumps(r["depth_encoding"])
                    if r.get("depth_encoding")
                    else "",
                }
            )


def read_recordings_csv(path: str | Path) -> list[dict]:
    """Inverse of :func:`write_recordings_csv`."""
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            row["cameras"] = [c for c in str(row.get("cameras", "")).split(";") if c]
            row["depth_cameras"] = [c for c in str(row.get("depth_cameras", "")).split(";") if c]
            row["n_frames"] = [int(n) for n in str(row["n_frames"]).split(";") if n]
            row["success"] = [s == "1" for s in str(row.get("success", "")).split(";") if s]
            enc = row.get("depth_encoding") or ""
            row["depth_encoding"] = json.loads(enc) if enc else None
            for k in ("batch", "n_batches", "n_traj", "episode_offset"):
                row[k] = int(row[k])
            for k in ("depth_min_m", "depth_max_m"):
                row[k] = float(row[k])
            out.append(row)
    return out


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass(kw_only=True)
class SynthRobotDatasetCfg(DatasetCfg):
    root: str = "/mnt/fast/mzhobro/synth_robot_data"

    image_shape: list[int] = field(default_factory=lambda: [360, 640])  # native 640x360
    near: float = 0.05  # the depth codec's range
    far: float = 5.0
    fps: int = 15  # policy_dt_ms = 66.0

    # Exact-name whitelist intersected per group (rigs differ between recordings); the
    # base ``camera_ixs_allowed`` then indexes positionally into what is left.
    camera_names: list[str] | None = None

    # "recording" = recording-level split; "none" = every recording in both stages.
    split_mode: Literal["recording", "none"] = "recording"
    require_success: bool = False  # keep only trajectories whose final success flag is set

    with_depth: bool = False
    video_backend: VideoBackend = "torchcodec"

    # Actions: "tcp"/"frame" = the two TCPs (see build_tcp_action_tensor);
    # "robot_points" = points sampled along every link from the recorded joint angles.
    load_actions: bool = True
    action_mode: Literal["tcp", "frame", "robot_points"] = "frame"
    action_axis_len: float = 0.05

    # Only read when action_mode == "robot_points".
    action_point_kind: Literal["skeleton", "surface"] = "skeleton"
    action_n_per_link: int = 10
    action_temporal_mode: Literal["tracks", "independent"] = "tracks"
    action_along: Literal["linspace", "uniform"] = "linspace"
    action_noise: float = 0.0
    action_visual_only: bool = True  # surface mode: drop the group-3 collision hulls
    action_seed: int = 0

    # Extra "robot_surface" stream (T, 1, P, 10) = xyz + normal + colour + presence,
    # meant as additional STATE anchors rather than an agent stream.
    state_surface_points: bool = False
    state_surface_n_per_link: int = 60

    workspace_min: list[float] | None = None
    workspace_max: list[float] | None = None

    def build(
        self, stage: Stage, *, step_fn: StepFn | None = None, horizon_fn: HorizonFn | None = None
    ):
        view_sampler = self.build_view_sampler(stage)
        return DatasetSynthRobot(self, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


class DatasetSynthRobot(TemporalSceneDataset):
    """A record is one (recording group, trajectory) pair; the recording-level split
    happens first, then trajectories are flattened into records."""

    cfg: SynthRobotDatasetCfg
    DEPTH_IS_METRIC = True

    def __init__(
        self,
        cfg: SynthRobotDatasetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        *,
        step_fn: StepFn | None = None,
        horizon_fn: HorizonFn | None = None,
    ) -> None:
        if not has_video_backend(cfg.video_backend):
            raise ImportError(
                f"SynthRobot needs the {cfg.video_backend!r} video backend; install ontic-data[video]"
            )
        self.root = Path(cfg.root).resolve()
        super().__init__(cfg, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)

    def _discover(self) -> list[dict]:
        meta_csv = self.root / META_CSV_NAME
        recordings = (
            read_recordings_csv(meta_csv) if meta_csv.exists() else discover_recordings(self.root)
        )
        if not recordings:
            raise FileNotFoundError(
                f"no SynthRobot recordings under {self.root} ('{H5_GLOB}' with episode videos)"
            )
        if self.cfg.split_mode != "none":
            recordings = split_records(recordings, self.stage, self._val_frac())
        self.recordings = recordings
        return [
            {"rec": rec, "traj": traj}
            for rec in recordings
            for traj in range(rec["n_traj"])
            if not (self.cfg.require_success and not rec["success"][traj])
        ]

    def _should_split(self) -> bool:
        return False  # recording-level split already applied in _discover

    def _n_frames_of(self, rec: dict) -> int:
        return int(rec["rec"]["n_frames"][rec["traj"]])

    def _scene_name(self, rec: dict, t_start: int) -> str:
        sp = self.cfg.speedup
        end = t_start + self.n_full_steps * sp
        return f"{rec['rec']['rel_dir']}/traj_{rec['traj']}_{t_start}:{end}:{sp}"

    def _record_label(self, rec: dict) -> str:
        return f"{rec['rec']['rel_dir']}/traj_{rec['traj']}"

    def _open_scene(self, rec: dict) -> SynthRobotSceneView:
        return SynthRobotSceneView(self, rec["rec"], rec["traj"])

    # ------------------------------------------------------------------ #
    # Recording access
    # ------------------------------------------------------------------ #

    def _group_dir(self, rec: dict) -> Path:
        return self.root / rec["rel_dir"]

    def _cameras_for(self, rec: dict) -> list[str]:
        names = list(rec["cameras"])
        if self.cfg.camera_names is not None:
            wanted = [c for c in self.cfg.camera_names if c in names]
            names = wanted or names
        if self.cfg.camera_ixs_allowed is not None:
            names = [names[i] for i in self.cfg.camera_ixs_allowed if i < len(names)]
        return names

    def recording_cameras(self, r_idx: int) -> list[str]:
        return self._cameras_for(self.recordings[r_idx])

    @staticmethod
    @lru_cache(maxsize=8)
    def _h5(path: str):
        return _h5py().File(path, "r")  # per process; workers keep their own handle

    def _traj_group(self, rec: dict, traj: int):
        return self._h5(str(self._group_dir(rec) / rec["h5_name"]))[f"traj_{traj}"]

    def _camera_arrays(
        self, rec: dict, traj: int, cam_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Per-timestep c2w ``(V, T, 4, 4)`` and pixel-space ``K (V, 3, 3)`` (frame 0)."""
        g = self._traj_group(rec, traj)
        c2w, K = [], []
        for cam in cam_names:
            sp = g[f"obs/sensor_param/{cam}"]
            w2c = np.asarray(sp["extrinsic_cv"])  # (T, 3, 4)
            c2w.append(np.stack([w2c34_to_c2w44(w2c[t]) for t in range(w2c.shape[0])]))
            K.append(np.asarray(sp["intrinsic_cv"][0], dtype=np.float32))
        return np.stack(c2w, 0), np.stack(K, 0)

    def _video_path(self, rec: dict, traj: int, cam: str, depth: bool) -> Path:
        episode = rec["episode_offset"] + traj
        suffix = "_depth" if depth else ""
        name = f"episode_{episode:08d}_{cam}{suffix}_batch_{rec['batch']}_of_{rec['n_batches']}.mp4"
        return self._group_dir(rec) / name

    def _world_tcp_poses(self, rec: dict, traj: int, side: str) -> np.ndarray:
        """``(T, 7)`` world-frame TCP poses."""
        g = self._traj_group(rec, traj)
        base = np.asarray(g["obs/extra/robot_base_pose"], dtype=np.float64)
        local = np.asarray(g[f"obs/extra/tcp_pose_{side}"], dtype=np.float64)
        out = np.empty_like(local)
        for t in range(local.shape[0]):
            M = compose_pose(base[t], local[t])
            out[t, :3] = M[:3, 3]
            out[t, 3:7] = _matrix_to_quat_wxyz(M[:3, :3])
        return out

    def _joint_stream(self, rec: dict, traj: int, frame_indices: list[int]):
        """Per-frame joint angles + the trajectory's (frame-0) base pose."""
        from ..robot import duo_joint_angles

        g = self._traj_group(rec, traj)
        n = int(g["obs/extra/robot_base_pose"].shape[0])
        idx = [min(max(i, 0), n - 1) for i in frame_indices]
        angles = [
            duo_joint_angles(
                json.loads(np.asarray(g["obs/agent/qpos"][i]).tobytes().rstrip(b"\x00"))
            )
            for i in idx
        ]
        base = np.asarray(g["obs/extra/robot_base_pose"][idx[0]], dtype=np.float64)
        return angles, base

    def _robot_surface_stream(self, rec: dict, traj: int, frame_indices: list[int]) -> Tensor:
        """``(T, 1, P, 10)`` surface-state stream: xyz + normal + colour + presence."""
        from ..robot import franka_duo_kinematics, sample_surface_points

        angles, base = self._joint_stream(rec, traj, frame_indices)
        pts, nrm, colors, _links = sample_surface_points(
            franka_duo_kinematics(),
            angles,
            n_per_link=self.cfg.state_surface_n_per_link,
            temporal_mode=self.cfg.action_temporal_mode,
            base_pose=base,
            visual_only=self.cfg.action_visual_only,
            seed=self.cfg.action_seed,
        )
        t, p = pts.shape[:2]
        out = np.ones((t, p, 10), dtype=np.float32)
        out[..., 0:3] = pts
        out[..., 3:6] = nrm
        out[..., 6:9] = np.broadcast_to(colors[None], (t, p, 3))
        return torch.from_numpy(out[:, None])

    def _robot_point_actions(self, rec: dict, traj: int, frame_indices: list[int]) -> Tensor:
        """``(T, 1, P, 4)`` points riding along the robot's links (whole snippet sampled in
        one call so ``temporal_mode="tracks"`` draws its local samples once)."""
        from ..robot import (
            franka_duo_kinematics,
            sample_action_points,
            sample_surface_points,
            to_action_tensor,
        )

        angles, base = self._joint_stream(rec, traj, frame_indices)
        if self.cfg.action_point_kind == "surface":
            points, _normals, _colors, _links = sample_surface_points(
                franka_duo_kinematics(),
                angles,
                n_per_link=self.cfg.action_n_per_link,
                temporal_mode=self.cfg.action_temporal_mode,
                base_pose=base,
                visual_only=self.cfg.action_visual_only,
                seed=self.cfg.action_seed,
            )
        else:
            points, _links = sample_action_points(
                franka_duo_kinematics(),
                angles,
                n_per_link=self.cfg.action_n_per_link,
                temporal_mode=self.cfg.action_temporal_mode,
                along=self.cfg.action_along,
                noise=self.cfg.action_noise,
                base_pose=base,
                seed=self.cfg.action_seed,
            )
        return torch.from_numpy(to_action_tensor(points))

    def robot_state_of(self, rec: dict, traj: int, frame_idx: int) -> dict | None:
        """``{"qpos": named joint groups, "base_pose": (7,) [xyz, wxyz]}`` or None."""
        g = self._traj_group(rec, traj)
        if "obs/agent/qpos" not in g or "obs/extra/robot_base_pose" not in g:
            return None
        fi = min(frame_idx, g["obs/extra/robot_base_pose"].shape[0] - 1)
        raw = np.asarray(g["obs/agent/qpos"][fi]).tobytes().rstrip(b"\x00")
        return {
            "qpos": json.loads(raw),
            "base_pose": np.asarray(g["obs/extra/robot_base_pose"][fi], dtype=np.float64),
        }


class SynthRobotSceneView(SceneView):
    """Cameras from the h5 (per-timestep c2w), pixels from per-camera mp4s."""

    def __init__(self, ds: DatasetSynthRobot, rec: dict, traj: int) -> None:
        self.ds, self.rec, self.traj = ds, rec, traj
        self.cam_names = ds._cameras_for(rec)
        self.cam_c2w, self.K_px = ds._camera_arrays(rec, traj, self.cam_names)
        self.intrinsics = torch.stack(
            [torch.from_numpy(normalized_K_from_pixel_K(K)) for K in self.K_px]
        )
        self._readers: dict[str, VideoReader] = {}

    def extrinsics(self, frame_idx: int) -> Tensor:
        fi = min(frame_idx, self.cam_c2w.shape[1] - 1)
        return torch.from_numpy(self.cam_c2w[:, fi]).float()

    def _frame(self, cam: str, frame_idx: int, depth: bool) -> Tensor:
        """One raw ``(h, w, 3)`` uint8 frame, still macroblock-padded."""
        path = str(self.ds._video_path(self.rec, self.traj, cam, depth))
        vr = self._readers.get(path)
        if vr is None:
            vr = self._readers[path] = VideoReader(path, self.ds.cfg.video_backend)
        return vr.get_frames([min(max(frame_idx, 0), len(vr) - 1)])[0]

    def load_views(self, cam_ixs, frame_idx, out_hw, *, depth=False, side="target"):
        names = [self.cam_names[int(i)] for i in cam_ixs]
        Ks = [self.K_px[int(i)] for i in cam_ixs]

        imgs = []
        for cam, K in zip(names, Ks):
            raw = self._frame(cam, frame_idx, depth=False)
            h, w = content_hw_from_intrinsics(K, raw.shape[0], raw.shape[1])
            imgs.append(raw[:h, :w])
        image = torch.stack(imgs).permute(0, 3, 1, 2).float() / 255.0
        out = {"image": resize_frames(image, *out_hw, "bilinear")}

        if depth:
            missing = [c for c in names if c not in self.rec["depth_cameras"]]
            if missing:
                raise FileNotFoundError(
                    f"{self.rec['rel_dir']}: no depth video for camera(s) {missing}"
                )
            declared = self.rec.get("depth_encoding")
            spec = HueLogSpec(**declared) if declared else None
            zs = []
            for cam, K in zip(names, Ks):
                raw = self._frame(cam, frame_idx, depth=True)
                enc = (
                    DEPTH_ENCODING_HUE
                    if spec is not None
                    else detect_depth_encoding_rgb(raw.numpy())
                )
                if enc == DEPTH_ENCODING_UNKNOWN:
                    raise NotImplementedError(f"{self.rec['rel_dir']}: unrecognised depth encoding")
                h, w = content_hw_from_intrinsics(K, raw.shape[0], raw.shape[1])
                crop = raw[:h, :w].numpy()
                if enc == DEPTH_ENCODING_PACKED16:
                    z = decode_depth_rgb(crop, self.rec["depth_min_m"], self.rec["depth_max_m"])
                else:
                    # ``kind`` (VALID / SKY / HOLE) is dropped: both invalid kinds read 0.
                    lo, hi = (
                        (spec.min_m, spec.max_m)
                        if spec is not None
                        else (self.rec["depth_min_m"], self.rec["depth_max_m"])
                    )
                    z, _kind = depth_codec.decode_rgb_to_depth(crop, lo, hi)
                zs.append(torch.from_numpy(z))
            out["depth"] = resize_frames(torch.stack(zs).unsqueeze(1), *out_hw, "nearest")
        return out

    def load_actions(self, frame_indices: list[int]) -> dict[str, Tensor] | None:
        cfg = self.ds.cfg
        if not cfg.load_actions:
            return None
        if cfg.action_mode == "robot_points":
            out = {"robot_points": self.ds._robot_point_actions(self.rec, self.traj, frame_indices)}
            if cfg.state_surface_points:
                out["robot_surface"] = self.ds._robot_surface_stream(
                    self.rec, self.traj, frame_indices
                )
            return out
        return {
            f"{side}_tcp": build_tcp_action_tensor(
                self.ds._world_tcp_poses(self.rec, self.traj, side),
                frame_indices,
                mode=cfg.action_mode,
                axis_len=cfg.action_axis_len,
            )
            for side in ("left", "right")
        }

    def robot_state(self, frame_idx: int) -> dict | None:
        return self.ds.robot_state_of(self.rec, self.traj, frame_idx)
