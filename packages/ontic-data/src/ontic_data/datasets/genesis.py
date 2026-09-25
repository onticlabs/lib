"""Genesis synthetic scenes: per-frame image files + per-scene ``metadata.json``.

On-disk layout: ``<root>/(<sub>/)*<scene>/metadata.json`` with ``n_cams``, per-camera
``cam_<j>`` blocks (``extrinsics`` w2c 4x4, ``intrinsics`` pixel 3x3, ``resolution``
``[W, H]``), a ``file_path_template`` and optional ``extras.npy`` actions
``(T, N_copies, N_points, 3)``. Images come back at native resolution; the crop shim
resizes. PIL is imported lazily (extra ``images``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import Tensor

from .._np import normalize_intrinsics_np
from ..config import DatasetCfg, HorizonFn, Stage, StepFn
from ..temporal import SceneView, TemporalSceneDataset
from ..view_sampler import ViewSampler
from ontic_lib.transforms.rigid import invert_rigid_transform


@dataclass
class AgentCfg:
    n_points: int


@dataclass(kw_only=True)
class DatasetGenesisCfg(DatasetCfg):
    near: float = 0.6
    far: float = 7.0

    roots: list[str] = field(
        default_factory=lambda: ["/fast/mzhobro/datasets/soft_genesis_elastic"]
    )
    root_depths: list[int] = field(default_factory=lambda: [0])
    overfit_to_scene: str | None = None

    with_mask: bool = False
    with_seg: bool = False

    agents: Optional[dict[str, AgentCfg]] = None

    def build(
        self, stage: Stage, *, step_fn: StepFn | None = None, horizon_fn: HorizonFn | None = None
    ):
        view_sampler = self.build_view_sampler(stage)
        return DatasetGenesis(self, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)


def _open_image(path: str):
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("genesis frames need pillow; install ontic-data[images]") from e
    return Image.open(path)


def _pil_to_tensor(img, scaled: bool) -> Tensor:
    """PIL image -> ``(C, H, W)``; ``scaled`` divides uint8 by 255 into float."""
    arr = np.array(img, copy=True)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if not scaled:
        return t
    return t.float().div(255) if t.dtype == torch.uint8 else t.float()


def get_list_of_scene_configs(root, root_depth: int = 0, agents=None) -> list[dict]:
    """Parse every ``metadata.json`` ``root_depth`` levels below ``root``."""
    with_actions = bool(agents)
    if with_actions:
        agent_name, _ = agents[0]

    config_paths = list(Path(root).glob("*/" * root_depth + "metadata.json"))
    scene_paths = [p.parent for p in config_paths]
    scene_configs = [json.load(open(p, "r")) for p in config_paths]

    for scene_path, scene_config in zip(scene_paths, scene_configs):
        scene_config["file_path_template"] = str(
            Path(scene_path) / scene_config["file_path_template"]
        )
        scene_config["scene_name"] = scene_path.name + "_" + scene_path.name

        action_path = scene_path / "extras.npy"
        if action_path.exists() and with_actions:
            actions = np.load(action_path)  # (T, N_copies, N_points, 3)
            if actions.size > 0:  # empty files are saved when no action was recorded
                scene_config[agent_name] = actions
    return scene_configs


def get_split(scene_configs: list[dict], test_ratio: float, stage: str) -> list[dict]:
    """Every-nth scene split; ``"test"`` aliases ``"val"``."""
    n_scenes = len(scene_configs)
    N_test = 1 if n_scenes < 15 else int(test_ratio * n_scenes)
    every_nth = int(n_scenes / N_test)
    test_idxs = list(range(0, n_scenes, every_nth))
    if stage == "train":
        return [
            scene_configs[i] for i in range(n_scenes) if i not in test_idxs or len(test_idxs) < 15
        ]
    return [scene_configs[i] for i in test_idxs]


class DatasetGenesis(TemporalSceneDataset):
    """One record per scene config.

    ``overfit_to_scene`` rewrites ``self.stage`` to "train" but keeps ``validation`` from
    the requested stage; validation with ``n_step_predict != 0`` rolls out the full scene.
    """

    cfg: DatasetGenesisCfg

    def __init__(
        self,
        cfg: DatasetGenesisCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        *,
        step_fn: StepFn | None = None,
        horizon_fn: HorizonFn | None = None,
    ) -> None:
        super().__init__(cfg, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)
        if cfg.overfit_to_scene:
            self.stage = "train"

    def _discover(self) -> list[dict]:
        cfg = self.cfg
        agents = list(cfg.agents.items()) if cfg.agents else None

        if cfg.overfit_to_scene is not None and Path(cfg.overfit_to_scene).exists():
            scene_configs = get_list_of_scene_configs(
                Path(cfg.overfit_to_scene), cfg.root_depths[0], agents=agents
            )
        else:
            scene_configs = []
            for root, root_depth in zip(cfg.roots, cfg.root_depths):
                scene_configs += get_list_of_scene_configs(root, root_depth, agents=agents)
            if not scene_configs:
                raise FileNotFoundError(f"no scene found in {cfg.roots}")
            if cfg.overfit_to_scene is not None:
                scene_configs = [scene_configs[0]]

        num_cams = scene_configs[0]["n_cams"]
        for scene in scene_configs:
            if scene["n_cams"] != num_cams:
                raise ValueError(f"n_cams: {scene['n_cams']} != {num_cams}")

        # The metadata's n_steps is unreliable; count the frames on disk (first scene,
        # first present camera — scenes are recorded uniformly).
        n_frames_per_scene = 0
        scene = scene_configs[0]
        scene_path = Path(scene["file_path_template"]).parent.parent.parent
        ext = ".jpg" if scene.get("video_format") in ("jpg", "jpeg") else ".png"
        for cam_ix in range(num_cams):
            rgb_dir = scene_path / f"cam_{cam_ix}" / "rgb"
            if rgb_dir.exists():
                n_frames_per_scene = len(list(rgb_dir.glob(f"*{ext}")))
                break
        if n_frames_per_scene == 0:
            raise ValueError("no valid frames found in any scene")
        self.n_frames_per_scene = int(n_frames_per_scene)

        self.num_cams = num_cams
        self.cams = (
            list(range(num_cams))
            if cfg.camera_ixs_allowed is None
            else list(cfg.camera_ixs_allowed)
        )

        stage = "train" if cfg.overfit_to_scene else self.stage
        return get_split(scene_configs, 0.12, stage)

    def _should_split(self) -> bool:
        return False  # get_split runs inside _discover

    def _n_frames_of(self, rec: dict) -> int:
        return self.n_frames_per_scene

    def _n_full_steps(self) -> int:
        if self.validation and self.cfg.n_step_predict != 0:
            return self.n_frames_per_scene  # validation rolls out the full scene
        return super()._n_full_steps()

    def _scene_name(self, rec: dict, t_start: int) -> str:
        return f"{rec['scene_name']}_{t_start}:{t_start + self.n_full_steps * self.cfg.speedup}"

    def _record_label(self, rec: dict) -> str:
        return str(rec["scene_name"])

    def _open_scene(self, rec: dict) -> GenesisSceneView:
        return GenesisSceneView(self, rec)


class GenesisSceneView(SceneView):
    """Per-scene JSON calibration (static rig), pixels from per-frame files."""

    def __init__(self, ds: DatasetGenesis, sc: dict) -> None:
        self.ds, self.sc = ds, sc
        self.cam_names = [f"cam_{j}" for j in ds.cams]
        # Stored per-cam "extrinsics" are w2c; invert to the pipeline's c2w.
        w2c = torch.tensor(
            np.array([sc[f"cam_{j}"]["extrinsics"] for j in ds.cams], dtype=np.float32)
        )
        self._extr = invert_rigid_transform(w2c)
        intr = [
            normalize_intrinsics_np(
                np.array(sc[f"cam_{j}"]["intrinsics"]), *sc[f"cam_{j}"]["resolution"]
            )
            for j in ds.cams
        ]
        self.intrinsics = torch.tensor(np.array(intr), dtype=torch.float32)
        self._ext = ".jpg" if sc.get("video_format") in ("jpg", "jpeg") else ".png"

    def extrinsics(self, frame_idx: int) -> Tensor:
        return self._extr

    def _open(self, cam_j: int, frame_idx: int, img_type: str, ext: str):
        path = self.sc["file_path_template"].format(
            cam_ix=cam_j, n_step=frame_idx, img_type=img_type
        )
        return _open_image(path + ext)

    def load_views(self, cam_ixs, frame_idx, out_hw, *, depth=False, side="target"):
        cfg = self.ds.cfg
        cams = [self.ds.cams[int(i)] for i in cam_ixs]
        out = {
            "image": torch.stack(
                [_pil_to_tensor(self._open(j, frame_idx, "rgb", self._ext), True) for j in cams]
            )
        }
        if cfg.with_mask or cfg.with_seg:
            seg = torch.stack(
                [_pil_to_tensor(self._open(j, frame_idx, "seg", ".png"), False) for j in cams]
            )
            if cfg.with_mask:
                out["state_mask"] = (seg > seg[0].min()).float()
            if cfg.with_seg:
                out["static_float"] = (seg > 2).float()
        return out

    def load_actions(self, frame_indices: list[int]) -> dict[str, Tensor] | None:
        agents = list(self.ds.cfg.agents.items()) if self.ds.cfg.agents else None
        if not agents:
            return None
        out = {}
        for agent_name, _ in agents:
            if agent_name in self.sc:
                arr = np.asarray(self.sc[agent_name], dtype=np.float32)
                idx = [min(i, arr.shape[0] - 1) for i in frame_indices]
                xyz = torch.from_numpy(arr[idx].copy())  # (T, N_copies, N_points, 3)
                out[agent_name] = torch.cat([xyz, torch.ones(*xyz.shape[:-1], 1)], dim=-1)
        return out or None
