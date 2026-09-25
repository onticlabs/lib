"""HO-Cap: 8 static RealSense cameras, per-frame hand labels, optional depth / masks."""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ..config import DatasetCfg, HorizonFn, Stage, StepFn
from ..temporal import SceneView, TemporalSceneDataset
from ..view_sampler import ViewSampler
from .hocap_seq_loader import SequenceLoader
from ontic_lib.camera.intrinsics import normalize_intrinsics


def build_hocap_hand_actions(
    hand_keypoints_world: np.ndarray, mano_sides: list[str]
) -> dict[str, torch.Tensor]:
    """Per-hand ``(T, 1, 21, 4)`` actions with a per-frame presence bit.

    ``hand_keypoints_world``: ``(T, 2, 21, 3)`` world-frame keypoints, slot 0 = right,
    slot 1 = left (a missing side carries a sentinel). Presence is 1 when the side
    appears in ``mano_sides``; absent-hand xyz is zero-filled.
    """
    if hand_keypoints_world.ndim != 4 or hand_keypoints_world.shape[1:] != (2, 21, 3):
        raise ValueError(f"expected (T, 2, 21, 3); got {hand_keypoints_world.shape}")
    T_full = hand_keypoints_world.shape[0]
    out: dict[str, torch.Tensor] = {}
    for slot, name in enumerate(("right", "left")):
        present = name in mano_sides
        xyz = hand_keypoints_world[:, slot : slot + 1].astype(np.float32)
        if not present:
            xyz[...] = 0.0
        presence = np.full((T_full, 1, 21, 1), float(present), dtype=np.float32)
        out[f"{name}_hand"] = torch.from_numpy(np.concatenate([xyz, presence], axis=-1))
    return out


@dataclass(kw_only=True)
class HocapDatasetCfg(DatasetCfg):
    root: str = "/fast/mzhobro/hocap_dataset2/hocap_dataset"

    image_shape: list[int] = field(default_factory=lambda: [480, 640])
    near: float = 0.1
    far: float = 30.0
    camera_ixs_allowed: list[int] | None = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])

    workspace_min: list[float] | None = field(default_factory=lambda: [-0.96, -0.83, -0.3])
    workspace_max: list[float] | None = field(default_factory=lambda: [0.84, 0.48, 0.3])

    background_color: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    with_mask: bool = False
    with_depth: bool = False
    # Skip building per-frame hand actions (NVS pretraining does not consume them).
    load_hand_poses: bool = True

    def build(
        self, stage: Stage, *, step_fn: StepFn | None = None, horizon_fn: HorizonFn | None = None
    ):
        view_sampler = self.build_view_sampler(stage)
        return DatasetHOCap(self, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)


class DatasetHOCap(TemporalSceneDataset):
    """One record per SequenceLoader; images at native 480x640, the crop shim resizes."""

    cfg: HocapDatasetCfg

    def __init__(
        self,
        cfg: HocapDatasetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        *,
        step_fn: StepFn | None = None,
        horizon_fn: HorizonFn | None = None,
    ) -> None:
        super().__init__(cfg, stage, view_sampler, step_fn=step_fn, horizon_fn=horizon_fn)
        self.num_views = self.records[0].num_cams

    def _discover(self) -> list[SequenceLoader]:
        all_paths = sorted(glob.glob(f"{self.cfg.root}/subject_*/*_*"))
        paths = self._split_sequences(all_paths, self.stage, self.cfg.n_step_predict)
        return [SequenceLoader(p, device="cpu") for p in paths if (Path(p) / "meta.yaml").exists()]

    def _should_split(self) -> bool:
        return False  # sequence-level split applied in _discover

    @staticmethod
    def _split_sequences(paths: list[str], stage: str, n_step_predict: int) -> list[str]:
        """Dyn (``n_step_predict > 0``): hold out 4 sequences; NVS: 90/10 by shuffled
        index. ``"test"`` aliases ``"val"``."""
        n = len(paths)
        if n < 5:
            return paths
        order = np.arange(n)
        np.random.RandomState(42).shuffle(order)
        n_val = min(4, n - 1) if n_step_predict > 0 else max(1, int(round(n * 0.10)))
        val_idx = sorted(order[:n_val].tolist())
        train_idx = sorted(order[n_val:].tolist())
        idx = train_idx if stage == "train" else val_idx
        return [paths[i] for i in idx]

    def _n_frames_of(self, sl: SequenceLoader) -> int:
        return int(sl.num_frames)

    def _snippet_starts(self, sl, n_frames: int):
        # Dyn validation: one snippet per sequence starting at raw frame ~40.
        if self.validation and self.cfg.n_step_predict != 0:
            sp = self.cfg.speedup
            snippet_len = self.n_full_steps * sp
            return [min((40 // sp) * sp, max(0, n_frames - snippet_len))]
        return super()._snippet_starts(sl, n_frames)

    def _scene_name(self, sl: SequenceLoader, t_start: int) -> str:
        sp = self.cfg.speedup
        end = t_start + self.n_full_steps * sp
        return f"{sl.subject_id}_{sl.task_id}_{sl.sequence_name}_{t_start}:{end}:{sp}"

    def _record_label(self, sl: SequenceLoader) -> str:
        return f"{sl.subject_id}_{sl.task_id}"

    def _open_scene(self, sl: SequenceLoader) -> HocapSceneView:
        return HocapSceneView(self, sl)


class HocapSceneView(SceneView):
    """Static rig from the sequence calibration; pixels via the SequenceLoader."""

    def __init__(self, ds: DatasetHOCap, sl: SequenceLoader) -> None:
        self.ds, self.sl = ds, sl
        allowed = (
            np.arange(sl.num_cams)
            if ds.cfg.camera_ixs_allowed is None
            else np.array(ds.cfg.camera_ixs_allowed)
        )
        self.cam_names = [sl.rs_serials[i] for i in allowed]
        self._extr = sl.rs_RTs[allowed]  # (V, 4, 4) c2w
        self.intrinsics = normalize_intrinsics(sl.rs_Ks[allowed], (sl.rs_height, sl.rs_width))

    def extrinsics(self, frame_idx: int) -> Tensor:
        return self._extr

    def load_views(self, cam_ixs, frame_idx, out_hw, *, depth=False, side="target"):
        cfg, sl = self.ds.cfg, self.sl
        serials = [self.cam_names[int(i)] for i in cam_ixs]
        rgbs = np.stack([sl.get_rgb_image(frame_idx, s) for s in serials], 0) / 255.0
        out = {"image": torch.from_numpy(rgbs).float().permute(0, 3, 1, 2)}
        if depth:
            d = np.stack([sl.get_depth_image(frame_idx, s) for s in serials], 0)
            out["depth"] = torch.from_numpy(d).float().unsqueeze(1)
        if cfg.with_mask and side != "viewer":
            full = np.ones((sl.rs_height, sl.rs_width), dtype=bool)
            seg = np.stack(
                [sl.get_seg_mask(frame_idx, s).get("combined", full) for s in serials], 0
            )
            out["state_mask"] = torch.from_numpy(seg).bool().unsqueeze(1)
        return out

    def load_actions(self, frame_indices: list[int]) -> dict[str, torch.Tensor] | None:
        if not self.ds.cfg.load_hand_poses:
            return None
        sl = self.sl
        keypoints = []
        for frame_idx in frame_indices:
            kp_world = None
            # hand_joints_3d are stored per camera in that camera's frame; any labelled
            # camera yields the same world-frame keypoints after its c2w.
            for i, serial in enumerate(sl.rs_serials):
                label = sl.get_image_label(frame_idx, serial)
                if "hand_joints_3d" in label:
                    kp_cam = label["hand_joints_3d"]  # (2, 21, 3) camera frame
                    kp_h = np.concatenate([kp_cam, np.ones((*kp_cam.shape[:-1], 1))], axis=-1)
                    kp_world = np.einsum("ij,...j->...i", sl.rs_RTs[i].numpy(), kp_h)[..., :3]
                    break
            if kp_world is None:
                raise ValueError(f"no hand labels for {sl.sequence_name} frame {frame_idx}")
            keypoints.append(kp_world)
        return build_hocap_hand_actions(np.stack(keypoints, 0), list(sl.mano_sides))
