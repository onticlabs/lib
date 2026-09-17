"""Uniform single-frame, all-camera access to the registered datasets.

Every ``ontic_data`` dataset builds a ``TemporalSceneDataset`` exposing
``record_labels`` / ``record_n_frames`` / ``load_sequence_views``, so one
:class:`GenericSource` covers them all.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Callable, Protocol, runtime_checkable

import torch
from torch import Tensor

from ontic_data.temporal import TemporalSceneDataset

#: Dev-box dataset roots (local ``/mnt/fast`` == cluster ``/fast``); override per CLI flag.
DEFAULT_ROOTS: dict[str, str] = {
    "dextris": "/mnt/fast/mzhobro/dextris_dataset",
    "robot-dextris": "/mnt/fast/mzhobro/trailer-demo",
    "hocap": "/mnt/fast/mzhobro/hocap_dataset2/hocap_dataset",
    "taco": "/mnt/fast/mzhobro/taco_dataset_resized",
    "genesis": "/mnt/fast/mzhobro/datasets/soft_genesis_elastic",
    "physinone": "/mnt/fast/mzhobro/physinone_dataset",
    "synthrobot": "/mnt/fast/mzhobro/synth_robot_data",
}

#: Which datasets can hand back ground-truth depth (GUI hint for the ``gtdepth`` backbone).
HAS_GT_DEPTH: dict[str, bool] = {
    "dextris": False,
    "robot-dextris": False,
    "hocap": True,
    "taco": False,
    "genesis": False,
    "physinone": True,
    "synthrobot": True,
}


def dataset_has_gt_depth(name: str) -> bool:
    return HAS_GT_DEPTH.get(name, False)


def dataset_names(registry: dict | None = None) -> list[str]:
    """Registered dataset keys (``ontic_data.DATASETS`` by default)."""
    if registry is None:
        from ontic_data import DATASETS as registry
    return list(registry)


@dataclass
class Frame:
    """All cameras of one (trajectory, timestep).

    ``images (V, 3, H, W)`` in ``[0, 1]``; ``intrinsics (V, 3, 3)`` normalised;
    ``extrinsics (V, 4, 4)`` c2w; ``cam_names`` per view; ``hands (H, 21, 3)`` world
    keypoints of the present hands or ``None``; ``depth (V, 1, H, W)`` metric GT depth
    (only when requested and available); ``robot`` recorded ``{"qpos", "base_pose"}``.
    """

    images: Tensor
    intrinsics: Tensor
    extrinsics: Tensor
    cam_names: list[str]
    hands: Tensor | None = None
    depth: Tensor | None = None
    robot: dict | None = None


@runtime_checkable
class DataSource(Protocol):
    name: str

    def list_trajectories(self) -> list[str]: ...

    def num_timesteps(self, traj: int) -> int: ...

    def get_frame(self, traj: int, t: int, with_depth: bool = False) -> Frame: ...


def _valid_hand(kp: Tensor) -> bool:
    return bool(torch.isfinite(kp).all()) and float(kp.std(dim=0).sum()) > 1e-4


def present_hands_from_action(action: dict) -> Tensor | None:
    """Stack ``(H, 21, 3)`` keypoints of the present hands from a single-frame action
    dict (``left_hand`` / ``right_hand`` entries of shape ``(1, 21, C>=3)``, an optional
    presence bit at channel 3)."""
    out = []
    for key in ("left_hand", "right_hand"):
        if key not in action:
            continue
        kp = action[key][0]
        if kp.shape[-1] >= 4 and float(kp[0, 3]) <= 0.5:
            continue
        kp = kp[:, :3].float()
        if _valid_hand(kp):
            out.append(kp)
    return torch.stack(out) if out else None


class GenericSource:
    """Adapter over any ``TemporalSceneDataset``-based dataset."""

    def __init__(self, name: str, ds) -> None:
        self.name = name
        self.ds = ds
        self._frame_lock = Lock()
        self._scene = None
        self._scene_traj = None
        labels = ds.record_labels()
        if len(set(labels)) < len(labels):
            labels = [f"{i:03d}_{label}" for i, label in enumerate(labels)]
        self._labels = labels

    def list_trajectories(self) -> list[str]:
        return self._labels

    def num_timesteps(self, traj: int) -> int:
        return int(self.ds.record_n_frames(traj))

    def get_frame(self, traj: int, t: int, with_depth: bool = False) -> Frame:
        # Keep only this source's current trajectory open. Serialize reads because
        # playback and background tracking can otherwise seek the same decoder.
        with self._frame_lock:
            kwargs = {"with_robot": True}
            if isinstance(self.ds, TemporalSceneDataset):
                if self._scene_traj != traj:
                    self._scene = None
                    self._scene_traj = None
                    self._scene = self.ds._open_scene(self.ds.records[traj])
                    self._scene_traj = traj
                kwargs["scene_view"] = self._scene
            try:
                d = self.ds.load_sequence_views(traj, t, with_depth=with_depth, **kwargs)
            except NotImplementedError:
                # An undecodable depth stream still permits calibrated RGB viewing.
                d = self.ds.load_sequence_views(traj, t, with_depth=False, **kwargs)
        try:
            hands = present_hands_from_action(d.get("actions") or {})
        except Exception:
            hands = None
        depth = d.get("depth")
        return Frame(
            images=d["image"].float(),
            intrinsics=d["intrinsics"].float(),
            extrinsics=d["extrinsics"].float(),
            cam_names=[str(c) for c in d["index"]],
            hands=hands,
            depth=depth.float() if depth is not None else None,
            robot=d.get("robot"),
        )


def _root_kwargs(root: str | None) -> dict:
    return {"root": root} if root else {}


def _genesis_kwargs(root: str | None) -> dict:
    kwargs: dict = {"root_depths": [2]}
    if root:
        kwargs["roots"] = [root]
    return kwargs


#: How a CLI root maps onto each dataset config's constructor kwargs.
CFG_KWARGS: dict[str, Callable[[str | None], dict]] = {"genesis": _genesis_kwargs}


def build_source(
    name: str,
    root: str | None = None,
    stage: str = "val",
    *,
    registry: dict | None = None,
) -> DataSource:
    """Build the adapter for dataset ``name`` from ``registry`` (``ontic_data.DATASETS``)."""
    if registry is None:
        from ontic_data import DATASETS as registry
    if name not in registry:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(registry)}")
    cfg = registry[name](**CFG_KWARGS.get(name, _root_kwargs)(root))
    return GenericSource(name, cfg.build(stage))
