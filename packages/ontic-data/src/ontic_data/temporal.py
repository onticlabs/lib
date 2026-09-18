"""Shared skeleton for temporal multi-view datasets.

Every dataset serves temporal snippets of a multi-camera scene — ``n_full_steps``
timesteps at stride ``speedup``, each split by the view sampler into context views
(encoder input) and target views (render supervision), plus optional per-frame actions.
Only *discovery* (what scenes exist), *calibration parsing* (where the cameras are) and
*pixel decoding* (how a frame is read) differ per dataset; :class:`TemporalSceneDataset`
owns the snippet index, the train/val split, the ``__getitem__`` loop and the viewer's
``load_sequence_views``.

Subclass contract
-----------------
* ``_discover() -> list[record]`` — one record per trajectory (any type). The base then
  applies the deterministic split unless ``_should_split()`` is False.
* ``_n_frames_of(record) -> int``, ``_scene_name(record, t_start) -> str``,
  ``_record_label(record) -> str``.
* ``_open_scene(record) -> SceneView`` — a per-sample handle bundling cameras and pixel
  access; expensive readers live on it and die with the sample.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from einops import repeat
from torch import Tensor
from torch.utils.data import Dataset

from .config import DatasetCfg, HorizonFn, Stage, StepFn
from .shims import apply_augmentation_shim, apply_crop_shim
from .view_sampler import ViewSampler

log = logging.getLogger(__name__)


def resize_frames(x: Tensor, h: int, w: int, mode: str) -> Tensor:
    """Resize ``(V, C, h0, w0)`` to ``(V, C, h, w)`` (no-op if equal).

    ``mode="bilinear"`` for colour, ``"nearest"`` for depth/masks.
    """
    if x.shape[-2] == h and x.shape[-1] == w:
        return x
    kw = {} if mode == "nearest" else {"align_corners": False}
    return torch.nn.functional.interpolate(x, size=(h, w), mode=mode, **kw)


def split_records(records: list, stage: str, val_frac: float) -> list:
    """Deterministic seed-42 record split; fewer than 5 records are never split."""
    n = len(records)
    if n < 5:
        return records
    order = np.arange(n)
    np.random.RandomState(42).shuffle(order)
    n_val = max(1, int(round(n * val_frac)))
    keep = sorted(order[n_val:] if stage == "train" else order[:n_val])
    return [records[i] for i in keep]


def loader_horizon(
    cfg: DatasetCfg,
    stage: str,
    validation: bool,
    step_fn: StepFn | None,
    horizon_fn: HorizonFn | None,
    n_full_steps: int,
) -> int:
    """Number of timesteps to decode for one sample (horizon-aware loading).

    Returns ``n_full_steps`` unless horizon-aware loading is on, the stage is training
    and both callables are given; then ``n_step_state + horizon_fn(step + margin,
    n_pred_full)`` clamped to the full horizon. ``stage`` and ``validation`` are passed
    separately because they can disagree (genesis ``overfit_to_scene``).
    """
    if (
        not cfg.horizon_aware_loading
        or validation
        or stage != "train"
        or step_fn is None
        or horizon_fn is None
    ):
        return n_full_steps
    step = int(step_fn()) + cfg.horizon_load_margin_steps
    n_pred_full = n_full_steps - cfg.n_step_state
    n_pred = int(horizon_fn(step, n_pred_full))
    return cfg.n_step_state + max(0, min(n_pred, n_pred_full))


def add_workspace(cfg: DatasetCfg, example: dict) -> None:
    """Emit the config's workspace AABB into an example dict (no-op when unset)."""
    if cfg.workspace_min is not None and cfg.workspace_max is not None:
        example["workspace_min"] = torch.tensor(cfg.workspace_min, dtype=torch.float32)
        example["workspace_max"] = torch.tensor(cfg.workspace_max, dtype=torch.float32)


class SceneView:
    """Per-sample handle for one scene/trajectory: cameras + pixel access.

    Subclasses set ``cam_names`` (after camera filtering) and ``intrinsics`` ``(V, 3, 3)``
    normalised in ``__init__`` and implement the methods below.
    """

    cam_names: list[str]
    intrinsics: Tensor  # (V, 3, 3) normalised [0, 1]

    def extrinsics(self, frame_idx: int) -> Tensor:
        """c2w ``(V, 4, 4)`` at ``frame_idx`` (clamped internally)."""
        raise NotImplementedError

    def load_views(
        self,
        cam_ixs,
        frame_idx: int,
        out_hw: tuple[int, int],
        *,
        depth: bool = False,
        side: str = "target",
    ) -> dict[str, Tensor]:
        """Pixels for the selected cameras at one frame.

        Returns at least ``{"image": (V, 3, H, W) float in [0, 1]}``; optionally ``depth``
        ``(V, 1, H, W)`` metres and ``state_mask`` / ``static_float`` / ``seg``
        ``(V, 1, H, W)``. ``side`` is "context" / "target" / "viewer".
        """
        raise NotImplementedError

    def load_actions(self, frame_indices: list[int]) -> dict[str, Tensor] | None:
        """Actions for the snippet, each ``(T, 1, P, 4)`` xyz + presence; or None."""
        return None

    def robot_state(self, frame_idx: int) -> dict | None:
        """Articulated-agent pose at one frame (viewer overlay), or None.

        Image-fitted poses carry ``source='image_fit'`` to distinguish them from
        recorded joint measurements. They apply only to their calibrated frame.
        """
        return None


class TemporalSceneDataset(Dataset):
    """Snippet indexing + the generic context/target loop."""

    cfg: DatasetCfg
    stage: Stage
    view_sampler: ViewSampler
    #: Whether GT depth is in metres; emitted per view as ``depth_is_metric``.
    DEPTH_IS_METRIC: bool = True

    def __init__(
        self,
        cfg: DatasetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        *,
        step_fn: StepFn | None = None,
        horizon_fn: HorizonFn | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.validation = stage in ("val", "test")
        self.view_sampler = view_sampler
        self.step_fn = step_fn
        self.horizon_fn = horizon_fn

        self.near = torch.tensor(cfg.near, dtype=torch.float32)
        self.far = torch.tensor(cfg.far, dtype=torch.float32)
        self.depth_is_metric = torch.tensor(
            1.0 if type(self).DEPTH_IS_METRIC else 0.0, dtype=torch.float32
        )

        self.records = self._discover()
        if not self.records:
            raise FileNotFoundError(
                f"{type(self).__name__}: no records discovered (check the dataset root)"
            )
        if self._should_split():
            self.records = self._split(self.records, stage)

        self.n_full_steps = self._n_full_steps()
        self._index: list[tuple[int, int]] = [
            (r, t)
            for r, rec in enumerate(self.records)
            for t in self._snippet_starts(rec, self._n_frames_of(rec))
        ]

        self.view_sampler.update_camera_indices(cfg.camera_ixs_allowed)
        log.info(
            "%s (%s): %d records, %d snippets, image_shape=%s, n_step_state/predict=%d/%d, "
            "speedup=%d",
            type(self).__name__,
            self.stage,
            len(self.records),
            len(self._index),
            cfg.image_shape,
            cfg.n_step_state,
            cfg.n_step_predict,
            cfg.speedup,
        )

    # ------------------------------------------------------------------ #
    # Subclass hooks
    # ------------------------------------------------------------------ #

    def _discover(self) -> list:
        raise NotImplementedError

    def _open_scene(self, rec) -> SceneView:
        raise NotImplementedError

    def _n_frames_of(self, rec) -> int:
        raise NotImplementedError

    def _scene_name(self, rec, t_start: int) -> str:
        raise NotImplementedError

    def _record_label(self, rec) -> str:
        return self._scene_name(rec, 0)

    def _should_split(self) -> bool:
        return getattr(self.cfg, "split_mode", "scene") != "none"

    def _val_frac(self) -> float:
        # dyn holds out 10% of trajectories, nvs 15%
        return 0.10 if self.cfg.n_step_predict > 0 else 0.15

    def _split(self, records: list, stage: str) -> list:
        return split_records(records, stage, self._val_frac())

    def _n_full_steps(self) -> int:
        if self.validation and self.cfg.n_step_predict != 0:
            return self.cfg.n_step_state + self.cfg.val_n_step_predict
        return self.cfg.n_step_state + self.cfg.n_step_predict

    def _snippet_starts(self, rec, n_frames: int):
        snippet_len = self.n_full_steps * self.cfg.speedup
        return range(0, n_frames - snippet_len + 1, self.cfg.speedup)

    # ------------------------------------------------------------------ #
    # Viewer access
    # ------------------------------------------------------------------ #

    def record_labels(self) -> list[str]:
        return [self._record_label(rec) for rec in self.records]

    def record_n_frames(self, rec_ix: int) -> int:
        return self._n_frames_of(self.records[rec_ix])

    def load_sequence_views(
        self,
        rec_ix: int,
        frame_idx: int,
        target_hw: tuple[int, int] | None = None,
        with_depth: bool = False,
        with_robot: bool = False,
        *,
        scene_view: SceneView | None = None,
    ) -> dict:
        """ALL cameras of one (record, timestep).

        Returns ``image (V,3,H,W)``, ``extrinsics (V,4,4)`` c2w, ``intrinsics (V,3,3)``
        normalised, ``index`` (camera names), ``scene``; ``depth`` when asked for and
        available; ``actions`` (``(1, P, 4)`` per key) when the dataset has them;
        ``robot`` when ``with_robot`` and the dataset records agent state.
        A viewer may supply its current ``scene_view`` to reuse open video readers;
        the caller owns that handle and must serialize concurrent reads.
        """
        rec = self.records[rec_ix]
        sv = self._open_scene(rec) if scene_view is None else scene_view
        hw = tuple(target_hw) if target_hw is not None else tuple(self.cfg.image_shape)
        all_ixs = torch.arange(len(sv.cam_names))
        views = sv.load_views(all_ixs, frame_idx, hw, depth=with_depth, side="viewer")
        out = {
            "image": views["image"],
            "extrinsics": sv.extrinsics(frame_idx),
            "intrinsics": sv.intrinsics,
            "index": list(sv.cam_names),
            "scene": self._record_label(rec),
        }
        if "depth" in views:
            out["depth"] = views["depth"]
        if with_robot:
            robot = sv.robot_state(frame_idx)
            if robot is not None:
                out["robot"] = robot
        try:
            # Best-effort: a frame without labels yields no actions rather than an error.
            actions = sv.load_actions([frame_idx])
        except Exception:
            actions = None
        if actions:
            out["actions"] = {k: v[0] for k, v in actions.items()}
        return out

    # ------------------------------------------------------------------ #
    # The generic sample loop
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        rec_ix, t_start = self._index[idx]
        rec = self.records[rec_ix]
        cfg = self.cfg
        speedup = cfg.speedup
        scene_name = self._scene_name(rec, t_start)
        frame_indices = [t_start + i * speedup for i in range(self.n_full_steps)]

        # Horizon-aware loading: decode only the first n_keep timesteps; the loop still
        # samples views for the full horizon so the RNG stream is unchanged.
        n_keep = loader_horizon(
            cfg, self.stage, self.validation, self.step_fn, self.horizon_fn, self.n_full_steps
        )

        sv = self._open_scene(rec)
        intrinsics = sv.intrinsics
        extr_repr = sv.extrinsics(frame_indices[0])
        out_hw = tuple(cfg.image_shape)
        with_depth = bool(getattr(cfg, "with_depth", False))

        contexts: list[dict] = []
        targets: list[dict] = []
        n_sp = cfg.n_step_state_predict
        # (context, target) pairs per state step; prediction steps cycle the last n_sp.
        state_cameras: list[tuple] = []
        include_extras = self.view_sampler.cfg.target_includes_context

        for t_idx, frame_idx in enumerate(frame_indices):
            if not cfg.consistent_cameras:
                ci, ti = self.view_sampler.sample(
                    scene_name, extr_repr, intrinsics, camera_group_ix=t_idx, cam_names=sv.cam_names
                )
            elif t_idx < cfg.n_step_state:
                ci, ti = self.view_sampler.sample(
                    scene_name, extr_repr, intrinsics, camera_group_ix=t_idx, cam_names=sv.cam_names
                )
                if not include_extras:
                    ti = ci
                state_cameras.append((ci, ti))
            else:
                pred_offset = (t_idx - cfg.n_step_state) % n_sp
                ci, ti = state_cameras[-(n_sp - pred_offset)]

            extr_t = sv.extrinsics(frame_idx)

            if t_idx < cfg.n_step_state:
                contexts.append(
                    {
                        **sv.load_views(ci, frame_idx, out_hw, depth=with_depth, side="context"),
                        "extrinsics": extr_t[ci],
                        "intrinsics": intrinsics[ci],
                        "near": repeat(self.near, "-> v", v=len(ci)),
                        "far": repeat(self.far, "-> v", v=len(ci)),
                        "depth_is_metric": repeat(self.depth_is_metric, "-> v", v=len(ci)),
                        "index": ci,
                    }
                )

            # During validation, keep the target views fixed across the rollout.
            if self.validation and t_idx >= cfg.n_step_state:
                ti = targets[-1]["index"]

            if t_idx >= n_keep:
                continue

            targets.append(
                {
                    **sv.load_views(ti, frame_idx, out_hw, depth=with_depth, side="target"),
                    "extrinsics": extr_t[ti],
                    "intrinsics": intrinsics[ti],
                    "near": repeat(self.near, "-> v", v=len(ti)),
                    "far": repeat(self.far, "-> v", v=len(ti)),
                    "depth_is_metric": repeat(self.depth_is_metric, "-> v", v=len(ti)),
                    "index": ti,
                }
            )

        contexts = {k: torch.stack([c[k] for c in contexts]) for k in contexts[0].keys()}
        targets = {k: torch.stack([t[k] for t in targets]) for k in targets[0].keys()}
        example = {"scene": scene_name, "context": contexts, "target": targets}
        add_workspace(cfg, example)

        actions = sv.load_actions(frame_indices[:n_keep])
        if actions:
            example["actions"] = actions

        if self.stage == "train" and cfg.augment:
            example = apply_augmentation_shim(example, self.view_sampler.generator)
        example = apply_crop_shim(example, out_hw)
        return example
