"""Data source over a fake registry entry (no real dataset)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
import weakref

import pytest
import torch

from ontic_viz.backbone_viewer.data_source import (
    DEFAULT_ROOTS,
    Frame,
    GenericSource,
    build_source,
    dataset_has_gt_depth,
    dataset_names,
    present_hands_from_action,
)


class _FakeTemporalDataset:
    def __init__(self, labels, n_frames=4, with_depth_ok=True):
        self._labels = labels
        self._n = n_frames
        self._with_depth_ok = with_depth_ok
        self.calls = []

    def record_labels(self):
        return list(self._labels)

    def record_n_frames(self, rec_ix):
        return self._n

    def load_sequence_views(
        self, rec_ix, frame_idx, target_hw=None, with_depth=False, with_robot=False
    ):
        self.calls.append((rec_ix, frame_idx, with_depth, with_robot))
        if with_depth and not self._with_depth_ok:
            raise NotImplementedError("undecodable depth")
        v = 3
        out = {
            "image": torch.rand(v, 3, 8, 8, dtype=torch.float64),
            "extrinsics": torch.eye(4).expand(v, 4, 4).clone(),
            "intrinsics": torch.eye(3).expand(v, 3, 3).clone(),
            "index": [f"cam{i}" for i in range(v)],
            "scene": self._labels[rec_ix],
            "actions": {"left_hand": torch.cat([torch.rand(1, 21, 3), torch.ones(1, 21, 1)], -1)},
            "robot": {"qpos": {}, "base_pose": None},
        }
        if with_depth:
            out["depth"] = torch.rand(v, 1, 8, 8)
        return out


@dataclass
class _FakeCfg:
    root: str | None = None
    built: list = field(default_factory=list)
    stage_seen: str | None = None

    def build(self, stage, *, step_fn=None):
        self.stage_seen = stage
        return _FakeTemporalDataset(["a", "b", "b"])


@dataclass
class _FakeRootsCfg:
    roots: list | None = None
    root_depths: list | None = None

    def build(self, stage, *, step_fn=None):
        return _FakeTemporalDataset(["s"])


def test_dataset_names_and_hints():
    assert dataset_names({"x": _FakeCfg, "y": _FakeCfg}) == ["x", "y"]
    assert dataset_has_gt_depth("hocap") and not dataset_has_gt_depth("taco")
    assert dataset_has_gt_depth("unknown") is False
    assert set(DEFAULT_ROOTS) == {
        "dextris",
        "robot-dextris",
        "hocap",
        "taco",
        "genesis",
        "physinone",
        "synthrobot",
    }


def test_build_source_from_fake_registry(tmp_path):
    registry = {"fake": _FakeCfg, "genesis": _FakeRootsCfg}
    src = build_source("fake", root=str(tmp_path), stage="test", registry=registry)
    assert isinstance(src, GenericSource) and src.name == "fake"
    assert src.list_trajectories() == ["000_a", "001_b", "002_b"]  # duplicates disambiguated
    assert src.num_timesteps(1) == 4
    frame = src.get_frame(1, 2)
    assert isinstance(frame, Frame)
    assert frame.images.dtype == torch.float32 and frame.images.shape == (3, 3, 8, 8)
    assert frame.cam_names == ["cam0", "cam1", "cam2"]
    assert frame.hands.shape == (1, 21, 3) and frame.depth is None and frame.robot is not None
    assert src.ds.calls[-1] == (1, 2, False, True)
    with pytest.raises(ValueError):
        build_source("nope", registry=registry)


def test_robot_dextris_viewer_registration_and_root_override(tmp_path, monkeypatch):
    from ontic_data import DATASETS
    from ontic_viz.backbone_viewer.cli import build_parser
    from ontic_viz.backbone_viewer.config import dataset_defaults

    names = dataset_names()
    assert "robot-dextris" in names
    args = build_parser(names).parse_args(["--robot-dextris-root", str(tmp_path)])
    seen = []

    def build(cfg, stage):
        seen.append((cfg.root, cfg.flat_layout, cfg.load_hand_poses, stage))
        return _FakeTemporalDataset(["demo_001"])

    monkeypatch.setattr(DATASETS["robot-dextris"], "build", build)
    src = build_source("robot-dextris", root=args.robot_dextris_root)
    assert src.name == "robot-dextris" and src.list_trajectories() == ["demo_001"]
    assert seen == [(str(tmp_path), True, False, "val")]
    assert not dataset_has_gt_depth("robot-dextris")
    assert not dataset_defaults()["robot-dextris"].crop_enabled


def test_genesis_root_maps_to_roots_list():
    registry = {"genesis": _FakeRootsCfg}
    src = build_source("genesis", root="/data/g", registry=registry)
    assert src.list_trajectories() == ["s"]
    cfg = _FakeRootsCfg(roots=["/data/g"], root_depths=[2])
    assert cfg.roots == ["/data/g"]


def test_depth_requested_and_undecodable_fallback():
    src = GenericSource("d", _FakeTemporalDataset(["a"]))
    assert src.get_frame(0, 0, with_depth=True).depth.shape == (3, 1, 8, 8)
    src2 = GenericSource("d", _FakeTemporalDataset(["a"], with_depth_ok=False))
    frame = src2.get_frame(0, 0, with_depth=True)
    assert frame.depth is None and src2.ds.calls[-1][2] is False


def test_present_hands_filters_absent_and_collapsed():
    present = torch.cat([torch.rand(1, 21, 3), torch.ones(1, 21, 1)], -1)
    absent = present.clone()
    absent[..., 3] = 0.0
    collapsed = torch.cat([-torch.ones(1, 21, 3), torch.ones(1, 21, 1)], -1)
    hands = present_hands_from_action(
        {"left_hand": present, "right_hand": absent, "left_tcp": present}
    )
    assert hands.shape == (1, 21, 3)
    assert present_hands_from_action({"right_hand": collapsed}) is None
    assert present_hands_from_action({}) is None


def test_source_reuses_readers_and_releases_previous_trajectory():
    from ontic_data.temporal import SceneView, TemporalSceneDataset

    opened = []

    class View(SceneView):
        def __init__(self, rec):
            self.rec = rec
            self.cam_names = ["P00"]
            self.intrinsics = torch.eye(3)[None]

        def extrinsics(self, frame_idx):
            return torch.eye(4)[None]

        def load_views(self, cam_ixs, frame_idx, out_hw, **kwargs):
            return {"image": torch.full((1, 3, 2, 2), (self.rec * 10 + frame_idx) / 100)}

    class Dataset(TemporalSceneDataset):
        def __init__(self):
            self.records = [0, 1]
            self.cfg = SimpleNamespace(image_shape=[2, 2])

        def _record_label(self, rec):
            return str(rec)

        def _open_scene(self, rec):
            view = View(rec)
            opened.append(weakref.ref(view))
            return view

    source = GenericSource("test", Dataset())
    first = source.get_frame(0, 1)
    second = source.get_frame(0, 2)
    assert len(opened) == 1 and opened[0]() is not None
    assert not torch.equal(first.images, second.images)
    third = source.get_frame(1, 1)
    assert len(opened) == 2 and opened[0]() is None
    assert torch.allclose(third.images, torch.full_like(third.images, 0.11))
