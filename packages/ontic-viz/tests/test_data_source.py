"""Data source over a fake registry entry (no real dataset)."""

from __future__ import annotations

from dataclasses import dataclass, field

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
    assert set(DEFAULT_ROOTS) == {"dextris", "hocap", "taco", "genesis", "physinone", "synthrobot"}


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
