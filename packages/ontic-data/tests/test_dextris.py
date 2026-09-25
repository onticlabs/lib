"""Shared DEXTRIS loading for nested hand captures and flat robot recordings."""

import json
import hashlib
import subprocess
from types import SimpleNamespace

import pytest
import torch

from ontic_data import DATASETS
from ontic_data.datasets import dextris


def _sample(root, sid, *, hands=False):
    sample = root / sid
    sample.mkdir(parents=True)
    cameras = {
        cam: {
            "camera_matrix": [[8, 0, 4], [0, 6, 3], [0, 0, 1]],
            "R": torch.eye(3).tolist(),
            "t": [i * 0.1, 0, 1],
            "dist_coeffs": [0] * 5,
            "image_size": [6, 8],
        }
        for i, cam in enumerate(dextris.CAMERA_NAMES)
    }
    (sample / "calibration_result.json").write_text(json.dumps({"cameras": cameras}))
    for cam in dextris.CAMERA_NAMES:
        (sample / f"{sid}_{cam}.mp4").touch()
    if hands:
        (sample / f"{sid}_hand_tracking.json").write_text("{}")
    return sample


def test_discovery_keeps_nested_hand_requirements_and_supports_flat_rgb(tmp_path):
    nested = _sample(tmp_path / "pickUp", "pickUp_001", hands=True)
    _sample(tmp_path / "pickUp", "pickUp_002")
    flat = _sample(tmp_path, "demo_001")
    _sample(tmp_path, "alignment_001")
    incomplete = _sample(tmp_path, "demo_002")
    (incomplete / "demo_002_P07.mp4").unlink()
    (tmp_path / "empty").mkdir()

    assert dextris.discover_samples(tmp_path, ["pickUp"]) == [nested]
    assert dextris.discover_samples(tmp_path, flat_layout=True) == []
    assert dextris.discover_samples(
        tmp_path, ["demo"], flat_layout=True, require_hand_tracking=False
    ) == [flat]


@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_robot_dataset_loads_calibrated_rgb_without_hands_or_split(tmp_path, monkeypatch, stage):
    for i in range(6):
        _sample(tmp_path, f"demo_{i:03d}")

    class Reader:
        def __init__(self, path, backend):
            self.path = str(path)

        def __len__(self):
            return 4

        def get_frames(self, indices):
            # Exercise the real all-camera loading, resize and normalization paths.
            assert indices == [2]
            return torch.full((1, 6, 8, 3), 128, dtype=torch.uint8)

    monkeypatch.setattr(dextris, "has_video_backend", lambda backend: True)
    monkeypatch.setattr(dextris, "VideoReader", Reader)
    cfg = DATASETS["robot-dextris"](root=str(tmp_path), image_shape=[3, 4])
    ds = cfg.build(stage)
    assert isinstance(ds, dextris.DatasetDextris)
    assert ds.record_labels() == [f"demo_{i:03d}" for i in range(6)]
    assert ds.record_n_frames(0) == 4
    assert cfg.workspace_min is None and cfg.workspace_max is None
    frame = ds.load_sequence_views(0, 2, with_depth=True, with_robot=True)
    assert frame["image"].shape == (8, 3, 3, 4)
    assert torch.allclose(frame["image"], torch.full_like(frame["image"], 128 / 255))
    assert frame["index"] == dextris.CAMERA_NAMES
    assert torch.allclose(
        frame["intrinsics"][0], torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]])
    )
    assert torch.allclose(frame["extrinsics"][1, :3, 3], torch.tensor([-0.1, 0, -1.0]))
    assert not {"depth", "actions", "robot"} & frame.keys()


def test_metadata_reuses_counts_without_opening_videos_and_preserves_failed_rebuild(
    tmp_path, monkeypatch
):
    good = _sample(tmp_path, "demo_001")
    bad = _sample(tmp_path, "demo_002")

    def probe(command, **kwargs):
        if command[-1].endswith("demo_002_P07.mp4"):
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout=json.dumps({"streams": [{"nb_frames": "4"}]}))

    monkeypatch.setattr(dextris.subprocess, "run", probe)
    cfg = DATASETS["robot-dextris"](root=str(tmp_path))
    path = tmp_path / dextris.META_CSV_NAME
    rows = dextris.write_metadata(cfg, path)
    assert [row["sample_id"] for row in rows] == [good.name]
    original = path.read_bytes()

    def no_probe(*args, **kwargs):
        pytest.fail("cached metadata must avoid opening or probing any video")

    monkeypatch.setattr(dextris, "has_video_backend", lambda backend: True)
    monkeypatch.setattr(dextris.DatasetDextris, "_probe_n_frames", no_probe)
    ds = cfg.build("val")
    assert ds.record_labels() == [good.name] and ds.record_n_frames(0) == 4
    assert len(ds) == 4

    (good / f"{good.name}_P07.mp4").unlink()
    (bad / f"{bad.name}_P00.mp4").unlink()
    with pytest.raises(ValueError, match="index was not written"):
        dextris.write_metadata(cfg, path)
    assert path.read_bytes() == original


def test_robot_alignment_is_bound_to_its_frame_and_camera_calibration(tmp_path):
    sample = _sample(tmp_path, "demo_fit")
    calibration = sample / "calibration_result.json"
    alignment = {
        "schema_version": 1,
        "robot_model": "franka_duo",
        "calibration_sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "frame_index": 0,
        "base_pose": [-0.6, 0.0, -0.4, 1.0, 0.0, 0.0, 0.0],
        "qpos": {"left_arm": [0.1] * 7, "right_arm": [-0.1] * 7},
    }
    path = sample / "robot_alignment.json"
    path.write_text(json.dumps(alignment))
    ds = SimpleNamespace(
        cfg=dextris.RobotDextrisDatasetCfg(root=str(tmp_path)),
        _sample_dir=lambda rec: sample,
        _calib_for=lambda directory: dextris.parse_calibration(calibration),
    )

    def view():
        return dextris.DextrisSceneView(ds, {"sample_id": sample.name})

    scene = view()
    state = scene.robot_state(0)
    assert state["source"] == "image_fit"
    assert state["base_pose"] == alignment["base_pose"]
    assert state["qpos"] == alignment["qpos"]
    assert scene.robot_state(1) is None  # Never freeze an estimated pose over a video.
    assert scene.robot_state(0) == state

    alignment["base_pose"][3] = 0.0
    path.write_text(json.dumps(alignment))
    with pytest.raises(ValueError, match="Invalid robot pose"):
        view().robot_state(0)

    alignment["calibration_sha256"] = "a-different-world-frame"
    path.write_text(json.dumps(alignment))
    assert view().robot_state(0) is None

    ds.cfg.load_robot_alignment = False
    assert view().robot_state(0) is None
