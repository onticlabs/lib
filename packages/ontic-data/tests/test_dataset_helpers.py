"""Pure-python helpers of the dataset modules (no data on disk)."""

import numpy as np
import pytest
import torch

from ontic_data._np import normalize_intrinsics_np, w2c_to_c2w_np
from ontic_data.datasets import dextris, hocap, physinone, synthrobot


def test_np_adapters_match_conventions():
    K = np.array([[500.0, 0, 320], [0, 400, 240], [0, 0, 1]])
    Kn = normalize_intrinsics_np(K, 640, 480)
    assert Kn.dtype == np.float32 and np.allclose(
        Kn, [[500 / 640, 0, 0.5], [0, 400 / 480, 0.5], [0, 0, 1]]
    )
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    t = np.array([1.0, 2.0, 3.0])
    c2w = w2c_to_c2w_np(R, t)
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = R, t
    assert c2w.dtype == np.float32 and np.allclose(c2w @ w2c, np.eye(4), atol=1e-6)


def test_synthrobot_packed16_round_trip_and_sniff():
    depth = np.linspace(0.05, 5.0, 64).reshape(8, 8)
    rgb = synthrobot.encode_depth_rgb(depth, 0.05, 5.0)
    back = synthrobot.decode_depth_rgb(rgb, 0.05, 5.0)
    assert np.allclose(back, depth, atol=(5.0 - 0.05) / 65535 + 1e-6)
    assert synthrobot.detect_depth_encoding_rgb(rgb) == synthrobot.DEPTH_ENCODING_PACKED16
    hue = np.stack([rgb[..., 0], rgb[..., 1], rgb[..., 0]], -1)
    assert synthrobot.detect_depth_encoding_rgb(hue) == synthrobot.DEPTH_ENCODING_HUE


def test_synthrobot_hue_spec_reads_range_and_rejects_drift():
    assert synthrobot.HueLogSpec.from_obs_scene({}) is None
    spec = synthrobot.HueLogSpec.from_obs_scene(
        {"depth_encoding": {"name": "hue-log", "range_m": [0.1, 10]}}
    )
    assert spec == synthrobot.HueLogSpec(0.1, 10.0)
    with pytest.raises(ValueError, match="disagrees"):
        synthrobot.HueLogSpec.from_obs_scene({"depth_encoding": {"name": "hue-log", "guard": 16}})


def test_synthrobot_video_name_and_csv_round_trip(tmp_path):
    info = synthrobot.parse_video_name("episode_00000012_wrist_camera_l_depth_batch_1_of_2.mp4")
    assert info == {
        "episode": 12,
        "camera": "wrist_camera_l",
        "depth": True,
        "batch": 1,
        "n_batches": 2,
    }
    assert synthrobot.parse_video_name("episode_00000012_1-1_batch_1_of_2.mp4")["camera"] == "1-1"
    assert synthrobot.parse_video_name("trajectories_batch_1_of_1.h5") is None
    rows = [
        {
            "rel_dir": "franka_duo/house_0", "h5_name": "trajectories_batch_1_of_1.h5",
            "batch": 1, "n_batches": 1, "episode_offset": 0, "n_traj": 2, "n_frames": [87, 73],
            "cameras": ["ring_00", "1-1"], "depth_cameras": ["ring_00"],
            "depth_min_m": 0.05, "depth_max_m": 5.0, "depth_encoding": None,
            "task_type": "pick", "success": [True, False],
        },
        {
            "rel_dir": "pert/house_1", "h5_name": "trajectories_batch_1_of_1.h5",
            "batch": 1, "n_batches": 1, "episode_offset": 3, "n_traj": 1, "n_frames": [10],
            "cameras": ["1-1"], "depth_cameras": [], "depth_min_m": 0.05, "depth_max_m": 5.0,
            "depth_encoding": {"min_m": 0.05, "max_m": 5.0}, "task_type": "", "success": [True],
        },
    ]  # fmt: skip
    synthrobot.write_recordings_csv(rows, tmp_path / "meta.csv")
    assert synthrobot.read_recordings_csv(tmp_path / "meta.csv") == rows


def test_synthrobot_tcp_actions_and_content_crop():
    pose = np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]])
    frame = synthrobot.build_tcp_action_tensor(pose, [0, 5], mode="frame", axis_len=0.1)
    assert frame.shape == (2, 1, 4, 4)
    assert torch.allclose(frame[1, 0, 1, :3], torch.tensor([1.1, 2.0, 3.0]))
    assert synthrobot.build_tcp_action_tensor(pose, [0], mode="tcp").shape == (1, 1, 1, 4)
    K = np.array([[600.0, 0, 320], [0, 600, 180], [0, 0, 1]])
    assert synthrobot.content_hw_from_intrinsics(K, 368, 640) == (360, 640)
    assert np.allclose(synthrobot.normalized_K_from_pixel_K(K)[:2, 2], 0.5)
    R = synthrobot.quat_wxyz_to_matrix([0.7071068, 0, 0, 0.7071068])
    assert np.allclose(synthrobot._matrix_to_quat_wxyz(R), [0.7071068, 0, 0, 0.7071068], atol=1e-6)


def test_hocap_hand_actions_presence():
    kp = np.ones((3, 2, 21, 3))
    out = hocap.build_hocap_hand_actions(kp, ["left"])
    assert out["left_hand"].shape == (3, 1, 21, 4) and torch.all(out["left_hand"][..., 3] == 1)
    assert torch.all(out["right_hand"] == 0)
    with pytest.raises(ValueError):
        hocap.build_hocap_hand_actions(np.ones((3, 21, 3)), ["left"])


def test_dextris_hand_action_tensor_zero_fills_missing():
    kp = np.full((4, 21, 3), np.nan)
    kp[1] = 1.0
    present = np.array([False, True, False, False])
    out = dextris.build_hand_action_tensor(kp, present, [0, 1, 9])
    assert out.shape == (3, 1, 21, 4)
    assert torch.all(out[0] == 0) and torch.all(out[1, 0, :, 3] == 1) and not torch.isnan(out).any()


def test_physinone_geometry_helpers():
    K = physinone.intrinsics_from_fov(np.pi / 2, 200, 100)
    assert np.isclose(K[0, 0], 0.5) and np.isclose(K[1, 1], 1.0)
    c2w = physinone.blender_c2w_to_opencv(np.eye(4))
    assert np.allclose(c2w, np.diag([1, -1, -1, 1]))
    series = np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]])
    act = physinone.build_object_action_tensor(series, [0, 1, 7], 0.01)
    assert act.shape == (3, 1, 1, 4)
    assert torch.allclose(act[0, 0, 0], torch.tensor([0.01, 0.02, 0.03, 1.0]))
    assert torch.all(act[1] == 0)
