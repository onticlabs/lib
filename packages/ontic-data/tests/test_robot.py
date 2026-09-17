"""Robot kinematics helpers: pure-numpy parts always, mujoco parts when installed."""

import importlib
import sys

import numpy as np
import pytest

from ontic_data import robot


def _has(name):
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def test_mujoco_missing_names_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "mujoco", None)
    with pytest.raises(ImportError, match=r"ontic-data\[robot\]"):
        robot.RobotKinematics.from_mjcf(tmp_path / "robot.xml")
    assert robot.robot_model_available() is False


def test_duo_joint_angles_fans_out_gripper_linkage():
    q = robot.duo_joint_angles({"left_arm": [0.1] * 7, "left_gripper": [0.3, 0.4], "base": []})
    assert q["left_fr3_joint1"] == 0.1 and q["left_fr3_joint7"] == 0.1
    for part in ("driver", "spring_link", "follower"):
        assert q[f"left_gripper_left_{part}_joint"] == 0.3
        assert q[f"left_gripper_right_{part}_joint"] == 0.4
    assert not any(k.startswith("right_") for k in q)


def test_to_action_tensor_layout():
    out = robot.to_action_tensor(np.zeros((3, 5, 3)))
    assert out.shape == (3, 1, 5, 4) and np.all(out[..., 3] == 1)


def test_describe_unavailable_without_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv(robot.ENV_REPO, str(tmp_path))
    assert robot.robotics_repo() is None
    assert "no robotics checkout" in robot.describe_unavailable()


TWO_LINK = """
<mujoco>
  <worldbody>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1"/>
      <geom type="box" size="0.05 0.05 0.05"/>
      <body name="link2" pos="1 0 0">
        <joint name="j2" type="hinge" axis="0 0 1"/>
        <geom type="box" size="0.05 0.05 0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.mark.skipif(not _has("mujoco"), reason="mujoco not installed")
def test_sample_action_points_follow_forward_kinematics():
    import mujoco

    kin = robot.RobotKinematics.from_model(mujoco.MjModel.from_xml_string(TWO_LINK))
    assert kin.movable_links == ["link1", "link2"]
    q = np.array([[0.0, 0.0], [np.pi / 2, 0.0]])
    pts, links = robot.sample_action_points(kin, q, n_per_link=2)
    assert pts.shape == (2, 4, 3) and list(links) == ["link1", "link1", "link2", "link2"]
    # link1 spans (0,0,0)->(1,0,0); rotating j1 by 90deg swings its far end to (0,1,0)
    assert np.allclose(pts[0, :2], [[0, 0, 0], [1, 0, 0]], atol=1e-6)
    assert np.allclose(pts[1, 1], [0, 1, 0], atol=1e-6)
    tracks, _ = robot.sample_action_points(kin, q, n_per_link=3, along="uniform", seed=1)
    again, _ = robot.sample_action_points(kin, q, n_per_link=3, along="uniform", seed=1)
    assert np.array_equal(tracks, again)
