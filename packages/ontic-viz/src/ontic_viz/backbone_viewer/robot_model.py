"""Franka FR3 Duo link geometry posed from recorded joint angles, for the overlay.

The cell is composed by ``ontic_data.robot.build_franka_duo_model`` from the sibling
``onticlabs/robotics`` checkout; nothing is vendored here. FK runs in the cell frame
(origin at the mount base); ``obs/extra/robot_base_pose`` (``xyz`` + ``wxyz``) maps
cell -> world. Requires ``mujoco`` (``ontic-viz[robot]``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch

from ontic_lib.transforms.rotations import matrix_to_quaternion, quaternion_to_matrix


def robot_model_available() -> bool:
    from ontic_data.robot import robot_model_available as available

    return available()


def describe_unavailable() -> str:
    from ontic_data.robot import describe_unavailable as describe

    return describe()


@dataclass(frozen=True)
class LinkGeom:
    """One rigid mesh of the robot in its own geom frame; ``name`` keys
    :meth:`DuoRobotModel.geom_world_poses`."""

    name: str
    vertices: np.ndarray  # (N, 3) float32
    faces: np.ndarray  # (F, 3) int32
    color: tuple[float, float, float]


def _geom_rgb(model, g: int) -> np.ndarray:
    mat = int(model.geom_matid[g])
    if mat >= 0:
        return np.asarray(model.mat_rgba[mat][:3], dtype=np.float64)
    return np.asarray(model.geom_rgba[g][:3], dtype=np.float64)


def pose7_to_matrix(pose) -> np.ndarray:
    """``[x, y, z, qw, qx, qy, qz]`` -> ``(4, 4)`` (quaternion normalised first)."""
    p = np.asarray(pose, dtype=np.float64).reshape(7)
    q = torch.as_tensor(p[3:7])
    q = q / q.norm().clamp_min(1e-12)
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix(q).numpy()
    matrix[:3, 3] = p[:3]
    return matrix


def matrix_to_wxyz(rotation: np.ndarray) -> np.ndarray:
    return matrix_to_quaternion(torch.as_tensor(np.asarray(rotation), dtype=torch.float64)).numpy()


class DuoRobotModel:
    """The FR3 Duo cell: joint angles in, posed link geometry out.

    ``with_worktable`` adds the cell's table (off: recorded scenes place the arms over
    house furniture); ``visual_only`` drops collision hulls (MuJoCo group 3).
    """

    def __init__(
        self,
        repo: Path | None = None,
        with_worktable: bool = False,
        visual_only: bool = True,
    ) -> None:
        try:
            import mujoco
        except ImportError as e:
            raise ImportError("DuoRobotModel requires mujoco; install ontic-viz[robot]") from e
        from ontic_data.robot import build_franka_duo_model

        self._mujoco = mujoco
        self._model = build_franka_duo_model(repo, with_worktable=with_worktable)
        self._data = mujoco.MjData(self._model)
        self._visual_only = visual_only
        self._qadr = {
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, j): int(
                self._model.jnt_qposadr[j]
            )
            for j in range(self._model.njnt)
        }

    @property
    def joint_names(self) -> list[str]:
        mj = self._mujoco
        return [
            mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_JOINT, j) for j in range(self._model.njnt)
        ]

    @cached_property
    def _drawn_geoms(self) -> list[tuple[int, str]]:
        """``(geom id, "<body>/<n>")`` for the mesh geoms we draw, in model order."""
        mj, m = self._mujoco, self._model
        out: list[tuple[int, str]] = []
        seen: dict[str, int] = {}
        for g in range(m.ngeom):
            if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                continue
            if self._visual_only and m.geom_group[g] >= 3:
                continue
            body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "world"
            n = seen.get(body, 0)
            seen[body] = n + 1
            out.append((g, f"{body}/{n}"))
        return out

    @cached_property
    def link_geoms(self) -> list[LinkGeom]:
        m = self._model
        out: list[LinkGeom] = []
        for g, name in self._drawn_geoms:
            mesh = int(m.geom_dataid[g])
            v0, nv = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            f0, nf = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
            out.append(
                LinkGeom(
                    name=name,
                    vertices=np.asarray(m.mesh_vert[v0 : v0 + nv], dtype=np.float32).reshape(-1, 3),
                    faces=np.asarray(m.mesh_face[f0 : f0 + nf], dtype=np.int32).reshape(-1, 3),
                    color=tuple(float(c) for c in _geom_rgb(m, g)),
                )
            )
        return out

    def set_qpos(self, qpos: dict[str, list[float]]) -> None:
        """Write one ``obs/agent/qpos`` reading (named groups) into the model state;
        gripper driver angles are fanned out across each jaw's four-bar linkage."""
        from ontic_data.robot import duo_joint_angles

        self._data.qpos[:] = 0.0
        for joint, value in duo_joint_angles(qpos).items():
            adr = self._qadr.get(joint)
            if adr is not None:
                self._data.qpos[adr] = float(value)
        self._mujoco.mj_forward(self._model, self._data)

    def geom_world_poses(
        self, qpos: dict[str, list[float]], base_pose: np.ndarray | None = None
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """``{geom name: (position (3,), wxyz (4,))}`` in the cell frame, or in world
        when ``base_pose`` (7-vector, ``xyz`` then ``wxyz``) is given."""
        self.set_qpos(qpos)
        d = self._data
        base = pose7_to_matrix(base_pose) if base_pose is not None else np.eye(4)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for g, name in self._drawn_geoms:
            pose = np.eye(4)
            pose[:3, :3] = np.asarray(d.geom_xmat[g]).reshape(3, 3)
            pose[:3, 3] = np.asarray(d.geom_xpos[g])
            pose = base @ pose
            out[name] = (pose[:3, 3].copy(), matrix_to_wxyz(pose[:3, :3]))
        return out

    def tcp_poses(
        self, qpos: dict[str, list[float]], base_pose: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """``{side: (4, 4)}`` of the two TCP sites (cell frame unless ``base_pose``)."""
        self.set_qpos(qpos)
        mj, m, d = self._mujoco, self._model, self._data
        base = pose7_to_matrix(base_pose) if base_pose is not None else np.eye(4)
        out = {}
        for side in ("left", "right"):
            sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, f"tcp_{side}")
            pose = np.eye(4)
            pose[:3, :3] = np.asarray(d.site_xmat[sid]).reshape(3, 3)
            pose[:3, 3] = np.asarray(d.site_xpos[sid])
            out[side] = base @ pose
        return out
