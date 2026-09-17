"""Kinematics + joint angles -> 3-D action points.

Grounds a low-level action (joint angles) in 3-D as a set of points that ride along with
the robot's links. Sample in *link-local* coordinates once, then transform by forward
kinematics per frame: a point sampled in link ``L``'s own frame is a material point of
that link, so correspondence across time is exact by construction.
``temporal_mode="tracks"`` draws the local samples once; ``"independent"`` redraws every
frame.

Kinematics come from MuJoCo (loads URDF as well as MJCF), imported lazily under the
``robot`` extra.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import numpy as np

AlongMode = Literal["linspace", "uniform"]
TemporalMode = Literal["tracks", "independent"]

_ALONG: tuple[str, ...] = ("linspace", "uniform")
_TEMPORAL: tuple[str, ...] = ("tracks", "independent")


def _mujoco():
    try:
        import mujoco
    except ImportError as e:
        raise ImportError("robot kinematics require mujoco; install ontic-data[robot]") from e
    return mujoco


@dataclass(frozen=True)
class _Segment:
    """A link's sampling segment in its own frame; ``end`` is the child origin (zeros for
    a leaf, which samples to a point at the origin)."""

    link: str
    end: np.ndarray  # (3,)


@dataclass(frozen=True)
class LinkSurface:
    """One link's drawable geometry, merged and expressed in the link frame.

    Vertices are baked out of their geom frames into the body frame at load; ``normals``
    are area-weighted vertex normals from the topology; ``colors`` are per-vertex RGB in
    ``[0, 1]``; ``cum_area`` is the cumulative triangle area for O(points) sampling.
    """

    link: str
    vertices: np.ndarray  # (N, 3) float64, link-local
    faces: np.ndarray  # (F, 3) int32
    normals: np.ndarray  # (N, 3) float64, unit
    colors: np.ndarray  # (N, 3) float64 in [0, 1]
    cum_area: np.ndarray  # (F,) float64

    @property
    def total_area(self) -> float:
        return float(self.cum_area[-1]) if len(self.cum_area) else 0.0


class RobotKinematics:
    """Forward kinematics for a robot loaded from URDF or MJCF.

    Construct with :meth:`from_urdf`, :meth:`from_mjcf` or :meth:`from_model`. The
    instance carries MuJoCo state and is not thread-safe; build one per worker.
    """

    def __init__(self, model) -> None:
        mujoco = _mujoco()
        self._mj = mujoco
        self._model = model
        self._data = mujoco.MjData(model)

        self.joint_names: list[str] = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
            for j in range(model.njnt)
        ]
        self._qadr = {n: int(model.jnt_qposadr[j]) for j, n in enumerate(self.joint_names)}
        self.link_names: list[str] = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body_{b}"
            for b in range(model.nbody)
        ]
        self._surface_cache: dict[bool, dict[str, LinkSurface]] = {}

    @classmethod
    def from_urdf(cls, path: str | Path) -> RobotKinematics:
        """Load a URDF. Meshes it references must resolve, as MuJoCo compiles them."""
        return cls._from_file(path)

    @classmethod
    def from_mjcf(cls, path: str | Path) -> RobotKinematics:
        return cls._from_file(path)

    @classmethod
    def from_model(cls, model) -> RobotKinematics:
        """Wrap an already-compiled ``mujoco.MjModel``."""
        return cls(model)

    @classmethod
    def _from_file(cls, path: str | Path) -> RobotKinematics:
        mujoco = _mujoco()
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"no robot description at {p}")
        return cls(mujoco.MjModel.from_xml_path(str(p)))

    @property
    def n_joints(self) -> int:
        return self._model.njnt

    @property
    def movable_links(self) -> list[str]:
        """Bodies that carry a joint — the ones whose pose the action changes."""
        m = self._model
        return [self.link_names[int(m.jnt_bodyid[j])] for j in range(m.njnt)]

    def _segments(self, links: Sequence[str]) -> list[_Segment]:
        m = self._model
        out: list[_Segment] = []
        for name in links:
            bid = self._mj.mj_name2id(m, self._mj.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"unknown link {name!r}; have {self.link_names}")
            kids = [b for b in range(m.nbody) if int(m.body_parentid[b]) == bid and b != bid]
            # The first child spans the link's own extent; leaves collapse to the origin.
            end = np.asarray(m.body_pos[kids[0]], dtype=np.float64) if kids else np.zeros(3)
            out.append(_Segment(link=name, end=end))
        return out

    def link_surfaces(
        self, links: Sequence[str] | None = None, visual_only: bool = True
    ) -> list[LinkSurface]:
        """Per-link merged mesh geometry in link frames; built once and cached.

        ``visual_only`` drops MuJoCo group-3 geoms (collision hulls).
        """
        cache = self._surface_cache.get(visual_only)
        if cache is None:
            cache = self._build_surfaces(visual_only)
            self._surface_cache[visual_only] = cache
        if links is None:
            return list(cache.values())
        missing = [ln for ln in links if ln not in cache]
        if missing:
            raise ValueError(f"no mesh geometry for link(s) {missing}")
        return [cache[ln] for ln in links]

    def _build_surfaces(self, visual_only: bool) -> dict[str, LinkSurface]:
        mj, m = self._mj, self._model
        per_body: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
        for g in range(m.ngeom):
            if m.geom_type[g] != mj.mjtGeom.mjGEOM_MESH:
                continue
            if visual_only and m.geom_group[g] >= 3:
                continue
            mesh = int(m.geom_dataid[g])
            v0, nv = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
            f0, nf = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
            verts = np.asarray(m.mesh_vert[v0 : v0 + nv], dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(m.mesh_face[f0 : f0 + nf], dtype=np.int64).reshape(-1, 3)

            # Geom frame -> body frame. Constant, so bake it in now.
            R = _quat_wxyz_to_rot(np.asarray(m.geom_quat[g], dtype=np.float64))
            verts = verts @ R.T + np.asarray(m.geom_pos[g], dtype=np.float64)

            body = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or "world"
            rgb = np.asarray(_geom_rgb(m, g), dtype=np.float64)
            per_body.setdefault(body, []).append(
                (verts, faces, np.broadcast_to(rgb, (len(verts), 3)).copy())
            )

        out: dict[str, LinkSurface] = {}
        for body, parts in per_body.items():
            verts, faces, colors = _merge_meshes(parts)
            normals = _vertex_normals(verts, faces)
            areas = _triangle_areas(verts, faces)
            keep = areas > 0  # degenerate triangles would break the area CDF
            faces, areas = faces[keep], areas[keep]
            if not len(faces):
                continue
            out[body] = LinkSurface(
                link=body,
                vertices=verts,
                faces=faces.astype(np.int32),
                normals=normals,
                colors=colors,
                cum_area=np.cumsum(areas),
            )
        return out

    def link_poses(self, qpos: np.ndarray | Mapping[str, float]) -> dict[str, np.ndarray]:
        """``{link name: (4, 4)}`` in the model's own root frame.

        ``qpos`` is a ``(n_joints,)`` array in model joint order or a mapping of joint
        name -> angle (absent joints stay at zero).
        """
        d = self._data
        d.qpos[:] = 0.0
        if isinstance(qpos, Mapping):
            for name, value in qpos.items():
                adr = self._qadr.get(name)
                if adr is not None:
                    d.qpos[adr] = float(value)
        else:
            q = np.asarray(qpos, dtype=np.float64).ravel()
            for j, name in enumerate(self.joint_names):
                if j < len(q):
                    d.qpos[self._qadr[name]] = q[j]
        self._mj.mj_forward(self._model, d)

        out = {}
        for b, name in enumerate(self.link_names):
            T = np.eye(4)
            T[:3, :3] = np.asarray(d.xmat[b]).reshape(3, 3)
            T[:3, 3] = np.asarray(d.xpos[b])
            out[name] = T
        return out


def sample_action_points(
    kin: RobotKinematics,
    qpos,
    n_per_link: int = 10,
    temporal_mode: TemporalMode = "tracks",
    along: AlongMode = "linspace",
    noise: float = 0.0,
    links: Sequence[str] | None = None,
    base_pose: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Points riding along the robot's links, one set per timestep.

    ``qpos``: ``(T, D)`` joint angles, ``(D,)`` for one frame, or a mapping / sequence of
    mappings of joint name -> angle. ``base_pose``: ``(7,)`` ``[xyz, wxyz]`` placing the
    robot's root in the world. Returns ``(points (T, P, 3) float32 world-frame,
    link_ids (P,))`` with ``P = n_per_link * len(links)``.
    """
    if temporal_mode not in _TEMPORAL:
        raise ValueError(f"temporal_mode must be one of {_TEMPORAL}, got {temporal_mode!r}")
    if along not in _ALONG:
        raise ValueError(f"along must be one of {_ALONG}, got {along!r}")
    if n_per_link < 1:
        raise ValueError(f"n_per_link must be >= 1, got {n_per_link}")

    frames = _as_frames(qpos)
    chosen = list(links) if links is not None else _dedup(kin.movable_links)
    segments = kin._segments(chosen)
    rng = np.random.default_rng(seed)

    local = _local_samples(segments, n_per_link, along, noise, rng)
    base = _pose7_to_matrix(base_pose) if base_pose is not None else None

    out = np.empty((len(frames), len(segments) * n_per_link, 3), dtype=np.float32)
    for t, frame in enumerate(frames):
        if temporal_mode == "independent" and t > 0:
            local = _local_samples(segments, n_per_link, along, noise, rng)
        poses = kin.link_poses(frame)
        for i, seg in enumerate(segments):
            T = poses[seg.link]
            if base is not None:
                T = base @ T
            pts = local[i] @ T[:3, :3].T + T[:3, 3]
            out[t, i * n_per_link : (i + 1) * n_per_link] = pts
    link_ids = np.array([s.link for s in segments for _ in range(n_per_link)])
    return out, link_ids


def sample_surface_points(
    kin: RobotKinematics,
    qpos,
    n_per_link: int = 256,
    temporal_mode: TemporalMode = "tracks",
    links: Sequence[str] | None = None,
    base_pose: np.ndarray | None = None,
    visual_only: bool = True,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Area-uniform points on the robot's link surfaces, with colour and smoothed normals.

    Returns ``(points (T, P, 3), normals (T, P, 3), colors (P, 3) in [0, 1],
    link_ids (P,))`` with ``P = n_per_link * len(links)``. Surface sampling is random,
    so ``"tracks"`` and ``"independent"`` always differ.
    """
    if temporal_mode not in _TEMPORAL:
        raise ValueError(f"temporal_mode must be one of {_TEMPORAL}, got {temporal_mode!r}")
    if n_per_link < 1:
        raise ValueError(f"n_per_link must be >= 1, got {n_per_link}")

    surfaces = kin.link_surfaces(links, visual_only=visual_only)
    if not surfaces:
        raise ValueError("the model carries no mesh geometry to sample")

    frames = _as_frames(qpos)
    rng = np.random.default_rng(seed)
    base = _pose7_to_matrix(base_pose) if base_pose is not None else None

    local = [_sample_one_surface(s, n_per_link, rng) for s in surfaces]
    P = len(surfaces) * n_per_link
    pts = np.empty((len(frames), P, 3), dtype=np.float32)
    nrm = np.empty((len(frames), P, 3), dtype=np.float32)

    for t, frame in enumerate(frames):
        if temporal_mode == "independent" and t > 0:
            local = [_sample_one_surface(s, n_per_link, rng) for s in surfaces]
        poses = kin.link_poses(frame)
        for i, surf in enumerate(surfaces):
            T = poses[surf.link]
            if base is not None:
                T = base @ T
            lo, hi = i * n_per_link, (i + 1) * n_per_link
            p_loc, n_loc, _ = local[i]
            pts[t, lo:hi] = p_loc @ T[:3, :3].T + T[:3, 3]
            nrm[t, lo:hi] = n_loc @ T[:3, :3].T  # directions: rotate, never translate

    colors = np.concatenate([c for _, _, c in local], 0).astype(np.float32)
    link_ids = np.array([s.link for s in surfaces for _ in range(n_per_link)])
    return pts, nrm, colors, link_ids


def to_action_tensor(points: np.ndarray) -> np.ndarray:
    """``(T, P, 3)`` -> ``(T, 1, P, 4)`` xyz + presence bit (always 1: the robot exists)."""
    pts = np.asarray(points, dtype=np.float32)
    out = np.ones(pts.shape[:-1] + (4,), dtype=np.float32)
    out[..., :3] = pts
    return out[:, None]


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #


def _local_samples(
    segments: Sequence[_Segment], n: int, along: str, noise: float, rng
) -> list[np.ndarray]:
    """Per-segment ``(n, 3)`` samples in the link's own frame."""
    out = []
    for seg in segments:
        if along == "linspace":
            # A single point sits at the midpoint rather than at an endpoint.
            t = np.linspace(0.0, 1.0, n) if n > 1 else np.array([0.5])
        else:
            t = rng.uniform(0.0, 1.0, size=n)
        pts = t[:, None] * seg.end[None, :]
        if noise > 0:
            pts = pts + rng.normal(0.0, noise, size=pts.shape)
        out.append(pts.astype(np.float64))
    return out


def _as_frames(qpos) -> list:
    """Normalise the accepted ``qpos`` spellings to a list of per-frame values."""
    if isinstance(qpos, Mapping):
        return [qpos]
    if isinstance(qpos, (list, tuple)) and qpos and isinstance(qpos[0], Mapping):
        return list(qpos)
    arr = np.asarray(qpos, dtype=np.float64)
    return [arr] if arr.ndim == 1 else list(arr)


def _dedup(names: Sequence[str]) -> list[str]:
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _pose7_to_matrix(pose) -> np.ndarray:
    """``[x, y, z, qw, qx, qy, qz]`` -> ``(4, 4)``, normalising the quaternion."""
    p = np.asarray(pose, dtype=np.float64).reshape(7)
    q = p[3:7]
    norm = np.linalg.norm(q)
    w, x, y, z = q / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    T = np.eye(4)
    T[:3, :3] = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    T[:3, 3] = p[:3]
    return T


def _geom_rgb(model, g: int) -> np.ndarray:
    """A geom's RGB, preferring its material over the geom's own rgba."""
    mat = int(model.geom_matid[g])
    if mat >= 0:
        return np.asarray(model.mat_rgba[mat][:3], dtype=np.float64)
    return np.asarray(model.geom_rgba[g][:3], dtype=np.float64)


def _merge_meshes(parts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate ``(verts, faces, colors)`` triples, rebasing face indices."""
    verts, faces, colors, offset = [], [], [], 0
    for v, f, c in parts:
        verts.append(v)
        faces.append(f + offset)
        colors.append(c)
        offset += len(v)
    return np.concatenate(verts, 0), np.concatenate(faces, 0), np.concatenate(colors, 0)


def _triangle_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def _vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted vertex normals from the topology (exporters' normals vary)."""
    a, b, c = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    face_n = np.cross(b - a, c - a)  # un-normalised: already area-weighted
    out = np.zeros_like(verts)
    for k in range(3):
        np.add.at(out, faces[:, k], face_n)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    out = np.divide(out, norms, out=np.zeros_like(out), where=norms > 0)
    out[norms[:, 0] == 0] = np.array([0.0, 0.0, 1.0])  # isolated vertex: +z, not NaN
    return out


def _sample_one_surface(
    surf: LinkSurface, n: int, rng
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(points, normals, colors)`` in the link frame, uniform over the surface."""
    tri = np.searchsorted(surf.cum_area, rng.uniform(0.0, surf.total_area, size=n), side="right")
    tri = np.clip(tri, 0, len(surf.faces) - 1)
    f = surf.faces[tri]

    # Fold (u, v) about u + v = 1: uniform on the unit square becomes uniform on the triangle.
    u, v = rng.uniform(size=n), rng.uniform(size=n)
    over = u + v > 1.0
    u = np.where(over, 1.0 - u, u)
    v = np.where(over, 1.0 - v, v)
    w = 1.0 - u - v

    bary = np.stack([w, u, v], axis=1)[:, :, None]  # (n, 3, 1)
    pts = (surf.vertices[f] * bary).sum(axis=1)
    nrm = (surf.normals[f] * bary).sum(axis=1)
    norms = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.divide(nrm, norms, out=np.zeros_like(nrm), where=norms > 0)
    # Faces never span geoms, so all three vertices share a colour; take v0's.
    return pts, nrm, surf.colors[f[:, 0]]


def _quat_wxyz_to_rot(q: np.ndarray) -> np.ndarray:
    """``wxyz`` -> ``(3, 3)``."""
    return _pose7_to_matrix(np.concatenate([np.zeros(3), np.asarray(q, dtype=np.float64)]))[:3, :3]


# --------------------------------------------------------------------------- #
# Locating a robot description
#
# SynthRobot ships no robot model; both the dataset's ``robot_points`` action mode and
# a viewer overlay need the onticlabs/robotics checkout. One lookup, so the two cannot
# disagree about where the robot is or why it is missing.
# --------------------------------------------------------------------------- #

#: Env var pointing at an ``onticlabs/robotics`` checkout.
ENV_REPO = "ONTIC_ROBOTICS_REPO"

_FR3_XML = "robots/franka_duo/vendor/duobench/assets/robots/fr3/fr3.xml"
_MARKER = "robots/franka_duo/robot_franka_duo"


def _is_checkout(path: Path) -> bool:
    return (path / _MARKER).is_dir()


def _candidate_repos() -> list[Path]:
    """Places a ``robotics`` checkout may sit, nearest first (walking up from here)."""
    here = Path(__file__).resolve()
    return [parent / "robotics" for parent in here.parents[2:8]]


def robotics_repo() -> Path | None:
    """The robotics checkout, or ``None`` when none can be found."""
    override = os.environ.get(ENV_REPO)
    if override:
        root = Path(override)
        return root if _is_checkout(root) else None
    return next((c for c in _candidate_repos() if _is_checkout(c)), None)


def _has_mujoco() -> bool:
    try:
        _mujoco()
    except ImportError:
        return False
    return True


def robot_model_available() -> bool:
    """Whether a robot can be built: checkout, vendored assets and mujoco."""
    root = robotics_repo()
    if root is None or not (root / _FR3_XML).exists():
        return False
    return _has_mujoco()


def describe_unavailable() -> str:
    """One line saying what is missing, for a GUI or an exception to quote."""
    root = robotics_repo()
    if root is None:
        looked = ", ".join(str(c) for c in _candidate_repos())
        return f"no robotics checkout found (looked in {looked}; set ${ENV_REPO})"
    if not (root / _FR3_XML).exists():
        return (
            "duobench submodule not initialised "
            "(git submodule update --init robots/franka_duo/vendor/duobench)"
        )
    if not _has_mujoco():
        return "mujoco not installed (install ontic-data[robot])"
    return ""


def build_franka_duo_model(
    repo: Path | None = None, with_worktable: bool = False, with_cameras: bool = False
):
    """Compile the FR3 Duo cell as a ``mujoco.MjModel`` via upstream's MJCF builder."""
    root = repo or robotics_repo()
    if root is None:
        raise RuntimeError(describe_unavailable() or "robotics checkout not found")
    for sub in ("robots/franka_duo", "packages/robotics-core"):
        path = str(root / sub)
        if path not in sys.path:
            sys.path.insert(0, path)
    from robot_franka_duo.mjcf import load_franka_duo_mjcf

    return load_franka_duo_mjcf(with_worktable=with_worktable, with_cameras=with_cameras).compile()


def franka_duo_kinematics(repo: Path | None = None) -> RobotKinematics:
    """:class:`RobotKinematics` for the FR3 Duo cell."""
    return RobotKinematics.from_model(build_franka_duo_model(repo))


#: The 2F-85 links its driver, spring link and follower into a four-bar that rotates
#: together; MuJoCo's ``mj_forward`` does not project qpos onto the constraint manifold,
#: so a recorded driver angle is fanned out to all three by hand.
_DUO_LINKAGE = ("driver", "spring_link", "follower")


def duo_joint_angles(groups: Mapping[str, Sequence[float]]) -> dict[str, float]:
    """SynthRobot named qpos groups -> ``{joint name: angle}`` for the duo cell.

    Accepts the dict straight out of ``obs/agent/qpos``; unknown groups are ignored.
    """
    out: dict[str, float] = {}
    for side in ("left", "right"):
        for i, value in enumerate(list(groups.get(f"{side}_arm", []))[:7]):
            out[f"{side}_fr3_joint{i + 1}"] = float(value)
        grip = list(groups.get(f"{side}_gripper", []))
        if grip:
            # The two reported values are the left and right jaws' driver joints.
            for jaw, value in (("left", grip[0]), ("right", grip[-1])):
                for part in _DUO_LINKAGE:
                    out[f"{side}_gripper_{jaw}_{part}_joint"] = float(value)
    return out
