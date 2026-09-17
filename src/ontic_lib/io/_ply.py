"""``.ply`` backend (``ontic-lib[ply]``): standard 3DGS vertex layout, unbatched only.

Fields: ``x y z nx ny nz f_dc_{0..2} f_rest_{0..3*(d_sh-1)-1} opacity scale_{0..2} rot_{0..3}``,
all float32. ``scale_i`` holds ``log(scales)``, ``opacity`` holds ``logit(opacities)``,
``rot_i`` is ``wxyz``; ``f_rest`` is channel-major (``harmonics[:, c, 1 + k]`` at
``c * (d_sh - 1) + k``). Normals are written as zeros and ignored on read. Mask, extras and
explicit covariances have no place in this layout and raise ``ValueError``; metadata is
stored as ``key=value`` comments.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..structures.gaussians import Gaussians

_COMMENT_PREFIX = "ontic:"


def _plyfile() -> Any:
    try:
        import plyfile
    except ImportError as e:
        raise ImportError("ply Gaussians files require plyfile; install ontic-lib[ply]") from e
    return plyfile


def _columns(g: Gaussians) -> dict[str, Tensor]:
    cols: dict[str, Tensor] = {}
    means = g.means
    cols.update(x=means[:, 0], y=means[:, 1], z=means[:, 2])
    zeros = torch.zeros_like(means[:, 0])
    cols.update(nx=zeros, ny=zeros, nz=zeros)
    if g.harmonics is not None:
        sh = g.harmonics  # (N, 3, d_sh)
        for c in range(3):
            cols[f"f_dc_{c}"] = sh[:, c, 0]
        rest = sh[:, :, 1:].reshape(sh.shape[0], -1)  # channel-major
        for i in range(rest.shape[1]):
            cols[f"f_rest_{i}"] = rest[:, i]
    if g.opacities is not None:
        cols["opacity"] = torch.logit(g.opacities[:, 0])
    if g.scales is not None:
        for i in range(3):
            cols[f"scale_{i}"] = torch.log(g.scales[:, i])
    if g.rotations is not None:
        for i in range(4):
            cols[f"rot_{i}"] = g.rotations[:, i]
    return cols


def save(path: str, g: Gaussians, *, metadata: dict[str, str]) -> None:
    ply = _plyfile()
    if g.batch_shape != ():
        raise ValueError(
            f"ply supports unbatched Gaussians only, got batch shape {g.batch_shape}; "
            "index or flatten first, or use .npz/.safetensors"
        )
    unsupported = [
        name
        for name, present in (
            ("mask", g.mask is not None),
            ("covariances", g.covariances is not None),
            ("extras", bool(g.extras)),
        )
        if present
    ]
    if unsupported:
        raise ValueError(
            f"ply cannot store {unsupported}; drop them via replace(...) or use .npz/.safetensors"
        )
    cols = {k: v.detach().cpu().float().numpy() for k, v in _columns(g).items()}
    vertices = np.empty(g.num_gaussians, dtype=[(name, "f4") for name in cols])
    for name, values in cols.items():
        vertices[name] = values
    comments = [f"{_COMMENT_PREFIX}{k}={v}" for k, v in metadata.items()]
    ply.PlyData([ply.PlyElement.describe(vertices, "vertex")], comments=comments).write(path)


def _stack(data: Any, names: list[str]) -> Tensor:
    return torch.from_numpy(np.stack([np.asarray(data[n], dtype=np.float32) for n in names], -1))


def load(path: str, *, device: Any) -> Gaussians:
    ply = _plyfile()
    data = ply.PlyData.read(path)["vertex"].data
    names = set(data.dtype.names)
    fields: dict[str, Tensor | None] = {"means": _stack(data, ["x", "y", "z"])}
    if {"f_dc_0", "f_dc_1", "f_dc_2"} <= names:
        rest = sorted((n for n in names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
        if len(rest) % 3:
            raise ValueError(f"expected 3 * (d_sh - 1) f_rest fields, got {len(rest)}")
        dc = _stack(data, ["f_dc_0", "f_dc_1", "f_dc_2"]).unsqueeze(-1)  # (N, 3, 1)
        sh = dc
        if rest:
            sh = torch.cat([dc, _stack(data, rest).reshape(dc.shape[0], 3, -1)], dim=-1)
        fields["harmonics"] = sh
    if "opacity" in names:
        fields["opacities"] = torch.sigmoid(_stack(data, ["opacity"]))
    if {"scale_0", "scale_1", "scale_2"} <= names:
        fields["scales"] = torch.exp(_stack(data, ["scale_0", "scale_1", "scale_2"]))
    if {"rot_0", "rot_1", "rot_2", "rot_3"} <= names:
        fields["rotations"] = _stack(data, ["rot_0", "rot_1", "rot_2", "rot_3"])
    if device is not None:
        fields = {k: v.to(device) for k, v in fields.items() if v is not None}
    return Gaussians(**fields)
