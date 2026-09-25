"""k-nearest-neighbour queries for packed ("offset") point clouds.

Packed layout: ``N`` points of ``B`` groups concatenated, ``offset (B,)`` =
cumulative group sizes with ``offset[-1] == N``. Neighbours never cross groups.
``impl="torch"`` is the brute-force reference; ``impl="cuda"`` runs the vendored
``pointops`` extension (``scripts/install_cuda_ext.sh pointops``).
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch
from torch import Tensor

Impl = Literal["torch", "cuda"]


def _cuda_pointops():
    try:
        import pointops
    except ImportError as e:  # pragma: no cover - message tested with a blocked module
        raise ImportError(
            "impl='cuda' requires the vendored `pointops` CUDA extension; build it with "
            "scripts/install_cuda_ext.sh pointops"
        ) from e
    return pointops


def _offset_bounds(offset: Tensor) -> list:
    ends = offset.tolist()
    starts = [0] + ends[:-1]
    return list(zip(starts, ends))


def knn_query_torch(
    k: int,
    xyz: Tensor,
    offset: Tensor,
    new_xyz: Optional[Tensor] = None,
    new_offset: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Brute-force kNN per group: ``(M, k)`` long indices into ``xyz`` and ``(M, k)`` distances.

    ``xyz (N, 3)`` with ``offset (B,)``; queries ``new_xyz (M, 3)`` with
    ``new_offset (B,)`` default to the points themselves (then the first
    neighbour is the query, distance 0). Groups with fewer than ``k`` points pad
    with the nearest neighbour (distance repeated).
    """
    if new_xyz is None or new_offset is None:
        new_xyz, new_offset = xyz, offset
    if xyz.ndim != 2 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {tuple(xyz.shape)}")
    if offset.shape != new_offset.shape:
        raise ValueError("offset and new_offset must have the same number of groups")
    idx_parts, dist_parts = [], []
    for (s, e), (qs, qe) in zip(_offset_bounds(offset), _offset_bounds(new_offset)):
        n = e - s
        if qe == qs:
            continue
        if n == 0:
            raise ValueError("a query group has no reference points")
        d = torch.cdist(new_xyz[qs:qe], xyz[s:e])  # (m, n)
        kk = min(k, n)
        dist, local = d.topk(kk, dim=1, largest=False)
        if kk < k:
            pad = k - kk
            local = torch.cat([local, local[:, :1].expand(-1, pad)], dim=1)
            dist = torch.cat([dist, dist[:, :1].expand(-1, pad)], dim=1)
        idx_parts.append(local + s)
        dist_parts.append(dist)
    if not idx_parts:
        empty = xyz.new_empty((0, k))
        return empty.long(), empty
    return torch.cat(idx_parts, 0), torch.cat(dist_parts, 0)


def knn_query(
    k: int,
    xyz: Tensor,
    offset: Tensor,
    new_xyz: Optional[Tensor] = None,
    new_offset: Optional[Tensor] = None,
    *,
    impl: Impl = "torch",
) -> Tuple[Tensor, Tensor]:
    """kNN on a packed cloud: ``(M, k)`` long indices into ``xyz`` and ``(M, k)`` distances.

    See :func:`knn_query_torch` for the layout. ``impl="cuda"`` requires CUDA
    tensors and the ``pointops`` extension; the two backends agree up to
    tie-breaking among equidistant points.
    """
    if impl == "torch":
        return knn_query_torch(k, xyz, offset, new_xyz, new_offset)
    if impl == "cuda":
        ops = _cuda_pointops()
        if not xyz.is_cuda:
            raise ValueError("impl='cuda' requires CUDA tensors for knn_query")
        if new_xyz is None or new_offset is None:
            new_xyz, new_offset = xyz, offset
        idx, dist = ops.knn_query(
            k, xyz.contiguous(), offset.int(), new_xyz.contiguous(), new_offset.int()
        )
        return idx.long(), dist
    raise ValueError(f"impl must be 'torch' or 'cuda', got {impl!r}")


def local_knn_query(
    k: int,
    points_3d: Tensor,
    cam_to_world: Tensor,
    intrinsics: Tensor,
    v: int,
    h: int,
    w: int,
    spatial_radius: int = 3,
    num_neighbor_views: int = 4,
    cross_view_radius: int = 3,
) -> Tensor:
    """Windowed kNN for a multi-view depth-map point cloud (pure torch).

    ``points_3d (V*H*W, 3)`` in view-major, row-major order; ``cam_to_world
    (V, 4, 4)``; ``intrinsics (V, 3, 3)`` normalised to ``[0, 1]`` image
    coordinates. Candidates are the ``(2 r + 1)^2 - 1`` 2-D grid neighbours in the
    same view plus a ``(2 r_c + 1)^2`` window around each point's projection into
    the ``num_neighbor_views`` nearest cameras; the ``k`` closest candidates are
    returned as ``(N, k)`` long indices (self index fills empty slots).
    """
    n = points_3d.shape[0]
    if n != v * h * w:
        raise ValueError(f"expected {v * h * w} points for a ({v}, {h}, {w}) grid, got {n}")
    device = points_3d.device
    points_grid = points_3d.reshape(v, h, w, 3)

    # 1. Camera neighbour list (excluding self)
    cam_centers = cam_to_world[:, :3, 3]
    cam_dist = torch.cdist(cam_centers.unsqueeze(0), cam_centers.unsqueeze(0), p=2)[0]
    cam_dist_sorted = torch.argsort(cam_dist, dim=1)
    nv = min(num_neighbor_views, v - 1)
    neighbor_views = cam_dist_sorted[:, 1 : nv + 1]  # (V, nv)

    # 2. Spatial candidate indices (N, S)
    sr = spatial_radius
    offsets = torch.arange(-sr, sr + 1, device=device)
    dr, dc = torch.meshgrid(offsets, offsets, indexing="ij")
    dr, dc = dr.reshape(-1), dc.reshape(-1)
    center_mask = (dr != 0) | (dc != 0)
    dr, dc = dr[center_mask], dc[center_mask]

    view_idx = torch.arange(v, device=device).view(v, 1, 1).expand(v, h, w).reshape(n)
    row_idx = torch.arange(h, device=device).view(1, h, 1).expand(v, h, w).reshape(n)
    col_idx = torch.arange(w, device=device).view(1, 1, w).expand(v, h, w).reshape(n)

    nb_row = row_idx.unsqueeze(1) + dr.unsqueeze(0)
    nb_col = col_idx.unsqueeze(1) + dc.unsqueeze(0)
    spatial_valid = (nb_row >= 0) & (nb_row < h) & (nb_col >= 0) & (nb_col < w)
    nb_row = nb_row.clamp(0, h - 1)
    nb_col = nb_col.clamp(0, w - 1)
    spatial_indices = view_idx.unsqueeze(1) * (h * w) + nb_row * w + nb_col

    # 3. Cross-view candidate indices (N, nv * C)
    cr = cross_view_radius
    cv_offsets = torch.arange(-cr, cr + 1, device=device)
    cv_dr, cv_dc = torch.meshgrid(cv_offsets, cv_offsets, indexing="ij")
    cv_dr, cv_dc = cv_dr.reshape(-1), cv_dc.reshape(-1)
    c = cv_dr.shape[0]

    world_to_cam = torch.linalg.inv(cam_to_world)  # (V, 4, 4)
    hw = h * w
    ones = torch.ones(v, hw, 1, device=device, dtype=points_3d.dtype)
    pts_h = torch.cat([points_grid.reshape(v, hw, 3), ones], dim=-1)  # (V, HW, 4)

    w2c_tgt = world_to_cam[neighbor_views]  # (V, nv, 4, 4)
    intr_tgt = intrinsics[neighbor_views]  # (V, nv, 3, 3)

    cam_pts = pts_h.unsqueeze(1) @ w2c_tgt.transpose(-1, -2)  # (V, nv, HW, 4)
    cam_xyz = cam_pts[..., :3]
    valid_z = cam_xyz[..., 2] > 0

    uv = cam_xyz[..., :2] / cam_xyz[..., 2:3].clamp(min=1e-6)
    uv_ones = torch.ones(*uv.shape[:-1], 1, device=device, dtype=uv.dtype)
    proj = torch.cat([uv, uv_ones], dim=-1) @ intr_tgt.transpose(-1, -2)  # (V, nv, HW, 3)

    proj_col = (proj[..., 0] * w - 0.5).round().long()
    proj_row = (proj[..., 1] * h - 0.5).round().long()

    nb_r = proj_row.unsqueeze(-1) + cv_dr  # (V, nv, HW, C)
    nb_c = proj_col.unsqueeze(-1) + cv_dc

    cross_valid = (nb_r >= 0) & (nb_r < h) & (nb_c >= 0) & (nb_c < w) & valid_z.unsqueeze(-1)
    nb_r.clamp_(0, h - 1)
    nb_c.clamp_(0, w - 1)

    tgt_view_idx = neighbor_views.reshape(v, nv, 1, 1)
    flat_idx = tgt_view_idx * hw + nb_r * w + nb_c  # (V, nv, HW, C)

    cross_indices = flat_idx.permute(0, 2, 1, 3).reshape(n, nv * c)
    cross_valid = cross_valid.permute(0, 2, 1, 3).reshape(n, nv * c)

    # 4. Combine candidates and select the k nearest
    all_indices = torch.cat([spatial_indices, cross_indices], dim=1)
    all_valid = torch.cat([spatial_valid, cross_valid], dim=1)
    total_candidates = all_indices.shape[1]

    gather_idx = all_indices.clamp(0, n - 1)
    candidate_pos = points_3d[gather_idx]  # (N, total, 3)
    sq_dist = ((candidate_pos - points_3d.unsqueeze(1)) ** 2).sum(dim=-1)
    sq_dist[~all_valid] = float("inf")

    kk = min(k, total_candidates)
    topk_dist, topk_local = sq_dist.topk(kk, largest=False)
    result = gather_idx.gather(1, topk_local)
    self_idx = torch.arange(n, device=device).unsqueeze(1)
    if kk < k:
        result = torch.cat([result, self_idx.expand(n, k - kk)], dim=1)
        topk_dist = torch.cat([topk_dist, topk_dist.new_zeros(n, k - kk)], dim=1)

    inf_mask = torch.isinf(topk_dist)
    if inf_mask.any():
        result = torch.where(inf_mask, self_idx.expand_as(result), result)

    return result
