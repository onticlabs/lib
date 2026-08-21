"""3D particle / rendered-frame metrics for particle world-models.

Pure tensor/array functions shared by particle world-model experiments
(transformer-dynamics/ncross1, ptv3-genesis, ...): object/plate splitting of a
segmented particle cloud, dissolution (cloud-coherence) ratios, and per-frame
PSNR over rendered uint8 frames. Everything here is model-agnostic — inputs
are plain tensors/arrays, no wrapper/state objects.

Requires torch, which is a core dependency. Import this submodule explicitly
to keep these heavier 3D helpers out of the top-level namespace.
"""
from __future__ import annotations

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# uint8 frames + PSNR
# --------------------------------------------------------------------------- #
def to_u8(x):
    """(T,3,H,W) or (T,H,W,3) float[0,1] -> (T,H,W,3) uint8 numpy."""
    x = x.detach().float().cpu()
    if x.dim() == 4 and x.shape[1] == 3 and x.shape[-1] != 3:
        x = x.permute(0, 2, 3, 1)
    return x.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()


def psnr_per_step(gt_u8, pr_u8):
    """Per-frame PSNR (dB) over (T,H,W,3) uint8 frame stacks -> (T,) float array."""
    g = gt_u8.astype(np.float32) / 255.0
    p = pr_u8.astype(np.float32) / 255.0
    mse = ((g - p) ** 2).reshape(g.shape[0], -1).mean(1)
    return 10.0 * np.log10(1.0 / np.maximum(mse, 1e-10))


# --------------------------------------------------------------------------- #
# object/plate splitting of a segmented particle cloud
# --------------------------------------------------------------------------- #
@torch.no_grad()
def otsu_zcut(zvals, nbins=128):
    """The separating line of a BIMODAL z-density (object mode on top, plate mode
    below): Otsu's threshold = the z that maximizes inter-class variance of the
    z-histogram. Parameter-free (not an ad-hoc gap)."""
    z = zvals.detach().float().cpu()
    zmin, zmax = float(z.min()), float(z.max())
    if zmax - zmin < 1e-6:
        return zmin
    hist = torch.histc(z, bins=nbins, min=zmin, max=zmax)
    centers = torch.linspace(zmin, zmax, nbins)
    p = hist / hist.sum().clamp_min(1.0)
    omega = torch.cumsum(p, 0)  # weight of the "below" class up to each bin
    mu = torch.cumsum(p * centers, 0)
    muT = mu[-1]
    denom = (omega * (1.0 - omega)).clamp_min(1e-12)
    sigma_b2 = (muT * omega - mu) ** 2 / denom  # inter-class variance
    return float(centers[int(sigma_b2.argmax())])


@torch.no_grad()
def split_obj_plate(means0: torch.Tensor, mask: torch.Tensor):
    """1D z-gap object/plate split. means0 (N,3) frame-0 positions, mask (N,) bool
    -> (obj_mask, plate_mask) or (None, None). Cut = the largest vertical gap
    among masked particles (fallback: lowest z + 0.10)."""
    if int(mask.sum()) < 10:
        return None, None
    z = means0[:, 2]
    zs, _ = torch.sort(z[mask])
    gaps = zs[1:] - zs[:-1]
    gi = int(torch.argmax(gaps))
    zcut = float((zs[gi] + zs[gi + 1]) / 2) if float(gaps[gi]) > 0.03 else float(zs[0] + 0.10)
    obj = mask & (z > zcut)
    plate = mask & (z <= zcut)
    if int(obj.sum()) < 5 or int(plate.sum()) < 5:
        return None, None
    return obj, plate


@torch.no_grad()
def median_nn_dist(P):
    """Median nearest-neighbour distance of a point cloud P (M,3). Chunked cdist."""
    M = P.shape[0]
    nn = torch.empty(M, device=P.device, dtype=P.dtype)
    cs = 2048
    for i in range(0, M, cs):
        d = torch.cdist(P[i:i + cs], P)
        d[torch.arange(d.shape[0]), torch.arange(i, min(i + cs, M))] = float("inf")
        nn[i:i + cs] = d.min(dim=1).values
    return float(nn.median())


@torch.no_grad()
def radius_components(P, eps, min_size=10):
    """Connected components of the eps-radius graph over P (M,3), union-find on
    chunked cdist edges. Returns index tensors, largest first; components smaller
    than min_size are dropped."""
    M = P.shape[0]
    eps2 = eps * eps
    edges_a, edges_b = [], []
    cs = 1024
    for i in range(0, M, cs):
        d2 = torch.cdist(P[i:i + cs], P) ** 2
        ii, jj = torch.where(d2 <= eps2)
        ii = ii + i
        keep = ii < jj
        edges_a.append(ii[keep])
        edges_b.append(jj[keep])
    ea = torch.cat(edges_a).cpu().numpy() if edges_a else np.empty(0, np.int64)
    eb = torch.cat(edges_b).cpu().numpy() if edges_b else np.empty(0, np.int64)
    parent = np.arange(M, dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in zip(ea.tolist(), eb.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(x) for x in range(M)], dtype=np.int64)
    comps = []
    for r in np.unique(roots):
        idx = np.where(roots == r)[0]
        if idx.size >= min_size:
            comps.append(torch.from_numpy(idx).to(P.device).long())
    comps.sort(key=lambda t: -t.numel())
    return comps


@torch.no_grad()
def cluster_obj_plate_3d(pos_all: torch.Tensor, seg: torch.Tensor,
                         eps_scale=2.5, vgap_clean=0.18, mind3d_clean=0.25):
    """Full-3D object/plate clustering. pos_all (N,3) positions, seg (N,) bool
    foreground mask -> (obj_mask, plate_mask, clean, info) with (N,) bool masks.
    PLATE = the lower cluster; OBJECT = the higher one. clean = two dominant
    clusters with a real 3D gap (vgap >= vgap_clean & min-3D-dist >= mind3d_clean
    & the two clusters cover >80% of the foreground)."""
    seg_idx = torch.where(seg)[0]
    P = pos_all[seg_idx].float()
    M = P.shape[0]
    obj = torch.zeros(pos_all.shape[0], dtype=torch.bool, device=pos_all.device)
    plate = torch.zeros_like(obj)
    if M < 20:
        return obj, plate, False, {"reason": "too few seg particles"}
    nn = median_nn_dist(P)
    eps = eps_scale * nn
    comps = radius_components(P, eps, min_size=max(10, M // 100))
    if len(comps) < 2:
        if comps:
            obj[seg_idx[comps[0]]] = True
        return obj, plate, False, {"reason": f"{len(comps)} cluster(s)"}
    cA, cB = comps[0], comps[1]

    def geom(c):
        Q = P[c]
        xy = float((Q[:, :2].max(0).values - Q[:, :2].min(0).values).norm())
        zext = float(Q[:, 2].max() - Q[:, 2].min())
        return dict(n=int(c.numel()), cen_z=float(Q.mean(0)[2]), xy=xy, zext=zext,
                    flat=xy / max(zext, 1e-6))
    gA, gB = geom(cA), geom(cB)
    if gA["cen_z"] <= gB["cen_z"]:
        plate_c, obj_c, gP, gO = cA, cB, gA, gB
    else:
        plate_c, obj_c, gP, gO = cB, cA, gB, gA
    obj[seg_idx[obj_c]] = True
    plate[seg_idx[plate_c]] = True
    Po, Pp = P[obj_c], P[plate_c]
    vgap = float(Po[:, 2].min() - Pp[:, 2].max())
    md = float("inf")
    for i in range(0, Po.shape[0], 1024):
        md = min(md, float(torch.cdist(Po[i:i + 1024], Pp).min()))
    dom_frac = (gA["n"] + gB["n"]) / max(M, 1)
    clean = bool(vgap >= vgap_clean and md >= mind3d_clean and gO["cen_z"] > gP["cen_z"]
                 and dom_frac > 0.80)
    return obj, plate, clean, dict(vgap=vgap, mind3d=md, dom_frac=float(dom_frac),
                                   obj_n=gO["n"], plate_n=gP["n"],
                                   obj_flat=gO["flat"], plate_flat=gP["flat"])


# --------------------------------------------------------------------------- #
# dissolution (cloud / rendered-footprint coherence)
# --------------------------------------------------------------------------- #
def dissolution(m):
    """rms_radius(t_final)/rms_radius(t0) for particle means m (T,K,3).
    ~1 = the cloud stays coherent; >1 = it diffuses/dissolves."""
    def rms(P):
        c = P.mean(0)
        return float(((P - c) ** 2).sum(-1).mean().sqrt())
    return rms(m[-1]) / max(rms(m[0]), 1e-6)


def alpha_spread(alpha_t):
    """Alpha-weighted rms spatial spread of one rendered alpha frame (H,W)."""
    a = alpha_t.numpy() if hasattr(alpha_t, "numpy") else np.asarray(alpha_t)
    a = np.clip(a, 0.0, 1.0)
    m = float(a.sum())
    if m < 1e-6:
        return float("nan")
    H, W = a.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    cy = float((a * ys).sum() / m)
    cx = float((a * xs).sum() / m)
    var = float((a * ((ys - cy) ** 2 + (xs - cx) ** 2)).sum() / m)
    return float(np.sqrt(max(var, 0.0)))


def render_dissolution(alpha, n_state, early_w=6, late_lo=30, late_hi=50):
    """Render-based dissolution proxy: late-window foreground rms-spread /
    early-window rms-spread, from the predicted ALPHA (what the eye sees, incl.
    translucent halo). >1 = the rendered footprint puffs out (dissolving).
    alpha: (T,H,W) float over the full rollout (given+predicted frames).

    Caveat: the alpha frame covers the WHOLE scene — a large static plate
    dominates the alpha-weighted spread and compresses the ratio toward 1, so
    treat this as a directional signal, not an object-isolated measurement."""
    if alpha is None:
        return float("nan")
    T = alpha.shape[0]
    e0, e1 = n_state, min(T, n_state + early_w)  # just after handoff (object intact)
    l0, l1 = min(T, late_lo), min(T, late_hi)    # long-horizon window
    if e1 <= e0 or l1 <= l0:
        return float("nan")
    early = [alpha_spread(alpha[t]) for t in range(e0, e1)]
    late = [alpha_spread(alpha[t]) for t in range(l0, l1)]
    early = [v for v in early if v == v]
    late = [v for v in late if v == v]
    if not early or not late:
        return float("nan")
    re = sum(early) / len(early)
    rl = sum(late) / len(late)
    return float(rl / max(re, 1e-6))
