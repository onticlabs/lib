"""21-joint hand keypoints: projection helpers and OpenCV overlays.

Joints follow the MediaPipe order; ``HAND_BONES`` lists the skeleton edges. Drawing
functions take uint8 ``(H, W, 3)`` arrays and import cv2 lazily (extra ``opencv``).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

HAND_COLORS = {"left": (50, 220, 100), "right": (255, 220, 50)}

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # index
    (0, 9), (9, 10), (10, 11), (11, 12),  # middle
    (0, 13), (13, 14), (14, 15), (15, 16),  # ring
    (0, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (1, 5), (5, 9), (9, 13),  # palm cross-links
]  # fmt: skip

#: Default trail-point colour (RGB).
TRAIL_POINT_COLOR = (255, 220, 50)


def _cv2():
    try:
        import cv2
    except ImportError as e:
        raise ImportError("hand overlays require opencv; install ontic-data[opencv]") from e
    return cv2


# --------------------------------------------------------------------------- #
# Projection helpers (numpy)
# --------------------------------------------------------------------------- #


def project_points_w2c(
    pts_world: np.ndarray, K_px: np.ndarray, R: np.ndarray, T: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project ``(N, 3)`` world points with w2c ``R (3,3)``, ``T (3,)`` and pixel-space
    ``K``; returns ``(uv (N, 2), depth (N,))``."""
    p_cam = (R @ pts_world.T).T + T
    depth = p_cam[:, 2].copy()
    safe_z = np.where(np.abs(depth) > 1e-6, depth, 1e-6)
    u = K_px[0, 0] * p_cam[:, 0] / safe_z + K_px[0, 2]
    v = K_px[1, 1] * p_cam[:, 1] / safe_z + K_px[1, 2]
    return np.stack([u, v], axis=1), depth


def c2w_and_norm_K_to_w2c(
    c2w_4x4: np.ndarray, K_norm: np.ndarray, H: int, W: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """c2w ``(4,4)`` + normalised ``K`` -> w2c ``(R, T)`` and pixel-space ``K``."""
    w2c = np.linalg.inv(c2w_4x4.astype(np.float64))
    R = w2c[:3, :3]
    T = w2c[:3, 3]
    K_px = K_norm.astype(np.float64).copy()
    K_px[0, :] *= W
    K_px[1, :] *= H
    return R, T, K_px


# --------------------------------------------------------------------------- #
# Drawing (cv2)
# --------------------------------------------------------------------------- #


def draw_hand_skeleton(img, joints_world, K_px, R, T, color):
    """Draw the 21-joint skeleton onto ``img`` (uint8 HWC, in place)."""
    cv2 = _cv2()
    uv, depth = project_points_w2c(joints_world, K_px, R, T)
    H, W = img.shape[:2]
    uv_int = uv.astype(np.int32)

    joint_color = tuple(min(255, c + 60) for c in color)
    outline_color = tuple(max(0, c - 80) for c in color)
    bone_w = max(2, int(min(W, H) / 200))
    joint_r = max(3, int(min(W, H) / 150))

    for j1, j2 in HAND_BONES:
        if depth[j1] < 0.01 or depth[j2] < 0.01:
            continue
        p1, p2 = tuple(uv_int[j1]), tuple(uv_int[j2])
        cv2.line(img, p1, p2, outline_color, bone_w + 2, cv2.LINE_AA)
        cv2.line(img, p1, p2, joint_color, bone_w, cv2.LINE_AA)

    for i in range(len(joints_world)):
        if depth[i] < 0.01:
            continue
        u_s, v_s = uv_int[i]
        if not (-30 <= u_s < W + 30 and -30 <= v_s < H + 30):
            continue
        r = joint_r + 1 if i in (0, 1, 5, 9, 13, 17) else joint_r
        cv2.circle(img, (u_s, v_s), r, outline_color, -1, cv2.LINE_AA)
        cv2.circle(img, (u_s, v_s), max(1, r - 1), joint_color, -1, cv2.LINE_AA)
    return img


def draw_hand_skeleton_alpha(img, joints_world, K_px, R, T, color, alpha, radius=4):
    """Draw projected keypoints as alpha-blended filled circles (no bones), in place."""
    cv2 = _cv2()
    alpha = float(alpha)
    if alpha <= 0.0:
        return img
    uv, depth = project_points_w2c(joints_world, K_px, R, T)
    H, W = img.shape[:2]
    uv_int = uv.astype(np.int32)
    radius = max(1, int(radius))
    target = img if alpha >= 1.0 else img.copy()
    for i in range(len(joints_world)):
        if depth[i] < 0.01:
            continue
        u, v = int(uv_int[i, 0]), int(uv_int[i, 1])
        if not (-radius <= u < W + radius and -radius <= v < H + radius):
            continue
        cv2.circle(target, (u, v), radius, color, -1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(target, alpha, img, 1.0 - alpha, 0.0, dst=img)
    return img


def overlay_hand_trail(
    images: Tensor,
    trail_actions: dict[str, Tensor] | None,
    extrinsics: Tensor,
    intrinsics: Tensor,
    step_opacity,
    point_radius: int = 4,
    point_color: tuple[int, int, int] = TRAIL_POINT_COLOR,
) -> Tensor:
    """Paint a fading multi-step keypoint trail onto ``images (V, 3, H, W)`` in [0, 1].

    ``trail_actions``: ``left_hand`` / ``right_hand`` of shape ``(S, 1, P, D)``; with
    ``D >= 4`` channel 3 is per-step presence. ``extrinsics (V, 4, 4)`` c2w,
    ``intrinsics (V, 3, 3)`` normalised, ``step_opacity (S,)`` in [0, 1].
    """
    if not trail_actions:
        return images

    opac = np.asarray(step_opacity, dtype=np.float64).reshape(-1)
    if not np.any(opac > 0.0):
        return images

    V, _, H, W = images.shape
    out = images.clone()
    draw_order = np.argsort(opac, kind="stable")  # faintest first, most opaque on top

    for v in range(V):
        img_np = (out[v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
        R, Tvec, K_px = c2w_and_norm_K_to_w2c(
            extrinsics[v].cpu().numpy(), intrinsics[v].cpu().numpy(), H, W
        )
        for s in draw_order:
            alpha = float(opac[s])
            if alpha <= 0.0:
                continue
            for key in ("left_hand", "right_hand"):
                joints = trail_actions.get(key)
                if joints is None or s >= joints.shape[0]:
                    continue
                if joints.shape[-1] >= 4 and not bool((joints[s, 0, :, 3] > 0.5).any()):
                    continue
                jt = joints[s, 0, :, :3].cpu().numpy()
                draw_hand_skeleton_alpha(
                    img_np, jt, K_px, R, Tvec, point_color, alpha, radius=point_radius
                )
        out[v] = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    return out


def overlay_hand_skeletons(
    images: Tensor,
    actions: dict[str, Tensor] | None,
    extrinsics: Tensor,
    intrinsics: Tensor,
) -> Tensor:
    """Draw hand skeletons on ``images (T, V, 3, H, W)`` in [0, 1].

    ``actions``: ``left_hand`` / ``right_hand`` of shape ``(T, 1, 21, D)``; ``D == 3``
    always draws, ``D >= 4`` gates on the presence channel. ``extrinsics (T, V, 4, 4)``
    c2w, ``intrinsics (T, V, 3, 3)`` normalised. Returns a fresh tensor.
    """
    if actions is None:
        return images

    T, V, C, H, W = images.shape
    out = images.clone()
    hand_keys = [("left_hand", HAND_COLORS["left"]), ("right_hand", HAND_COLORS["right"])]

    for t in range(T):
        for v in range(V):
            img_np = (out[t, v].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8).copy()
            R, Tvec, K_px = c2w_and_norm_K_to_w2c(
                extrinsics[t, v].cpu().numpy(), intrinsics[t, v].cpu().numpy(), H, W
            )
            for key, color in hand_keys:
                if key not in actions:
                    continue
                joints = actions[key]
                if joints.shape[-1] >= 4 and not bool((joints[t, 0, :, 3] > 0.5).any()):
                    continue
                draw_hand_skeleton(img_np, joints[t, 0, :, :3].cpu().numpy(), K_px, R, Tvec, color)
            out[t, v] = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    return out
