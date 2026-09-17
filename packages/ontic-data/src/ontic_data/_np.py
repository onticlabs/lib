"""Numpy adapters over the torch camera helpers, for calibration parsing."""

from __future__ import annotations

import numpy as np
import torch

from ontic_lib.camera.intrinsics import normalize_intrinsics
from ontic_lib.transforms.rigid import invert_rigid_transform


def w2c_to_c2w_np(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-to-camera ``R (3,3)``, ``t (3,)`` -> camera-to-world ``(4,4)`` float32."""
    w2c = torch.eye(4, dtype=torch.float64)
    w2c[:3, :3] = torch.as_tensor(np.asarray(R, dtype=np.float64))
    w2c[:3, 3] = torch.as_tensor(np.asarray(t, dtype=np.float64).ravel())
    return invert_rigid_transform(w2c).to(torch.float32).numpy()


def normalize_intrinsics_np(K: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel-space ``K (3,3)`` -> normalised ``[0, 1]`` intrinsics, float32."""
    K_t = torch.as_tensor(np.asarray(K, dtype=np.float32))
    return normalize_intrinsics(K_t, (int(height), int(width))).numpy()
