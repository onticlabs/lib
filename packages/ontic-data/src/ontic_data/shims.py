"""Example-level transforms: random reflection, rescale + centre crop, patch crop.

All shims take and return a :class:`~ontic_data.example.BatchedTempExample`-shaped dict
whose views hold ``(*batch, C, H, W)`` images and ``(*batch, 3, 3)`` normalised
intrinsics. Image-like keys: ``image``, ``depth``, ``static_float``, ``state_mask``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

IMAGE_KEYS = ("image", "depth", "static_float", "state_mask")


# --------------------------------------------------------------------------- #
# Augmentation: horizontal reflection
# --------------------------------------------------------------------------- #


def apply_augmentation_shim(example: dict, generator: torch.Generator | None = None) -> dict:
    """With probability 1/2 mirror every view horizontally (images and cameras)."""
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example
    return {
        **example,
        "context": _reflect_views(example["context"]),
        "target": _reflect_views(example["target"]),
    }


def _reflect_views(views: dict) -> dict:
    keys = [k for k in IMAGE_KEYS if k in views]
    return {
        **views,
        "extrinsics": _reflect_extrinsics(views["extrinsics"]),
        **{k: views[k].flip(-1) for k in keys},
    }


def _reflect_extrinsics(extrinsics: Tensor) -> Tensor:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


# --------------------------------------------------------------------------- #
# Crop: rescale so the target fits, then centre crop
# --------------------------------------------------------------------------- #


def apply_crop_shim(example: dict, shape: tuple[int, int]) -> dict:
    """Rescale + centre-crop every view to ``shape=(H, W)`` and fix the intrinsics."""
    return {
        **example,
        "context": _crop_views(example["context"], shape),
        "target": _crop_views(example["target"], shape),
    }


def _crop_views(views: dict, shape: tuple[int, int]) -> dict:
    keys = [k for k in IMAGE_KEYS if k in views]
    images = [views[k] for k in keys]
    images, intrinsics = _rescale_and_crop(images, views["intrinsics"], shape)
    return {**views, "intrinsics": intrinsics, **dict(zip(keys, images))}


def _rescale_and_crop(
    images: list[Tensor], intrinsics: Tensor, shape: tuple[int, int]
) -> tuple[list[Tensor], Tensor]:
    *_, h_in, w_in = images[0].shape
    h_out, w_out = shape
    if h_out > h_in or w_out > w_in:
        raise ValueError(f"cannot upsample from ({h_in}, {w_in}) to ({h_out}, {w_out})")

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)

    rescaled = [_rescale(x, (h_scaled, w_scaled)) for x in images]
    return _center_crop(rescaled, intrinsics, shape)


def _center_crop(
    images: list[Tensor], intrinsics: Tensor, shape: tuple[int, int]
) -> tuple[list[Tensor], Tensor]:
    *_, h_in, w_in = images[0].shape
    h_out, w_out = shape
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2
    cropped = [x[..., row : row + h_out, col : col + w_out] for x in images]

    # Normalised intrinsics change only under cropping, not under rescaling.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out
    intrinsics[..., 1, 1] *= h_in / h_out
    return cropped, intrinsics


def _rescale(images: Tensor, shape: tuple[int, int]) -> Tensor:
    """Resize ``(*batch, C, h, w)`` to ``shape``: antialiased bilinear for RGB, nearest
    for everything else (masks, depth) so no new values are invented."""
    *batch, c, h, w = images.shape
    if (h, w) == tuple(shape):
        return images
    flat = images.reshape(-1, c, h, w)
    if c == 3:
        out = F.interpolate(flat, size=shape, mode="bilinear", align_corners=False, antialias=True)
    else:
        out = F.interpolate(flat.float(), size=shape, mode="nearest").to(images.dtype)
    return out.reshape(*batch, c, *shape)


# --------------------------------------------------------------------------- #
# Patch shim: crop to a multiple of the patch size
# --------------------------------------------------------------------------- #


def apply_patch_shim(example: dict, patch_size: int) -> dict:
    """Centre-crop every view so ``H`` and ``W`` are multiples of ``patch_size``."""
    return {
        **example,
        "context": _patch_views(example["context"], patch_size),
        "target": _patch_views(example["target"], patch_size),
    }


def _patch_views(views: dict, patch_size: int) -> dict:
    *_, h, w = views["image"].shape
    if h % 2 or w % 2:
        raise ValueError("image size must be even for an aligned centre crop")
    h_new = (h // patch_size) * patch_size
    w_new = (w // patch_size) * patch_size
    row = (h - h_new) // 2
    col = (w - w_new) // 2

    out = dict(views)
    for key in IMAGE_KEYS:
        if key in views:
            out[key] = views[key][..., row : row + h_new, col : col + w_new]
    intrinsics = views["intrinsics"].clone()
    intrinsics[..., 0, 0] *= w / w_new
    intrinsics[..., 1, 1] *= h / h_new
    out["intrinsics"] = intrinsics
    return out
