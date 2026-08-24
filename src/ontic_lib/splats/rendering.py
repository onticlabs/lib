"""3D Gaussian rasterization through gsplat, in ontic_lib conventions.

One scene (shared Gaussians), ``V`` pinhole cameras. Inputs follow the
package conventions — camera-to-world poses and normalized intrinsics — and
are converted to gsplat's world-to-camera / pixel-intrinsics form internally.
Requires the ``gsplat`` extra (``pip install ontic-lib[gsplat]``) and CUDA.

Distilled from the fwomo-3d rasterizer wrapper: consecutive cameras sharing
``(near, far)`` are rendered in one multi-camera gsplat call (gsplat takes
scalar near/far per call), and a boolean Gaussian mask is converted to long
indices once to avoid repeated host syncs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from ..camera.intrinsics import denormalize_intrinsics
from ..transforms.rigid import invert_rigid_transform

RenderMode = Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"]


@dataclass
class RenderOutput:
    """Rasterizer outputs, channels-first."""

    rgb: Tensor  # (V, C, H, W) — C=3, plus extra color channels if given
    alpha: Tensor  # (V, 1, H, W)
    depth: Tensor | None  # (V, 1, H, W) when render_mode includes depth, else None


def _gsplat():
    try:
        from gsplat import rasterization
    except ImportError as error:
        raise RuntimeError(
            "splats.rendering requires gsplat — install the extra: pip install ontic-lib[gsplat]"
        ) from error
    return rasterization


def render_gaussians(
    means: Tensor,
    covariances: Tensor,
    opacities: Tensor,
    *,
    camera_to_world: Tensor,
    intrinsics_normalized: Tensor,
    image_size: tuple[int, int],
    sh_coefficients: Tensor | None = None,
    colors: Tensor | None = None,
    near: float | Tensor = 0.01,
    far: float | Tensor = 1e3,
    background: Tensor | None = None,
    mask: Tensor | None = None,
    render_mode: RenderMode = "RGB+D",
) -> RenderOutput:
    """Rasterize one set of 3D Gaussians into ``V`` views.

    Args:
        means: ``(G, 3)`` world-space centers.
        covariances: ``(G, 3, 3)`` world-space covariances.
        opacities: ``(G,)`` in ``[0, 1]``.
        camera_to_world: ``(V, 4, 4)`` poses (package convention).
        intrinsics_normalized: ``(V, 3, 3)`` normalized intrinsics.
        image_size: ``(height, width)`` of the rendered images.
        sh_coefficients: ``(G, K, 3)`` view-independent spherical-harmonic
            coefficients with a complete band count ``K = (degree + 1)^2``;
            gsplat evaluates them per camera. Exactly one of
            ``sh_coefficients`` / ``colors`` must be given.
        colors: post-activation per-Gaussian colors — ``(G, C)`` shared or
            ``(V, G, C)`` view-dependent, any channel count ``C >= 1``.
        near, far: scalars or ``(V,)`` per-view clip planes. Consecutive views
            sharing both are rendered in one gsplat call.
        background: ``(C,)`` or ``(V, C)`` background color (zeros if None).
        mask: optional ``(G,)`` bool — render only the selected Gaussians.
        render_mode: gsplat render mode; ``"...D"`` = accumulated z-depth,
            ``"...ED"`` = alpha-normalized expected depth.
    """
    if (sh_coefficients is None) == (colors is None):
        raise ValueError("provide exactly one of sh_coefficients or colors")
    if means.ndim != 2 or means.shape[-1] != 3:
        raise ValueError(f"expected (G, 3) means, got {tuple(means.shape)}")
    rasterization = _gsplat()
    views = camera_to_world.shape[0]
    height, width = image_size

    sh_degree = None
    if sh_coefficients is not None:
        bands = sh_coefficients.shape[-2]
        sh_degree = math.isqrt(bands) - 1
        if (sh_degree + 1) ** 2 != bands:
            raise ValueError(f"SH coefficients must have (degree+1)^2 bands, got {bands}")

    if mask is not None:
        index = mask.nonzero(as_tuple=True)[0] if mask.dtype == torch.bool else mask
        means, covariances, opacities = means[index], covariances[index], opacities[index]
        if sh_coefficients is not None:
            sh_coefficients = sh_coefficients[index]
        elif colors is not None:
            colors = colors[index] if colors.ndim == 2 else colors[:, index]

    channels = 3 if sh_coefficients is not None else colors.shape[-1]
    if background is None:
        background = means.new_zeros(views, channels)
    elif background.ndim == 1:
        background = background.expand(views, -1)

    near_list = torch.as_tensor(near).expand(views).tolist()
    far_list = torch.as_tensor(far).expand(views).tolist()
    viewmats = invert_rigid_transform(camera_to_world)
    pixel_intrinsics = denormalize_intrinsics(intrinsics_normalized, image_size)

    rendered: list[Tensor] = []
    alphas: list[Tensor] = []
    start = 0
    while start < views:  # group consecutive cameras sharing (near, far)
        end = start + 1
        while (
            end < views
            and near_list[end] == near_list[start]
            and far_list[end] == far_list[start]
        ):
            end += 1
        if sh_coefficients is not None:
            call_colors = sh_coefficients
        else:
            call_colors = colors if colors.ndim == 2 else colors[start:end]
        color, alpha, _ = rasterization(
            means=means,
            covars=covariances,
            opacities=opacities,
            colors=call_colors,
            quats=None,
            scales=None,
            viewmats=viewmats[start:end],
            Ks=pixel_intrinsics[start:end],
            width=width,
            height=height,
            near_plane=near_list[start],
            far_plane=far_list[start],
            packed=False,
            rasterize_mode="classic",
            camera_model="pinhole",
            render_mode=render_mode,
            sh_degree=sh_degree,
            backgrounds=background[start:end],
        )
        rendered.append(color)
        alphas.append(alpha)
        start = end

    color = torch.cat(rendered, dim=0).permute(0, 3, 1, 2)  # (V, D, H, W)
    alpha = torch.cat(alphas, dim=0).permute(0, 3, 1, 2)
    if render_mode in ("RGB+D", "RGB+ED"):
        return RenderOutput(rgb=color[:, :-1], alpha=alpha, depth=color[:, -1:])
    if render_mode in ("D", "ED"):
        return RenderOutput(rgb=color[:, :0], alpha=alpha, depth=color)
    return RenderOutput(rgb=color, alpha=alpha, depth=None)
