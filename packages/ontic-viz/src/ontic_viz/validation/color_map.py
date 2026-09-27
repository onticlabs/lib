"""Colour maps for scalar images; ``depth_map`` is the log-space turbo depth colouring."""

from __future__ import annotations

import matplotlib
import torch
from einops import rearrange
from torch import Tensor


def apply_color_map(x: Tensor, color_map: str = "inferno") -> Tensor:
    """Map ``(*batch,)`` values in [0, 1] to ``(*batch, 3)`` RGB floats."""
    cmap = matplotlib.colormaps[color_map]
    mapped = cmap(x.detach().clip(min=0, max=1).cpu().numpy())[..., :3]
    return torch.tensor(mapped, device=x.device, dtype=torch.float32)


def apply_color_map_to_image(image: Tensor, color_map: str = "inferno") -> Tensor:
    """Map a ``(*batch, height, width)`` image to ``(*batch, 3, height, width)`` RGB."""
    image = apply_color_map(image, color_map)
    return rearrange(image, "... h w c -> ... c h w")


def depth_map(result: Tensor) -> Tensor:
    """Color-map depth values using log-space normalization and the turbo colormap.

    ``near`` / ``far`` are the 1 % / 99 % quantiles (of the positive values / of all values);
    when a quantile cannot be taken (no positive values, non-finite input, too many
    elements) they fall back to the clipped min / the max. Near is red, far is blue.
    """
    try:
        near = result[result > 0][:16_000_000].quantile(0.01).log()
    except Exception:
        near = result.min().clip(0.0).log()
    try:
        far = result.view(-1)[:16_000_000].quantile(0.99).log()
    except Exception:
        far = result.max().log()

    result = result.log()
    result = 1 - (result - near) / (far - near)
    return apply_color_map_to_image(result, "turbo")
