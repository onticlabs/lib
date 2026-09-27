"""Multi-sample rendering of a colour function over a pixel grid."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from einops import rearrange, reduce
from torch import Tensor


@runtime_checkable
class ColorFunction(Protocol):
    def __call__(self, xy: Tensor) -> Tensor:  # (point, 2) -> (point, 4) RGBA
        pass


def generate_sample_grid(shape: tuple[int, int], device: torch.device) -> Tensor:
    """Pixel-centre coordinates, ``(height, width, 2)``."""
    h, w = shape
    x = torch.arange(w, device=device) + 0.5
    y = torch.arange(h, device=device) + 0.5
    x, y = torch.meshgrid(x, y, indexing="xy")
    return torch.stack([x, y], dim=-1)


def detect_msaa_pixels(image: Tensor) -> Tensor:
    """Pixels of a ``(batch, 4, height, width)`` image that differ from any neighbour."""
    b, _, h, w = image.shape

    mask = torch.zeros((b, h, w), dtype=torch.bool, device=image.device)

    # Detect horizontal differences.
    horizontal = (image[:, :, :, 1:] != image[:, :, :, :-1]).any(dim=1)
    mask[:, :, 1:] |= horizontal
    mask[:, :, :-1] |= horizontal

    # Detect vertical differences.
    vertical = (image[:, :, 1:, :] != image[:, :, :-1, :]).any(dim=1)
    mask[:, 1:, :] |= vertical
    mask[:, :-1, :] |= vertical

    # Detect diagonal (top left to bottom right) differences.
    tlbr = (image[:, :, 1:, 1:] != image[:, :, :-1, :-1]).any(dim=1)
    mask[:, 1:, 1:] |= tlbr
    mask[:, :-1, :-1] |= tlbr

    # Detect diagonal (top right to bottom left) differences.
    trbl = (image[:, :, :-1, 1:] != image[:, :, 1:, :-1]).any(dim=1)
    mask[:, :-1, 1:] |= trbl
    mask[:, 1:, :-1] |= trbl

    return mask


def reduce_straight_alpha(rgba: Tensor) -> Tensor:
    """Average a ``(batch, 4, height, width)`` RGBA block to ``(batch, 4)`` straight alpha."""
    color, alpha = rgba.split((3, 1), dim=1)

    # Color becomes a weighted average of color (weighted by alpha).
    weighted_color = reduce(color * alpha, "b c h w -> b c", "sum")
    alpha_sum = reduce(alpha, "b c h w -> b c", "sum")
    color = weighted_color / (alpha_sum + 1e-10)

    # Alpha becomes mean alpha.
    alpha = reduce(alpha, "b c h w -> b c", "mean")

    return torch.cat((color, alpha), dim=-1)


@torch.no_grad()
def run_msaa_pass(
    xy: Tensor,
    color_function: ColorFunction,
    scale: float,
    subdivision: int,
    remaining_passes: int,
    device: torch.device,
    batch_size: int = int(2**16),
) -> Tensor:
    """Sample ``color_function`` at ``xy`` ``(batch, height, width, 2)`` and refine edge pixels
    recursively; returns ``(batch, 4, height, width)`` RGBA with straight alpha."""
    # Sample the color function.
    b, h, w, _ = xy.shape
    color = [
        color_function(batch) for batch in rearrange(xy, "b h w xy -> (b h w) xy").split(batch_size)
    ]
    color = torch.cat(color, dim=0)
    color = rearrange(color, "(b h w) c -> b c h w", b=b, h=h, w=w)

    # If any MSAA passes remain, subdivide.
    if remaining_passes > 0:
        mask = detect_msaa_pixels(color)
        batch_index, row_index, col_index = torch.where(mask)
        xy = xy[batch_index, row_index, col_index]

        offsets = generate_sample_grid((subdivision, subdivision), device)
        offsets = (offsets / subdivision - 0.5) * scale

        color_fine = run_msaa_pass(
            xy[:, None, None] + offsets,
            color_function,
            scale / subdivision,
            subdivision,
            remaining_passes - 1,
            device,
            batch_size=batch_size,
        )
        color[batch_index, :, row_index, col_index] = reduce_straight_alpha(color_fine)

    return color


@torch.no_grad()
def render(
    shape: tuple[int, int],
    color_function: ColorFunction,
    device: torch.device,
    subdivision: int = 8,
    num_passes: int = 2,
) -> Tensor:
    """Render ``color_function`` on a ``(height, width)`` canvas; ``(4, height, width)`` RGBA."""
    xy = generate_sample_grid(shape, device)
    return run_msaa_pass(
        xy[None],
        color_function,
        1.0,
        subdivision,
        num_passes,
        device,
    )[0]


def render_over_image(
    image: Tensor,
    color_function: ColorFunction,
    device: torch.device,
    subdivision: int = 8,
    num_passes: int = 1,
) -> Tensor:
    """Alpha-composite the rendered ``color_function`` over a ``(3, height, width)`` image."""
    _, h, w = image.shape
    overlay = render(
        (h, w),
        color_function,
        device,
        subdivision=subdivision,
        num_passes=num_passes,
    )
    color, alpha = overlay.split((3, 1), dim=0)
    return image * (1 - alpha) + color * alpha
