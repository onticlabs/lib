"""Layout utilities for ``(channel, height, width)`` float32 images.

- ``add_border``: pad an image with a solid colour.
- ``cat`` / ``hcat`` / ``vcat``: join images in a line with alignment and gaps.
- ``overlay``: paste a smaller image onto a larger one.
- ``resize``: resize preserving aspect ratio.
"""

from __future__ import annotations

from typing import Any, Generator, Iterable, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

Alignment = Literal["start", "center", "end"]
Axis = Literal["horizontal", "vertical"]
#: A scalar, an iterable of per-channel values, or a ``(#channel,)`` / ``()`` tensor.
Color = Union[int, float, Iterable[int], Iterable[float], Tensor]


def _sanitize_color(color: Color) -> Tensor:
    if isinstance(color, torch.Tensor):
        color = color.tolist()
    if isinstance(color, Iterable):
        color = list(color)
    else:
        color = [color]
    return torch.tensor(color, dtype=torch.float32)


def _intersperse(iterable: Iterable, delimiter: Any) -> Generator[Any, None, None]:
    it = iter(iterable)
    yield next(it)
    for item in it:
        yield delimiter
        yield item


def _get_main_dim(main_axis: Axis) -> int:
    return {"horizontal": 2, "vertical": 1}[main_axis]


def _get_cross_dim(main_axis: Axis) -> int:
    return {"horizontal": 1, "vertical": 2}[main_axis]


def _compute_offset(base: int, overlay: int, align: Alignment) -> slice:
    assert base >= overlay
    offset = {"start": 0, "center": (base - overlay) // 2, "end": base - overlay}[align]
    return slice(offset, offset + overlay)


def overlay(
    base: Tensor,
    overlay: Tensor,
    main_axis: Axis,
    main_axis_alignment: Alignment,
    cross_axis_alignment: Alignment,
) -> Tensor:
    """Paste ``overlay`` onto a copy of ``base``; both are ``(channel, height, width)``."""
    _, base_height, base_width = base.shape
    _, overlay_height, overlay_width = overlay.shape
    assert base_height >= overlay_height and base_width >= overlay_width

    main_dim = _get_main_dim(main_axis)
    main_slice = _compute_offset(base.shape[main_dim], overlay.shape[main_dim], main_axis_alignment)

    cross_dim = _get_cross_dim(main_axis)
    cross_slice = _compute_offset(
        base.shape[cross_dim], overlay.shape[cross_dim], cross_axis_alignment
    )

    selector = [..., None, None]
    selector[main_dim] = main_slice
    selector[cross_dim] = cross_slice
    result = base.clone()
    result[tuple(selector)] = overlay
    return result


def cat(
    main_axis: Axis,
    *images: Tensor,
    align: Alignment = "center",
    gap: int = 8,
    gap_color: Color = 1,
) -> Tensor:
    """Arrange images in a line. The interface resembles a CSS div with flexbox."""
    device = images[0].device
    gap_color = _sanitize_color(gap_color).to(device)

    cross_dim = _get_cross_dim(main_axis)
    cross_axis_length = max(image.shape[cross_dim] for image in images)

    padded_images = []
    for image in images:
        padded_shape = list(image.shape)
        padded_shape[cross_dim] = cross_axis_length
        base = torch.ones(padded_shape, dtype=torch.float32, device=device)
        base = base * gap_color[:, None, None]
        padded_images.append(overlay(base, image, main_axis, "start", align))

    if gap > 0:
        c, _, _ = images[0].shape
        separator_size = [gap, gap]
        separator_size[cross_dim - 1] = cross_axis_length
        separator = torch.ones((c, *separator_size), dtype=torch.float32, device=device)
        separator = separator * gap_color[:, None, None]
        padded_images = list(_intersperse(padded_images, separator))

    return torch.cat(padded_images, dim=_get_main_dim(main_axis))


def hcat(
    *images: Tensor,
    align: Literal["start", "center", "end", "top", "bottom"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
) -> Tensor:
    """Shorthand for horizontal concatenation."""
    return cat(
        "horizontal",
        *images,
        align={"start": "start", "center": "center", "end": "end", "top": "start", "bottom": "end"}[
            align
        ],
        gap=gap,
        gap_color=gap_color,
    )


def vcat(
    *images: Tensor,
    align: Literal["start", "center", "end", "left", "right"] = "start",
    gap: int = 8,
    gap_color: Color = 1,
) -> Tensor:
    """Shorthand for vertical concatenation."""
    return cat(
        "vertical",
        *images,
        align={"start": "start", "center": "center", "end": "end", "left": "start", "right": "end"}[
            align
        ],
        gap=gap,
        gap_color=gap_color,
    )


def add_border(image: Tensor, border: int = 8, color: Color = 1) -> Tensor:
    """Pad a ``(channel, height, width)`` image with ``border`` pixels of ``color``."""
    color = _sanitize_color(color).to(image)
    c, h, w = image.shape
    result = torch.empty(
        (c, h + 2 * border, w + 2 * border), dtype=torch.float32, device=image.device
    )
    result[:] = color[:, None, None]
    result[:, border : h + border, border : w + border] = image
    return result


def resize(
    image: Tensor,
    shape: Optional[tuple[int, int]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> Tensor:
    """Bilinear resize of a ``(channel, height, width)`` image; give exactly one target."""
    assert (shape is not None) + (width is not None) + (height is not None) == 1
    _, h, w = image.shape

    if width is not None:
        shape = (int(h * width / w), width)
    elif height is not None:
        shape = (height, int(w * height / h))

    return F.interpolate(
        image[None],
        shape,
        mode="bilinear",
        align_corners=False,
        antialias="bilinear",
    )[0]
