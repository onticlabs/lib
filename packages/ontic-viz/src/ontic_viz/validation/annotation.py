"""Text labels above images, drawn with PIL (``ontic-viz[images]``)."""

from __future__ import annotations

from pathlib import Path
from string import ascii_letters, digits, punctuation

import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from ontic_viz.validation.layout import vcat

EXPECTED_CHARACTERS = digits + punctuation + ascii_letters
#: Looked up relative to the working directory; PIL's bundled font is used when it is absent.
DEFAULT_FONT = Path("assets/Inter-Regular.otf")


def draw_label(
    text: str,
    font: Path | str = DEFAULT_FONT,
    font_size: int = 24,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Draw a black label on a white background with no border, as ``(3, height, width)``.

    The label height is that of the full ``EXPECTED_CHARACTERS`` set, so labels of one font
    size line up whatever their text.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as e:
        raise ImportError("text labels need pillow; install ontic-viz[images]") from e

    try:
        pil_font = ImageFont.truetype(str(font), font_size)
    except OSError:
        pil_font = ImageFont.load_default(font_size)
    left, _, right, _ = pil_font.getbbox(text)
    width = right - left
    _, top, _, bottom = pil_font.getbbox(EXPECTED_CHARACTERS)
    height = bottom - top
    image = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(image)
    draw.text((0, 0), text, font=pil_font, fill="black")
    image = torch.tensor(np.array(image) / 255, dtype=torch.float32, device=device)
    return rearrange(image, "h w c -> c h w")


def add_label(
    image: Tensor,
    label: str,
    font: Path | str = DEFAULT_FONT,
    font_size: int = 24,
) -> Tensor:
    """Add a text label above a ``(3, height, width)`` image (left aligned, 4 px gap)."""
    return vcat(
        draw_label(label, font, font_size, image.device),
        image,
        align="left",
        gap=4,
    )
