"""Matplotlib figures and float tensors to arrays and PNG files (PIL, ``ontic-viz[images]``)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Union

import numpy as np
import torch
from einops import rearrange, repeat
from matplotlib.figure import Figure
from torch import Tensor

#: ``(height, width)``, ``(channel, height, width)`` or ``(batch, channel, height, width)``.
FloatImage = Tensor


def fig_to_image(
    fig: Figure,
    dpi: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Convert a matplotlib Figure to a ``(3, height, width)`` float tensor."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="raw", dpi=dpi)
    buffer.seek(0)
    data = np.frombuffer(buffer.getvalue(), dtype=np.uint8)
    h = int(fig.bbox.bounds[3])
    w = int(fig.bbox.bounds[2])
    data = rearrange(data, "(h w c) -> c h w", h=h, w=w, c=4)
    buffer.close()
    return (torch.tensor(data, device=device, dtype=torch.float32) / 255)[:3]


def prep_image(image: FloatImage) -> np.ndarray:
    """Convert a float tensor in [0, 1] to a uint8 ``(height, width, channel)`` array.

    A batch is laid out side by side; one channel is repeated to three.
    """
    if image.ndim == 4:
        image = rearrange(image, "b c h w -> c h (b w)")
    if image.ndim == 2:
        image = rearrange(image, "h w -> () h w")

    channel, _, _ = image.shape
    if channel == 1:
        image = repeat(image, "() h w -> c h w", c=3)
    assert image.shape[0] in (3, 4)

    image = (image.detach().clip(min=0, max=1) * 255).type(torch.uint8)
    return rearrange(image, "c h w -> h w c").cpu().numpy()


def save_image(image: FloatImage, path: Union[Path, str]) -> None:
    """Save an image assumed to be in range 0-1; parent directories are created."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("saving images needs pillow; install ontic-viz[images]") from e
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    Image.fromarray(prep_image(image)).save(path)
