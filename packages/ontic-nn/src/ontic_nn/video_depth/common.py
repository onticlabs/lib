"""Shared contract for metric, per-camera temporal depth inference."""

from contextlib import contextmanager
from pathlib import Path
import sys

import torch


@contextmanager
def research_checkout(repo_path, package, relative_module):
    """Temporarily expose an explicit source checkout; reject mixed source versions."""
    if not repo_path:
        yield
        return
    root = Path(repo_path).expanduser().resolve()
    module = root / relative_module
    if not module.is_file():
        raise ValueError(f"Research checkout must contain {relative_module}: {root}")
    parts = Path(relative_module).parts
    source = root.joinpath(*parts[: parts.index(package)])
    loaded = sys.modules.get(package)
    if loaded is not None:
        locations = list(getattr(loaded, "__path__", []))
        if getattr(loaded, "__file__", None):
            locations.append(loaded.__file__)
        if not locations or any(not Path(p).resolve().is_relative_to(root) for p in locations):
            raise ImportError(f"Another {package} checkout is loaded; restart to change sources")
    sys.path.insert(0, str(source))
    try:
        yield
    finally:
        sys.path.remove(str(source))


def camera_videos(images, infer, *, progress, check_cancel):
    """Call infer(T,3,H,W) once per camera; return CPU float32 (B,T,V,H,W)."""
    if images.ndim != 6 or images.shape[3] != 3 or min(images.shape) < 1:
        raise ValueError("Video depth expects nonempty RGB images (B,T,V,3,H,W)")
    if not images.is_floating_point() or not bool(
        (torch.isfinite(images) & (images >= 0) & (images <= 1)).all()
    ):
        raise ValueError("Video depth expects finite RGB floats in [0,1]")
    b, t, v, _, h, w = images.shape
    output = torch.empty(b, t, v, h, w, dtype=torch.float32)
    for bi in range(b):
        for vi in range(v):
            check_cancel()
            progress(f"Video depth · sequence {bi + 1}/{b} · camera {vi + 1}/{v} · {t} frames")
            depth = infer(images[bi, :, vi])
            check_cancel()
            depth = torch.as_tensor(depth, dtype=torch.float32, device="cpu")
            if depth.shape != (t, h, w):
                raise ValueError(f"Video depth returned {tuple(depth.shape)}; expected {(t, h, w)}")
            output[bi, :, vi] = torch.where(torch.isfinite(depth) & (depth > 0), depth, 0)
    return output
