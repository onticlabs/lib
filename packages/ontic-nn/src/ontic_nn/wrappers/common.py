"""Shared types and helpers for the pretrained multi-view backbone wrappers.

Every wrapper turns ``images (B, V, 3, H, W)`` in ``[0, 1]`` (plus optional GT cameras) into a
:class:`BackboneOutput`: depth / confidence at the wrapper's ``long_side``-resized input
resolution, four ViT patch-feature taps at patch resolution, and cameras. Extrinsics are
camera-to-world ``(B, V, 4, 4)``; intrinsics are normalised ``[0, 1]`` ``(B, V, 3, 3)``.

The research packages the wrappers sit on are imported lazily inside ``build()`` / ``forward``
and raise an ``ImportError`` naming the ``ontic-nn[<extra>]`` extra and the repository.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple, TypedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config / module base
# ---------------------------------------------------------------------------
@dataclass(kw_only=True)
class BackboneConfig:
    """Base config; subclasses add model knobs and implement :meth:`build`.

    ``long_side`` is the canonical input resolution (longest image side, patch-snapped): the
    wrapper resizes its input internally, so the backbone owns its resolution. Checkpoint
    resolution: ``checkpoint_path`` (a pre-downloaded file or snapshot directory) is used
    first; else the model's default hub repo is fetched into ``cache_dir`` (``None`` → the
    HuggingFace / torch hub cache); ``allow_download=False`` serves from the cache only.
    """

    freeze_backbone: bool = True
    freeze_dpt_head: bool = True
    freeze_cam_dec: bool = True
    freeze_cam_enc: bool = True
    long_side: int = 518
    gradient_checkpointing: bool = False
    checkpoint_path: Optional[str] = None
    cache_dir: Optional[str] = None
    allow_download: bool = True

    def build(self) -> BackboneBase:
        raise NotImplementedError


class BackboneBase(nn.Module):
    """Abstract wrapper: ``forward(images, extrinsics, intrinsics, depth) -> BackboneOutput``.

    ``depth (B, V, 1, H, W)`` is ground-truth depth from the batch; only ``gtdepth`` consumes
    it. ``ACCEPTS_GT_CAMERAS`` says whether GT cameras *condition* the model (DA3 / MA / Pi3X)
    rather than being passed through by a pose-free model.
    """

    PATCH_SIZE: int = 14
    ACCEPTS_GT_CAMERAS: bool = False

    if TYPE_CHECKING:

        def __call__(self, *args, **kwargs) -> BackboneOutput: ...

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        extrinsics: Optional[Tensor] = None,
        intrinsics: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
    ) -> BackboneOutput: ...

    @property
    def patch_size(self) -> int:
        return self.PATCH_SIZE

    @property
    def accepts_gt_cameras(self) -> bool:
        return self.ACCEPTS_GT_CAMERAS

    @property
    @abstractmethod
    def encoder_dim(self) -> int:
        """Channel dimension of the patch features."""
        ...


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------
@dataclass
class BackboneOutput:
    """All ``(B, V, ...)`` tensors of a backbone forward, plus the three resolutions.

    ``patch_resolution (Ph, Pw)`` is the ViT grid, ``dpt_resolution (Hd, Wd)`` the depth grid
    and ``input_resolution (H, W)`` the ``long_side``-resized input; depth, confidence and
    sky mask live at ``dpt_resolution``, patch features at ``patch_resolution``.
    ``depth_conf`` uses the ``expp1`` convention (``exp(x) + 1``, so ``>= 1``).
    """

    class DataDict(TypedDict, total=False):
        depth: Tensor  # (B, V, Hd, Wd)
        depth_conf: Tensor  # (B, V, Hd, Wd), >= 1
        sky_mask: Tensor  # (B, V, Hd, Wd), 1 = sky / invalid
        patch_feat_0: Tensor  # (B, V, Ph, Pw, C)
        patch_feat_1: Tensor
        patch_feat_2: Tensor
        patch_feat_3: Tensor
        extrinsics: Tensor  # GT c2w (B, V, 4, 4)
        intrinsics: Tensor  # GT normalised (B, V, 3, 3)
        extrinsics_pred: Tensor  # predicted c2w (B, V, 4, 4)
        intrinsics_pred: Tensor  # predicted normalised (B, V, 3, 3)

    data: DataDict = field(default_factory=dict)
    input_resolution: Tuple[int, int] = (0, 0)
    dpt_resolution: Tuple[int, int] = (0, 0)
    patch_resolution: Tuple[int, int] = (0, 0)
    patch_feature_keys: Tuple[str, ...] = (
        "patch_feat_0",
        "patch_feat_1",
        "patch_feat_2",
        "patch_feat_3",
    )

    @property
    def patch_features(self) -> List[Tensor]:
        return [self.data[k] for k in self.patch_feature_keys if k in self.data]

    def get_extrinsics(self) -> Optional[Tensor]:
        """GT extrinsics when given, else the predicted ones, else ``None``."""
        return self.data.get("extrinsics", self.data.get("extrinsics_pred", None))

    def get_intrinsics(self) -> Optional[Tensor]:
        """GT intrinsics when given, else the predicted ones, else ``None``."""
        return self.data.get("intrinsics", self.data.get("intrinsics_pred", None))

    @property
    def resolution(self) -> Tuple[int, int]:
        return tuple(self.data["depth"].shape[-2:])


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def resize_to_long_side(images: Tensor, patch: int, long_side: int) -> Tensor:
    """Bilinearly resize ``(..., C, H, W)`` so the longest side is ``long_side``, snapped to ``patch``.

    Both output dims are multiples of ``patch`` (at least one patch); the aspect ratio is kept
    up to the snapping. Returns the input tensor itself when the shape already matches.
    """
    h, w = images.shape[-2:]
    scale = long_side / max(h, w)
    h2 = max(patch, round(h * scale / patch) * patch)
    w2 = max(patch, round(w * scale / patch) * patch)
    if (h2, w2) == (h, w):
        return images
    lead = images.shape[:-3]
    flat = images.reshape(-1, *images.shape[-3:])
    flat = F.interpolate(flat, size=(h2, w2), mode="bilinear", align_corners=False)
    return flat.reshape(*lead, *flat.shape[-3:])


def amp_dtype() -> torch.dtype:
    """bfloat16 where supported, else float16 (the autocast dtype the upstream demos use)."""
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def normalize_intrinsics_pixels(intrinsics_pixels: Tensor, height: int, width: int) -> Tensor:
    """Pixel-space ``(..., 3, 3)`` intrinsics → normalised ``[0, 1]`` at ``(height, width)``."""
    from ontic_lib.camera.intrinsics import normalize_intrinsics

    return normalize_intrinsics(intrinsics_pixels, (height, width))


def as_homogeneous_4x4(matrix: Tensor) -> Tensor:
    """Append the ``[0, 0, 0, 1]`` row to ``(..., 3, 4)`` matrices; ``(..., 4, 4)`` passes through."""
    if matrix.shape[-2:] == (4, 4):
        return matrix
    bottom = torch.zeros_like(matrix[..., :1, :])
    bottom[..., 0, 3] = 1.0
    return torch.cat([matrix, bottom], dim=-2)


# ---------------------------------------------------------------------------
# Parameter freezing
# ---------------------------------------------------------------------------
def convert_to_buffer(module: nn.Module, persistent: bool = True) -> None:
    """Recursively turn every parameter (and buffer) into a buffer with the same value.

    The module keeps working but exposes no parameters to optimizers / DDP.
    """
    for _name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)
    for name, value in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = value.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


def buffers_to_params(module: nn.Module) -> None:
    """Recursively turn every floating-point buffer back into a trainable ``nn.Parameter``."""
    for _name, child in list(module.named_children()):
        buffers_to_params(child)
    for name, buf in list(module.named_buffers(recurse=False)):
        if not (buf.is_floating_point() or buf.is_complex()):
            continue
        delattr(module, name)
        module.register_parameter(name, nn.Parameter(buf.clone()))


def extract_weights(state_dict: Dict[str, Tensor], prefix: str) -> Dict[str, Tensor]:
    """Entries of ``state_dict`` under ``prefix`` with the prefix stripped."""
    return {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}


def set_frozen(modules: Iterable[nn.Module | nn.Parameter], frozen: bool) -> None:
    """``requires_grad_(not frozen)`` on modules / parameters; frozen modules go to eval mode."""
    for m in modules:
        m.requires_grad_(not frozen)
        if isinstance(m, nn.Module):
            m.train(not frozen)


# ---------------------------------------------------------------------------
# Research-package imports and checkpoint resolution
# ---------------------------------------------------------------------------
def import_research_module(module: str, *, extra: str, repo_url: str, what: str):
    """``importlib.import_module(module)`` with an ImportError naming the extra and the repo."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        package = module.split(".")[0]
        raise ImportError(
            f"{what} requires the `{package}` package from {repo_url}; "
            f"install ontic-nn[{extra}] and that repository"
        ) from e


@contextlib.contextmanager
def offline_guard(allow_download: bool) -> Iterator[None]:
    """With ``allow_download=False``, keep HuggingFace and ``torch.hub`` cache-only for the block.

    Sets the HF offline env vars (and the already-imported ``huggingface_hub.constants`` flag),
    and replaces ``torch.hub.download_url_to_file`` — which the hub uses for both repo zips and
    checkpoints and which ignores the HF variables — with one that raises ``RuntimeError``.
    Everything is restored on exit.
    """
    if allow_download:
        yield
        return
    keys = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
    prev_env = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: "1" for k in keys})

    hf_constants = sys.modules.get("huggingface_hub.constants")
    prev_flag = getattr(hf_constants, "HF_HUB_OFFLINE", None)
    if hf_constants is not None:
        hf_constants.HF_HUB_OFFLINE = True

    import torch.hub

    real_download = torch.hub.download_url_to_file

    def _refuse(url, dst, *args, **kwargs):
        raise RuntimeError(
            f"allow_download=False, but torch.hub asked for {url!r} (not in the local cache); "
            "pre-download it or point checkpoint_path / cache_dir at a copy"
        )

    torch.hub.download_url_to_file = _refuse
    try:
        yield
    finally:
        torch.hub.download_url_to_file = real_download
        if hf_constants is not None:
            hf_constants.HF_HUB_OFFLINE = prev_flag
        for k, v in prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def resolve_checkpoint(
    checkpoint_path: Optional[str],
    repo_id: Optional[str],
    *,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None,
    allow_download: bool = True,
    extra: str = "",
    what: str = "checkpoint",
) -> Optional[str]:
    """Local path of a checkpoint: ``checkpoint_path`` first, else the HuggingFace hub.

    ``checkpoint_path`` is a file or a snapshot directory (a directory plus ``filename`` gives
    ``dir/filename``); a missing path raises ``FileNotFoundError``. Otherwise ``repo_id`` is
    fetched — one file when ``filename`` is given, else the whole snapshot — into ``cache_dir``
    (``None`` → the HF cache). ``allow_download=False`` serves from that cache only and raises
    ``RuntimeError`` on a miss. Returns ``None`` when neither source is given (random init).
    """
    if checkpoint_path:
        path = Path(checkpoint_path).expanduser()
        if path.is_file():
            return str(path)
        if path.is_dir():
            if filename is None:
                return str(path)
            if (path / filename).is_file():
                return str(path / filename)
            raise FileNotFoundError(f"{what}: {filename!r} not found in {path}")
        raise FileNotFoundError(f"{what}: checkpoint_path {checkpoint_path!r} does not exist")
    if not repo_id:
        return None
    try:
        import huggingface_hub
    except ImportError as e:
        hint = f"install ontic-nn[{extra}]" if extra else "pip install huggingface_hub"
        raise ImportError(
            f"fetching {what} {repo_id!r} needs the `huggingface_hub` package; {hint}"
        ) from e
    try:
        with offline_guard(allow_download):
            if filename is None:
                return huggingface_hub.snapshot_download(repo_id, cache_dir=cache_dir)
            return huggingface_hub.hf_hub_download(repo_id, filename, cache_dir=cache_dir)
    except Exception as e:
        where = cache_dir or "the HuggingFace cache"
        if not allow_download:
            raise RuntimeError(
                f"allow_download=False and {what} {repo_id!r} is not in {where} "
                f"({type(e).__name__}: {e})"
            ) from e
        raise RuntimeError(f"failed to fetch {what} {repo_id!r} ({type(e).__name__}: {e})") from e
