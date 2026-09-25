"""Build, cache and run one GPU-resident backbone; lift its depth into a metric world frame.

:class:`BackboneRunner` holds a single backbone (freeing the previous one's VRAM on
switch), runs it in inference mode, optionally conditions it on GT cameras (only when
the model ingests them, ``accepts_gt_cameras``), and returns a :class:`BackboneResult`
with the **raw** predicted-scale depth plus both predicted and GT cameras. The metric
alignment (:class:`AlignMode`) is chosen downstream by :func:`metric_unproject`, so it
can be changed without re-running.

Alignment modes (implemented by :func:`ontic_lib.pointops.align`):

* ``none`` — the backbone's own (predicted) frame.
* ``sim3_points`` — unproject with the predicted cameras, map the cloud into the GT
  frame with a Umeyama Sim(3) fit of predicted -> GT cameras.
* ``prescale_gt`` — scale depth by that Sim(3) scale, unproject with the GT cameras.
* ``metric_mono`` — scale depth and predicted camera centres by a global scale fitted
  to a metric monocular model (:meth:`BackboneRunner.compute_metric_scale`), then
  place them by a rigid SE(3) fit to the GT cameras.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ontic_lib.depth.alignment import fit_depth_scale
from ontic_lib.pointops import align
from ontic_lib.structures import PointCloud, pointcloud_from_depth_views

if TYPE_CHECKING:
    from ontic_nn.metric_depth import MetricDepthModel
    from ontic_nn.wrappers import BackboneBase

    from .data_source import Frame

# GUI hints keyed by registry name, used before a backbone is built. The built
# model's ``accepts_gt_cameras`` property is authoritative.
BACKBONE_ACCEPTS_GT_CAMERAS: dict[str, bool] = {
    "da3": True,
    "ma": True,
    "vggt": False,
    "pi3x": True,
    "dvlt": False,
    "moge3": False,
    "gtdepth": True,
}
# Backbones that consume ground-truth depth instead of predicting it.
BACKBONE_NEEDS_GT_DEPTH: dict[str, bool] = {"gtdepth": True}
# Monocular backbones emit no ``extrinsics_pred``; the Umeyama modes degrade to ``none``.
BACKBONE_IS_MONOCULAR: dict[str, bool] = {"moge3": True}


def backbone_accepts_gt_cameras(name: str) -> bool:
    return BACKBONE_ACCEPTS_GT_CAMERAS.get(name, False)


def backbone_needs_gt_depth(name: str) -> bool:
    return BACKBONE_NEEDS_GT_DEPTH.get(name, False)


def backbone_is_monocular(name: str) -> bool:
    return BACKBONE_IS_MONOCULAR.get(name, False)


class AlignMode(str, Enum):
    """How to put a backbone's depth into the GT metric world frame."""

    NONE = "none"
    SIM3_POINTS = "sim3_points"
    PRESCALE_GT = "prescale_gt"
    METRIC_MONO = "metric_mono"


ALIGN_MODES: list[str] = [m.value for m in AlignMode]


def backbone_names() -> list[str]:
    """Registered backbone keys (``ontic_nn.wrappers.BACKBONES``)."""
    from ontic_nn.wrappers import BACKBONES

    return list(BACKBONES)


def metric_model_names() -> list[str]:
    """Registered metric-model keys (``da3`` first, then the rest sorted)."""
    from ontic_nn.metric_depth import METRIC_MODELS

    keys = sorted(METRIC_MODELS)
    return (["da3"] if "da3" in keys else []) + [k for k in keys if k != "da3"]


def backbone_default_long_side(name: str) -> int:
    """Read the registered model's input size without loading its weights."""
    from ontic_nn.wrappers import BACKBONES

    return BACKBONES[name]().long_side


def build_backbone(name: str, long_side: int | None = None, **options) -> BackboneBase:
    """Build a pretrained backbone from its registered config; ``long_side`` overrides
    the backbone's canonical input resolution."""
    from ontic_nn.wrappers import BACKBONES

    if name not in BACKBONES:
        raise ValueError(f"unknown backbone {name!r}; choose from {list(BACKBONES)}")
    cfg = BACKBONES[name](**options)
    if long_side is not None:
        cfg.long_side = long_side
    return cfg.build()


def build_metric_model(name: str, long_side: int | None = None) -> MetricDepthModel:
    """Build a metric monocular depth model from its registered config."""
    from ontic_nn.metric_depth import METRIC_MODELS

    if name not in METRIC_MODELS:
        raise ValueError(f"unknown metric model {name!r}; choose from {sorted(METRIC_MODELS)}")
    cfg = METRIC_MODELS[name]()
    if long_side is not None:
        cfg.long_side = long_side
    return cfg.build()


@dataclass
class BackboneResult:
    """One backbone forward on the selected cameras, on the CPU.

    ``depth (V, Hd, Wd)`` is raw (predicted-scale); ``conf`` optional ``(V, Hd, Wd)``;
    ``pred_extrinsics (V, 4, 4)`` c2w / ``pred_intrinsics (V, 3, 3)`` normalised are
    ``None`` for backbones without a camera head; ``gt_*`` are the dataset cameras.
    ``metric_scale`` caches the ``metric_mono`` global scale once fitted.
    """

    depth: Tensor
    conf: Tensor | None
    pred_extrinsics: Tensor | None
    pred_intrinsics: Tensor | None
    gt_extrinsics: Tensor
    gt_intrinsics: Tensor
    conditioned: bool = False
    metric_scale: float | None = None


def effective_align_mode(result: BackboneResult, mode: AlignMode | str) -> AlignMode:
    """``mode`` with the Umeyama modes degraded to ``none`` when there are no predicted
    cameras (monocular backbones)."""
    mode = AlignMode(mode)
    if result.pred_extrinsics is None and mode in (AlignMode.PRESCALE_GT, AlignMode.SIM3_POINTS):
        return AlignMode.NONE
    return mode


def aligned_depth_and_cameras(
    result: BackboneResult, mode: AlignMode | str
) -> tuple[Tensor, Tensor, Tensor]:
    """``(depth (V, Hd, Wd), camera_to_world (V, 4, 4), intrinsics (V, 3, 3))`` to lift
    with for ``mode`` — the single-frame view of :func:`ontic_lib.pointops.align`."""
    mode = effective_align_mode(result, mode)
    if mode is AlignMode.METRIC_MONO and result.metric_scale is None:
        raise RuntimeError(
            "metric_mono: metric scale not computed; call "
            "BackboneRunner.compute_metric_scale(result, images, model) first"
        )
    pe, pi = result.pred_extrinsics, result.pred_intrinsics
    depth, cams, intr = align(
        result.depth.unsqueeze(0),
        None if pe is None else pe.unsqueeze(0),
        None if pi is None else pi.unsqueeze(0),
        result.gt_extrinsics.unsqueeze(0),
        result.gt_intrinsics.unsqueeze(0),
        mode.value,
        metric_scale=result.metric_scale,
    )
    return depth[0], cams[0], intr[0]


def metric_unproject(
    result: BackboneResult,
    rgb: Tensor,
    mode: AlignMode | str = AlignMode.NONE,
    *,
    stride: int = 1,
    conf_thresh: float = 0.0,
) -> PointCloud:
    """Unproject ``result`` into a coloured cloud under alignment ``mode``.

    ``rgb`` is the ``(V, 3, H, W)`` image painted onto the points.
    """
    depth, cams, intr = aligned_depth_and_cameras(result, mode)
    return pointcloud_from_depth_views(
        depth,
        cams,
        intr,
        rgb=rgb,
        confidence=result.conf,
        stride=stride,
        confidence_threshold=conf_thresh,
        minimum_depth=0.0,
    )


def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BackboneRunner:
    """Holds one backbone (and one metric model) at a time; rebuilds + frees VRAM on switch."""

    def __init__(
        self,
        device: str = "cuda",
        builder: Callable[..., BackboneBase] = build_backbone,
        long_side: int | None = None,
        metric_builder: Callable[..., MetricDepthModel] = build_metric_model,
    ):
        self.device = device
        self._builder = builder
        self._model_options: dict[str, dict] = {}
        self.long_side = long_side
        self._default_long_side: int | None = None
        self._name: str | None = None
        self._backbone: BackboneBase | None = None
        self._metric_builder = metric_builder
        self._metric_name: str | None = None
        self._metric_model: MetricDepthModel | None = None

    @property
    def current(self) -> str | None:
        return self._name

    @property
    def accepts_gt_cameras(self) -> bool:
        if self._backbone is None:
            return False
        return bool(getattr(self._backbone, "accepts_gt_cameras", False))

    def load(self, name: str) -> None:
        """Make ``name`` the active backbone, freeing the previous one's VRAM."""
        if name == self._name and self._backbone is not None:
            return
        self._free()
        backbone = self._builder(name, **self._model_options.get(name, {}))
        self._default_long_side = getattr(getattr(backbone, "cfg", None), "long_side", None)
        self._backbone = backbone.to(self.device).eval()
        self._name = name
        self._apply_long_side()

    def configure(
        self, name: str, *, checkpoint_path: str = "", allow_download: bool = False
    ) -> None:
        options = {"checkpoint_path": checkpoint_path or None, "allow_download": allow_download}
        if options != self._model_options.get(name):
            if self._name == name:
                self._free()
            self._model_options[name] = options

    def model_key(self, name: str) -> tuple:
        return (name, tuple(sorted(self._model_options.get(name, {}).items())))

    def set_long_side(self, long_side: int | None) -> None:
        """Override the input resolution of the active (and any future) backbone;
        ``None`` restores the per-backbone default. Takes effect on the next ``run``."""
        self.long_side = long_side
        self._apply_long_side()

    def _apply_long_side(self) -> None:
        cfg = getattr(self._backbone, "cfg", None)
        if cfg is None or not hasattr(cfg, "long_side"):
            return
        cfg.long_side = self.long_side if self.long_side is not None else self._default_long_side

    def _free(self) -> None:
        if self._backbone is None:
            return
        self._backbone = None
        self._name = None
        _free_cuda()

    def release(self) -> None:
        """Release inference models before another stage needs GPU memory."""
        self._free()
        self._free_metric()

    def _free_metric(self) -> None:
        if self._metric_model is None:
            return
        self._metric_model = None
        self._metric_name = None
        _free_cuda()

    def _load_metric(self, name: str) -> None:
        if name == self._metric_name and self._metric_model is not None:
            return
        self._free_metric()
        self._metric_model = self._metric_builder(name).to(self.device).eval()
        self._metric_name = name

    @torch.inference_mode()
    def compute_metric_scale(
        self,
        result: BackboneResult,
        images: Tensor,
        metric_name: str,
        conf_thresh: float = float("-inf"),
    ) -> float:
        """Global scale of ``result.depth`` to a metric monocular model's depth
        (median of ratios over pixels with ``conf > conf_thresh``).

        ``images`` are the run cameras' ``(V, 3, H, W)`` frames. The metric model gets
        the GT intrinsics (predicted ones as fallback).
        """
        self._load_metric(metric_name)
        intr = result.gt_intrinsics if result.gt_intrinsics is not None else result.pred_intrinsics
        intr_in = None if intr is None else intr.unsqueeze(0).to(self.device)
        out = self._metric_model(images.unsqueeze(0).to(self.device), intr_in)
        reference = out.depth[0]  # (V, Hm, Wm)
        hd, wd = result.depth.shape[-2:]
        reference = F.interpolate(
            reference.unsqueeze(1), size=(hd, wd), mode="bilinear", align_corners=False
        )[:, 0]
        reference = reference.detach().float().cpu()
        mask = result.conf > conf_thresh if result.conf is not None else None
        return float(fit_depth_scale(result.depth, reference, mask=mask))

    @torch.inference_mode()
    def run(
        self,
        frame: Frame,
        cam_ixs: list[int],
        condition_on_gt_cameras: bool = False,
        depth: Tensor | None = None,
    ) -> BackboneResult:
        """Run the active backbone on cameras ``cam_ixs`` of ``frame``.

        GT cameras are fed only when ``condition_on_gt_cameras`` and the backbone
        accepts them. ``depth`` (``(V, 1, H, W)`` metres, default ``frame.depth``) is
        forwarded only to backbones that consume GT depth; those raise ``ValueError``
        when it is missing.
        """
        if self._backbone is None:
            raise RuntimeError("no backbone loaded; call load(name) first")
        if depth is None:
            depth = getattr(frame, "depth", None)
        if backbone_needs_gt_depth(self._name) and depth is None:
            raise ValueError(
                f"backbone {self._name!r} needs GT depth, but this dataset/frame provides none; "
                "pick a dataset that ships depth, or a depth-predicting backbone"
            )

        ix = torch.as_tensor(cam_ixs, dtype=torch.long)
        gt_extr = frame.extrinsics[ix]
        gt_intr = frame.intrinsics[ix]
        images = frame.images[ix].unsqueeze(0).to(self.device)

        conditioned = bool(condition_on_gt_cameras) and self.accepts_gt_cameras
        extr_in = gt_extr.unsqueeze(0).to(self.device) if conditioned else None
        intr_in = gt_intr.unsqueeze(0).to(self.device) if conditioned else None

        depth_in = None
        if depth is not None and backbone_needs_gt_depth(self._name):
            d = depth if depth.dim() == 4 else depth.unsqueeze(1)
            depth_in = d[ix].unsqueeze(0).to(self.device)
        out = self._backbone(images, extr_in, intr_in, depth=depth_in)
        data = out.data

        def _cpu(t):
            return None if t is None else t.detach().float().cpu()

        pred_extr = data.get("extrinsics_pred")
        pred_intr = data.get("intrinsics_pred")
        return BackboneResult(
            depth=_cpu(data["depth"][0]),
            conf=_cpu(data["depth_conf"][0]) if "depth_conf" in data else None,
            pred_extrinsics=_cpu(pred_extr[0]) if pred_extr is not None else None,
            pred_intrinsics=_cpu(pred_intr[0]) if pred_intr is not None else None,
            gt_extrinsics=_cpu(gt_extr),
            gt_intrinsics=_cpu(gt_intr),
            conditioned=conditioned,
        )
