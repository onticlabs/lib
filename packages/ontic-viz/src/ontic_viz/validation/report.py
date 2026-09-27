"""The validation report: reconstruction strips, trajectory plots, scene projections, camera
plots, fly-through videos and Gaussian statistics, written through a :class:`VizLogger`.

Rendering is delegated to a ``decoder`` callback::

    decoder(spatial_latent: dict, extrinsics, intrinsics, near, far, image_shape,
            with_extras: bool = False)

with ``(batch, view, ...)`` cameras, returning any object with ``.color``
``(batch, view, 3, H, W)``, ``.depth`` ``(batch, view, 1, H, W)``, ``.alpha`` and ``.extras``
(a ``{name: (batch, view, C, H, W)}`` dict, or ``None``). The callback renders on its own
fixed background. Batches follow ``ontic_data.example.BatchedTempExample``.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange, repeat
from matplotlib.figure import Figure
from torch import Tensor

from ontic_data.example import BatchedTempExample, BatchedTempViews
from ontic_viz.validation.annotation import add_label
from ontic_viz.validation.color_map import depth_map
from ontic_viz.validation.image_io import fig_to_image
from ontic_viz.validation.layout import add_border, hcat, vcat
from ontic_viz.validation.logger import VizLogger
from ontic_viz.validation.projections import render_cameras, render_projections
from ontic_viz.validation.trajectory import (
    generate_wobble,
    interpolate_extrinsics,
    interpolate_intrinsics,
)

matplotlib.use("Agg")


@runtime_checkable
class TrajectoryFn(Protocol):
    """``t (time,)`` -> (extrinsics ``(batch, time, 4, 4)``, intrinsics ``(batch, time, 3, 3)``)."""

    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        pass


# ---------------------------------------------------------------------------
# Image visualization functions
# ---------------------------------------------------------------------------


def reconstruction(
    logger: VizLogger,
    step: int,
    key: str,
    scene: str,
    ctx_state_mask: dict,
    label: str = "",
    left_to_right: bool = True,
):
    """Log the per-frame comparison of named image sequences.

    ``ctx_state_mask`` maps a column label to ``(time, view, 3, H, W)`` images; frame ``i``
    shows every column's ``i``-th time step (columns shorter than the longest wrap around),
    each column's views stacked, all under a ``Frame i/n`` label. One frame is logged as an
    image, several as a 14 fps video captioned with ``scene``.
    """
    view_cat, elem_cat = (vcat, hcat) if left_to_right else (hcat, vcat)
    if ctx_state_mask is None:
        return

    n_max_frames = max([len(v) for v in ctx_state_mask.values()])
    n_frames = {k: len(v) for k, v in ctx_state_mask.items()}
    comparison = [
        add_label(
            add_border(
                elem_cat(
                    *[
                        add_label(view_cat(*v[i % n_frames[k]]), k)
                        for k, v in ctx_state_mask.items()
                    ],
                )
            ),
            f"Frame {i + 1}/{n_max_frames}",
        )
        for i in range(n_max_frames)
    ]
    if label:
        comparison = [add_label(img, label) for img in comparison]

    if len(comparison) > 1:
        logger.log_video(key, comparison, step, fps=14, caption=scene, loop_reverse=False)
    else:
        logger.log_image(key, comparison[0], step, caption=scene)


@torch.no_grad()
def log_traj(
    logger: VizLogger,
    step: int,
    key: str,
    traj: Tensor,
    n_pred: int,
    ix_vertical: int | None = None,
):
    """Scatter a per-time-step metric ``traj (time,)`` against time (with an optional dashed
    vertical at ``ix_vertical``) and log it as an image; skipped when ``n_pred == 0``."""
    if n_pred == 0:
        return
    plt.figure()
    t = traj.detach().float().cpu().numpy()
    x = np.arange(t.shape[0])
    plt.scatter(x[: x.shape[0]], t, color="b", alpha=0.8)
    if ix_vertical is not None:
        plt.axvline(ix_vertical, color="r", linestyle="--")
    plt.xlabel("Time")
    plt.ylabel(key)

    img = fig_to_image(plt.gcf())
    plt.close()
    logger.log_image(key, img, step, caption=f"Trajectory {key} at step {step}")


# ---------------------------------------------------------------------------
# Particle / 3D visualization functions
# ---------------------------------------------------------------------------


@torch.no_grad()
def latent_statistics(latent: dict) -> dict[str, np.ndarray]:
    """The distributions worth histogramming from a latent dict: ``opacities`` flattened and
    the norms of ``scales`` flattened, under those keys; empty when neither is present."""
    opacities = latent.get("opacities", None)
    scales = latent.get("scales", None)
    data_dict = {}
    if opacities is not None:
        data_dict["opacities"] = opacities.detach().cpu().numpy().reshape(-1)
    if scales is not None:
        data_dict["scales"] = scales.norm(dim=-1).detach().cpu().numpy().reshape(-1)
    return data_dict


def gaussian_statistics_figure(
    opacities: Tensor,
    scales: Tensor,
    mask: Tensor | None = None,
) -> tuple[Figure, dict[str, float]]:
    """Opacity and scale distribution plots plus three scalar summaries.

    A 2x3 grid: opacity histogram, opacity CDF, opacity-vs-scale scatter, per-axis scale
    histogram, scale norm histogram, scale anisotropy histogram. ``opacities`` is any shape
    with one value per Gaussian, ``scales`` ``(..., 3)``, ``mask`` an optional boolean
    selection of the active Gaussians. Returns the open figure (the caller closes it) and
    ``{"gs/frac_opacity_lt_0.1", "gs/frac_opacity_lt_0.3", "gs/scale_anisotropy_median"}``.
    The scatter subsamples 3000 points through ``np.random``; seed it for repeatable output.
    """
    matplotlib.use("Agg")

    opacities = opacities.detach().cpu().reshape(-1).float().numpy()
    scales = scales.detach().cpu().reshape(-1, 3).float()
    if mask is not None:
        m = mask.detach().cpu().reshape(-1).bool().numpy()
        opacities = opacities[m]
        scales = scales[m]
    scales_np = scales.numpy()
    scale_norms = np.linalg.norm(scales_np, axis=-1)
    N = len(opacities)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"Gaussian Stats — {N} active", fontsize=13)

    # Row 1: Opacity
    ax = axes[0, 0]
    ax.hist(opacities, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(
        np.mean(opacities), color="red", ls="--", lw=1, label=f"mean={np.mean(opacities):.3f}"
    )
    ax.axvline(
        np.median(opacities), color="orange", ls="--", lw=1, label=f"med={np.median(opacities):.3f}"
    )
    ax.set_xlabel("Opacity")
    ax.set_title("Opacity Distribution")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    sorted_op = np.sort(opacities)
    ax.plot(sorted_op, np.arange(1, len(sorted_op) + 1) / len(sorted_op), color="steelblue", lw=1.5)
    ax.set_xlabel("Opacity")
    ax.set_ylabel("CDF")
    ax.set_title("Opacity CDF")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    idx = np.random.choice(N, min(3000, N), replace=False)
    ax.scatter(opacities[idx], scale_norms[idx], s=1, alpha=0.3, c="steelblue")
    ax.set_xlabel("Opacity")
    ax.set_ylabel("Scale norm")
    ax.set_title("Opacity vs Scale")
    ax.set_yscale("log")

    # Row 2: Scales
    ax = axes[1, 0]
    for i, (lbl, clr) in enumerate(zip("XYZ", ["#e74c3c", "#2ecc71", "#3498db"])):
        ax.hist(scales_np[:, i], bins=80, alpha=0.6, label=lbl, color=clr, edgecolor="none")
    ax.set_xlabel("Scale")
    ax.set_title("Scale per axis")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(scale_norms, bins=80, color="purple", edgecolor="none", alpha=0.8)
    ax.axvline(
        np.mean(scale_norms), color="red", ls="--", lw=1, label=f"mean={np.mean(scale_norms):.5f}"
    )
    ax.set_xlabel("Scale norm")
    ax.set_title("Scale Norm")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    ratio = scales.max(dim=-1).values / scales.min(dim=-1).values.clamp(min=1e-8)
    ratio_np = ratio.clamp(max=50).numpy()
    ax.hist(ratio_np, bins=80, color="darkorange", edgecolor="none", alpha=0.8)
    ax.axvline(
        np.median(ratio_np), color="red", ls="--", lw=1, label=f"med={np.median(ratio_np):.1f}"
    )
    ax.set_xlabel("max/min ratio")
    ax.set_title("Scale Anisotropy")
    ax.legend(fontsize=8)

    plt.tight_layout()

    stats = {
        "gs/frac_opacity_lt_0.1": float((opacities < 0.1).mean()),
        "gs/frac_opacity_lt_0.3": float((opacities < 0.3).mean()),
        "gs/scale_anisotropy_median": float(np.median(ratio_np)),
    }
    return fig, stats


def visualize_projections(
    logger: VizLogger,
    step: int,
    key: str,
    past_future_latent_unbatched: dict,
    decoder,
):
    """Log the three axis-aligned projections of a ``(time, points, ...)`` latent, side by
    side per time step: an image for one step, a 4 fps video otherwise."""
    temp_projections = render_projections(
        past_future_latent_unbatched,
        256,
        decoder=decoder,
        extra_label="",
    )
    projections = [add_border(hcat(*img)) for img in temp_projections]
    if len(projections) > 1:
        logger.log_video(key, projections, step, fps=4, loop_reverse=False)
    else:
        logger.log_image(key, projections[0], step)


def visualize_cameras(
    logger: VizLogger,
    step: int,
    key: str,
    squeezed_batch: BatchedTempExample,
):
    """Log the batch's context / target camera frustums on the three axis planes."""
    cameras = hcat(*render_cameras(squeezed_batch, 256))
    logger.log_image(key, add_border(cameras), step)


def render_video_wobble(
    logger: VizLogger,
    step: int,
    device: torch.device,
    key: str,
    spatial_latent: dict,
    batch: BatchedTempExample,
    decoder,
    n_state: int,
    n_pred: int,
):
    """A 60-frame video wobbling the first context camera by a quarter of the distance to
    the second one; see :func:`render_video_generic`."""

    def trajectory_fn(t):
        origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
        origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
        delta = (origin_a - origin_b).norm(dim=-1)
        extrinsics = generate_wobble(
            batch["context"]["extrinsics"][:, 0],
            delta * 0.25,
            t,
        )
        intrinsics = repeat(
            batch["context"]["intrinsics"][:, 0],
            "b i j -> b v i j",
            v=t.shape[0],
        )
        return extrinsics, intrinsics

    return render_video_generic(
        logger,
        step,
        device,
        spatial_latent,
        batch,
        trajectory_fn,
        key,
        decoder,
        n_state,
        n_pred,
        num_frames=60,
    )


def render_video_interpolation(
    logger: VizLogger,
    step: int,
    device: torch.device,
    key: str,
    spatial_latent: dict,
    batch: BatchedTempExample,
    decoder,
    n_state: int,
    n_pred: int,
    ix1: int = 0,
    ix2: int = 1,
):
    """A 30-frame video interpolating from context camera ``ix1`` to ``ix2`` (to the first
    target camera when there is only one context view); see :func:`render_video_generic`."""
    _, v, _, _ = batch["context"]["extrinsics"].shape

    def trajectory_fn(t):
        extrinsics = interpolate_extrinsics(
            batch["context"]["extrinsics"][0, ix1],
            (
                batch["context"]["extrinsics"][0, ix2]
                if v > 1
                else batch["target"]["extrinsics"][0, 0]
            ),
            t,
        )
        intrinsics = interpolate_intrinsics(
            batch["context"]["intrinsics"][0, ix1],
            (
                batch["context"]["intrinsics"][0, ix2]
                if v > 1
                else batch["target"]["intrinsics"][0, 0]
            ),
            t,
        )
        return extrinsics[None], intrinsics[None]

    return render_video_generic(
        logger,
        step,
        device,
        spatial_latent,
        batch,
        trajectory_fn,
        key,
        decoder,
        n_state,
        n_pred,
    )


@torch.no_grad()
def render_video_generic(
    logger: VizLogger,
    step: int,
    device: torch.device,
    spatial_latent: dict,  # first dim is time
    batch: BatchedTempExample,
    trajectory_fn: TrajectoryFn,
    name: str,
    decoder,
    n_state: int,
    n_pred: int,
    num_frames: int = 30,
    smooth: bool = True,
    loop_reverse: bool = False,
    with_extras: bool = True,
) -> None:
    """Render a fly-through along ``trajectory_fn`` and log it as a video.

    Time runs with the camera: frame ``t`` in [0, 1] (cosine-eased when ``smooth``) is
    rendered from the latent time step nearest to ``t``. Each frame stacks the RGB render,
    the turbo depth map and, with ``with_extras``, every 1- or 3-channel extra as RGB (see
    :func:`extras_rgb`). Frames are rendered in chunks of 15 at the context image size.
    """
    downscale_factor = 1
    _, _, _, h, w = batch["context"]["image"].shape
    h, w = h // downscale_factor, w // downscale_factor
    if num_frames > 100:
        print(f" Reducing validation video frames from {num_frames} to 60 to save memory")
        num_frames = 60
    CHUNK_SIZE = 15
    t_all = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=device)
    if smooth:
        t_all = (torch.cos(torch.pi * (t_all + 1)) + 1) / 2
    t_g = torch.linspace(0, 1, spatial_latent["means"].shape[0], dtype=torch.float32, device=device)
    all_images = []
    for chunk_start in range(0, num_frames, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, num_frames)
        t = t_all[chunk_start:chunk_end]
        chunk_size = len(t)
        extrinsics, intrinsics = trajectory_fn(t)
        extrinsics = rearrange(extrinsics[:1], "b v i j -> v b i j")
        intrinsics = rearrange(intrinsics[:1], "b v i j -> v b i j")
        corresponding_ix = torch.abs(t.unsqueeze(1) - t_g).argmin(dim=1)
        chunk_spatial_latent = {k: v[corresponding_ix] for k, v in spatial_latent.items()}
        near = repeat(batch["context"]["near"][0, 0], " -> b 1", b=chunk_size)
        far = repeat(batch["context"]["far"][0, 0], " -> b 1", b=chunk_size)
        output_prob = decoder(
            chunk_spatial_latent, extrinsics, intrinsics, near, far, (h, w), with_extras=with_extras
        )
        rgbs, depths = output_prob.color[:, 0], depth_map(output_prob.depth[:, 0].squeeze(-3))
        if output_prob.extras and with_extras:
            extras = extras_rgb(output_prob.extras)
            other_imgs, keys = zip(*[(v[:, 0], k) for k, v in extras.items()])
        else:
            other_imgs, keys = [], []
        n_fut = len(t_g) - n_state
        desc = f"[Dyn-Model: {n_state}->{n_pred}, N={n_fut}]" if n_pred > 0 else ""
        chunk_images = []
        for rgb, depth, *ims in zip(rgbs, depths, *other_imgs):
            vis_components = [add_label(rgb, f"RGB {desc}"), add_label(depth, "Depth")]
            if with_extras:
                vis_components.extend([add_label(im, k) for im, k in zip(ims, keys)])
            image = vcat(*vis_components)
            chunk_images.append(add_border(image))
        all_images.extend(chunk_images)
        del output_prob, chunk_spatial_latent, extrinsics, intrinsics
    logger.log_video(name, all_images, step, loop_reverse=loop_reverse)


@torch.no_grad()
def extras_rgb(extras: dict | None) -> dict[str, Tensor]:
    """The 1- and 3-channel entries of a decoder's ``extras`` as RGB ``(..., 3, H, W)``:
    3-channel ones pass through, 1-channel ones go to the red channel, others are skipped."""
    ex_colors = {}
    for k, v in (extras or {}).items():
        if v.shape[-3] == 3:
            ex_colors[k] = v
        elif v.shape[-3] == 1:
            ex_colors[k] = torch.cat([v, torch.zeros_like(v), torch.zeros_like(v)], dim=-3)
    return ex_colors


@torch.no_grad()
def sh_dc_to_rgb(harmonics: Tensor) -> np.ndarray:
    """Convert 0th-order SH (DC) coefficients ``(..., 3, d_sh)`` to RGB in [0, 255].

    Uses the same conversion as gsplat's spherical_harmonics:
        rgb = clamp(C0 * dc + 0.5, 0, 1) * 255
    """
    C0 = 0.28209479177387814
    dc = harmonics[..., 0]  # (..., 3)
    rgb = (C0 * dc + 0.5).clamp(0.0, 1.0)
    return rgb.cpu().numpy() * 255.0


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def get_optionals(key: Literal["Context", "Target"], views: BatchedTempViews, spatial_decoded):
    """Optional visualizations for context/target views, keyed by their column label.

    1. Depth map if available.
    2. Confidence map if available.
    3. State mask if available.
    4. Alpha map if available.

    ``spatial_decoded`` is any decoder output with ``.depth`` and ``.extras``. The batch
    dimension is removed by only taking the first element.
    """
    out_dict = {}
    if spatial_decoded.depth is not None:
        out_dict[f"Pred {key} depth"] = depth_map(spatial_decoded.depth[0].squeeze(-3))

    if spatial_decoded.extras is not None and "confidence_map" in spatial_decoded.extras:
        conf_map = 1 + torch.exp(spatial_decoded.extras["confidence_map"][0])
        out_dict[f"Pred {key} confidence_map"] = depth_map(conf_map.squeeze(-3))

    if "state_mask" in views:
        mask = views["state_mask"][0].float()
        out_dict[f"GT {key} state_mask"] = torch.cat(
            [mask, torch.zeros_like(mask), 1 - mask], dim=-3
        )
        pred_mask = spatial_decoded.extras.get("state_mask", None)
        if pred_mask is not None:
            out_dict[f"pred {key} state_mask"] = torch.cat(
                [pred_mask[0].float(), torch.zeros_like(mask), 1 - pred_mask[0].float()], dim=-3
            )
    if "static_float" in views:
        mask = views["static_float"][0].float()
        out_dict[f"GT {key} static_mask"] = torch.cat(
            [mask, torch.zeros_like(mask), 1 - mask], dim=-3
        )
    stat_mask = (
        spatial_decoded.extras.get("static_float", None)
        if spatial_decoded.extras is not None
        else None
    )
    if stat_mask is not None:
        stat_mask = stat_mask[0].float()
        out_dict[f"pred {key} static_mask"] = torch.cat(
            [1 - stat_mask, stat_mask, 1 - stat_mask], dim=-3
        )

    if spatial_decoded.extras is not None and "workspace_mask" in spatial_decoded.extras:
        ws_mask = spatial_decoded.extras["workspace_mask"][0]
        out_dict[f"Pred {key} workspace_mask"] = torch.cat(
            [ws_mask, torch.zeros_like(ws_mask), 1 - ws_mask], dim=-3
        )
    return out_dict
