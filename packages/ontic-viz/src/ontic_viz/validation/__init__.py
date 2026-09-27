"""Validation visualisations for world-model training runs.

Image layout (``hcat`` / ``vcat`` / ``add_border`` / ``add_label``), turbo depth colouring,
anti-aliased camera and projection drawings, camera trajectories, the reconstruction /
projection / camera / fly-through report functions and Gaussian statistics, all written
through a pluggable :class:`VizLogger` (``NoOpVizLogger``, ``LocalVizLogger`` for files,
``OlympusVizLogger`` for the tracker via ``ontic_lib.tracking``).

Rendering itself is a callback (see :mod:`ontic_viz.validation.report`); nothing here
depends on a particular scene container. Optional extras, imported lazily where used:
``ontic-viz[images]`` (pillow: labels, PNGs), ``[video]`` (moviepy: local MP4s) and
``[ply]`` (plyfile: local point clouds).
"""

from ontic_viz.validation.annotation import add_label
from ontic_viz.validation.color_map import depth_map
from ontic_viz.validation.drawing.cameras import draw_cameras
from ontic_viz.validation.image_io import fig_to_image, prep_image, save_image
from ontic_viz.validation.layout import add_border, hcat, vcat
from ontic_viz.validation.logger import (
    LocalVizLogger,
    NoOpVizLogger,
    OlympusVizLogger,
    VizLogger,
)
from ontic_viz.validation.projections import render_cameras, render_projections
from ontic_viz.validation.report import (
    TrajectoryFn,
    extras_rgb,
    gaussian_statistics_figure,
    get_optionals,
    latent_statistics,
    log_traj,
    reconstruction,
    render_video_generic,
    render_video_interpolation,
    render_video_wobble,
    sh_dc_to_rgb,
    visualize_cameras,
    visualize_projections,
)
from ontic_viz.validation.trajectory import (
    generate_wobble,
    interpolate_extrinsics,
    interpolate_intrinsics,
)

__all__ = [
    "VizLogger",
    "NoOpVizLogger",
    "LocalVizLogger",
    "OlympusVizLogger",
    "TrajectoryFn",
    "hcat",
    "vcat",
    "add_border",
    "add_label",
    "depth_map",
    "fig_to_image",
    "prep_image",
    "save_image",
    "interpolate_extrinsics",
    "interpolate_intrinsics",
    "generate_wobble",
    "draw_cameras",
    "render_projections",
    "render_cameras",
    "reconstruction",
    "log_traj",
    "visualize_projections",
    "visualize_cameras",
    "render_video_wobble",
    "render_video_interpolation",
    "render_video_generic",
    "get_optionals",
    "extras_rgb",
    "sh_dc_to_rgb",
    "latent_statistics",
    "gaussian_statistics_figure",
]
