"""ontic_viz: interactive inspection tools built on ontic-nn and ontic-data.

``ontic_viz.backbone_viewer`` is a viser web app that runs any registered
backbone or metric-depth model on any registered dataset and shows the
aligned point clouds, cameras, and depth overlays. ``ontic_viz.validation``
holds the training-time validation visualisations (image layout, camera and
projection drawings, fly-through videos, Gaussian statistics) behind a
pluggable ``VizLogger``. Import subpackages explicitly; this module imports
nothing.
"""

__version__ = "0.5.0"
