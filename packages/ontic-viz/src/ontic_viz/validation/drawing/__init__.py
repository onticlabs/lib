"""Anti-aliased 2-D drawing on ``(3, height, width)`` images: lines, points, camera frustums."""

from ontic_viz.validation.drawing.cameras import compute_equal_aabb_with_margin, draw_cameras
from ontic_viz.validation.drawing.lines import draw_lines
from ontic_viz.validation.drawing.points import draw_points

__all__ = ["compute_equal_aabb_with_margin", "draw_cameras", "draw_lines", "draw_points"]
