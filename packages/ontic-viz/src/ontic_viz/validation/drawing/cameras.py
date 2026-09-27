"""Camera frustums projected onto the three axis-aligned planes."""

from __future__ import annotations

from typing import Optional

import torch
from einops import einsum, rearrange, repeat
from torch import Tensor

from ontic_lib.camera import unproject_camera_points
from ontic_viz.validation.annotation import add_label
from ontic_viz.validation.drawing.lines import draw_lines
from ontic_viz.validation.drawing.types import Scalar, sanitize_scalar


def draw_cameras(
    resolution: int,
    extrinsics: Tensor,
    intrinsics: Tensor,
    color: Tensor,
    near: Optional[Scalar] = None,
    far: Optional[Scalar] = None,
    margin: float = 0.1,  # relative to AABB
    frustum_scale: float = 0.05,  # relative to image resolution
) -> Tensor:
    """Draw ``(batch, 4, 4)`` camera-to-world poses with ``(batch, 3, 3)`` normalized
    intrinsics and per-camera ``(batch, 3)`` colours; returns the three labelled
    ``(3, resolution, resolution)`` projections stacked as ``(3, 3, height, width)``.
    Near / far planes are drawn in grey when given."""
    device = extrinsics.device

    # Compute scene bounds.
    minima, maxima = compute_aabb(extrinsics, intrinsics, near, far)
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(minima, maxima, margin=margin)
    span = (scene_maxima - scene_minima).max()

    # Compute frustum locations.
    corner_depth = (span * frustum_scale)[None]
    frustum_corners = unproject_frustum_corners(extrinsics, intrinsics, corner_depth)
    if near is not None:
        near_corners = unproject_frustum_corners(extrinsics, intrinsics, near)
    if far is not None:
        far_corners = unproject_frustum_corners(extrinsics, intrinsics, far)

    # Project the cameras onto each axis-aligned plane.
    projections = []
    for projected_axis in range(3):
        image = torch.zeros(
            (3, resolution, resolution),
            dtype=torch.float32,
            device=device,
        )
        image_x_axis = (projected_axis + 1) % 3
        image_y_axis = (projected_axis + 2) % 3

        def project(points: Tensor) -> Tensor:
            x = points[..., image_x_axis]
            y = points[..., image_y_axis]
            return torch.stack([x, y], dim=-1)

        x_range, y_range = torch.stack((project(scene_minima), project(scene_maxima)), dim=-1)

        # Draw near and far planes.
        if near is not None:
            projected_near_corners = project(near_corners)
            image = draw_lines(
                image,
                rearrange(projected_near_corners, "b p xy -> (b p) xy"),
                rearrange(projected_near_corners.roll(1, 1), "b p xy -> (b p) xy"),
                color=0.25,
                width=2,
                x_range=x_range,
                y_range=y_range,
            )
        if far is not None:
            projected_far_corners = project(far_corners)
            image = draw_lines(
                image,
                rearrange(projected_far_corners, "b p xy -> (b p) xy"),
                rearrange(projected_far_corners.roll(1, 1), "b p xy -> (b p) xy"),
                color=0.25,
                width=2,
                x_range=x_range,
                y_range=y_range,
            )
        if near is not None and far is not None:
            image = draw_lines(
                image,
                rearrange(projected_near_corners, "b p xy -> (b p) xy"),
                rearrange(projected_far_corners, "b p xy -> (b p) xy"),
                color=0.25,
                width=2,
                x_range=x_range,
                y_range=y_range,
            )

        # Draw the camera frustums themselves.
        projected_origins = project(extrinsics[:, :3, 3])
        projected_frustum_corners = project(frustum_corners)
        start = [
            repeat(projected_origins, "b xy -> (b p) xy", p=4),
            rearrange(projected_frustum_corners.roll(1, 1), "b p xy -> (b p) xy"),
        ]
        start = rearrange(torch.cat(start, dim=0), "(r b p) xy -> (b r p) xy", r=2, p=4)
        image = draw_lines(
            image,
            start,
            repeat(projected_frustum_corners, "b p xy -> (b r p) xy", r=2),
            color=repeat(color, "b c -> (b r p) c", r=2, p=4),
            width=2,
            x_range=x_range,
            y_range=y_range,
        )

        x_name = "XYZ"[image_x_axis]
        y_name = "XYZ"[image_y_axis]
        image = add_label(image, f"{x_name}{y_name} Projection")

        # TODO: Draw axis indicators.
        projections.append(image)

    return torch.stack(projections)


def compute_aabb(
    extrinsics: Tensor,
    intrinsics: Tensor,
    near: Optional[Scalar] = None,
    far: Optional[Scalar] = None,
) -> tuple[Tensor, Tensor]:
    """Compute an axis-aligned bounding box (``(3,)`` minima, ``(3,)`` maxima) for the camera
    origins and, when given, their near / far frustum corners."""

    device = extrinsics.device

    # These points are included in the AABB.
    points = [extrinsics[:, :3, 3]]

    if near is not None:
        near = sanitize_scalar(near, device)
        corners = unproject_frustum_corners(extrinsics, intrinsics, near)
        points.append(rearrange(corners, "b p xyz -> (b p) xyz"))

    if far is not None:
        far = sanitize_scalar(far, device)
        corners = unproject_frustum_corners(extrinsics, intrinsics, far)
        points.append(rearrange(corners, "b p xyz -> (b p) xyz"))

    points = torch.cat(points, dim=0)
    return points.min(dim=0).values, points.max(dim=0).values


def compute_equal_aabb_with_margin(
    minima: Tensor,
    maxima: Tensor,
    margin: float = 0.1,
) -> tuple[Tensor, Tensor]:
    """Grow ``(*batch, 3)`` bounds to a cube around their midpoint, ``margin`` wider than the
    largest extent (the max is taken over the whole tensor, so a batch shares one span)."""
    midpoint = (maxima + minima) * 0.5
    span = (maxima - minima).max() * (1 + margin)
    scene_minima = midpoint - 0.5 * span
    scene_maxima = midpoint + 0.5 * span
    return scene_minima, scene_maxima


def unproject_frustum_corners(
    extrinsics: Tensor,
    intrinsics: Tensor,
    depth: Tensor,
) -> Tensor:
    """World-space frustum corners ``(batch, 4, 3)`` at camera-z ``depth`` ``(#batch,)``,
    ordered so that following them in a circle draws the rectangle."""
    device = extrinsics.device

    # Get coordinates for the corners. Following them in a circle makes a rectangle.
    xy = torch.linspace(0, 1, 2, device=device)
    xy = torch.stack(torch.meshgrid(xy, xy, indexing="xy"), dim=-1)
    xy = rearrange(xy, "i j xy -> (i j) xy")
    xy = xy[torch.tensor([0, 1, 3, 2], device=device)]

    # Get ray directions in camera space.
    directions = unproject_camera_points(
        xy,
        torch.ones(1, dtype=torch.float32, device=device),
        rearrange(intrinsics, "b i j -> b () i j"),
    )

    # Divide by the z coordinate so that multiplying by depth will produce orthographic
    # depth (z depth) as opposed to Euclidean depth (distance from the camera).
    directions = directions / directions[..., -1:]
    directions = einsum(extrinsics[..., :3, :3], directions, "b i j, b r j -> b r i")

    origins = rearrange(extrinsics[:, :3, 3], "b xyz -> b () xyz")
    depth = rearrange(depth, "b -> b () ()")
    return origins + depth * directions
