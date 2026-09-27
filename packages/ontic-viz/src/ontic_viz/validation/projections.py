"""Axis-aligned views of a latent scene, and the camera-frustum plot of a batch."""

from __future__ import annotations

import torch
from einops import repeat
from torch import Tensor

from ontic_viz.validation.annotation import add_label
from ontic_viz.validation.drawing.cameras import compute_equal_aabb_with_margin, draw_cameras


def pad(images: list[Tensor]) -> list[Tensor]:
    """Pad every image with ones (top-left aligned) to the elementwise-max shape of the list."""
    shapes = torch.stack([torch.tensor(x.shape) for x in images])
    padded_shape = shapes.max(dim=0)[0]
    results = [torch.ones(padded_shape.tolist(), dtype=x.dtype, device=x.device) for x in images]
    for image, result in zip(images, results):
        slices = tuple(slice(0, x) for x in image.shape)
        result[slices] = image[slices]
    return results


def render_projections(
    spatial_latent: dict,
    resolution: int,
    decoder,
    margin: float = 0.1,
    draw_label: bool = True,
    extra_label: str = "",
) -> Tensor:
    """Render the scene from the three axis directions as ``(batch, 3, 3, height, width)``.

    ``spatial_latent["means"]`` is ``(batch, points, 3)``; a near-orthographic camera (10 deg
    field of view, moved back) frames the scene's cube-shaped AABB from +X, +Y and +Z.
    ``decoder(spatial_latent, extrinsics, intrinsics, near, far, image_shape)`` renders
    ``(batch, view, ...)`` cameras and returns an object with a ``.color`` tensor
    ``(batch, view, 3, height, width)``; it uses its own fixed background.
    """
    means = spatial_latent["means"]
    device = means.device
    b = means.shape[0]

    # Compute the minima and maxima of the scene.
    minima = means.min(dim=1).values
    maxima = means.max(dim=1).values
    scene_minima, scene_maxima = compute_equal_aabb_with_margin(minima, maxima, margin=margin)
    scene_minima[1:] = scene_minima[0]
    scene_maxima[1:] = scene_maxima[0]

    projections = []
    extrinsics_list = []
    intrinsics_list = []
    nears = []
    fars = []
    for look_axis in range(3):
        right_axis = (look_axis + 1) % 3
        down_axis = (look_axis + 2) % 3

        # Define the intrinsics for rendering.
        extents = scene_maxima - scene_minima
        far = extents[:, look_axis]
        near = torch.zeros_like(far)
        width = extents[:, right_axis]
        height = extents[:, down_axis]

        # Create fake "orthographic" projection by moving the camera back and picking a
        # small field of view.
        fov_x = torch.tensor(10.0, device=means.device).deg2rad()
        tan_fov_x = (0.5 * fov_x).tan()
        distance_to_near = (0.5 * width) / tan_fov_x
        tan_fov_y = 0.5 * height / distance_to_near
        near = near + distance_to_near
        far = far + distance_to_near

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32, device=device)
        intrinsics = repeat(intrinsics, "i j -> b i j", b=b).clone()
        intrinsics[:, 0, 0] = 1.0 / tan_fov_x
        intrinsics[:, 1, 1] = 1.0 / tan_fov_y
        intrinsics[:, 0, 2] = 0.5
        intrinsics[:, 1, 2] = 0.5

        # Define the extrinsics for rendering.
        extrinsics = torch.zeros((b, 4, 4), dtype=torch.float32, device=device)
        extrinsics[:, right_axis, 0] = 1
        extrinsics[:, down_axis, 1] = -1
        extrinsics[:, look_axis, 2] = -1
        extrinsics[:, right_axis, 3] = 0.5 * (
            scene_minima[:, right_axis] + scene_maxima[:, right_axis]
        )
        extrinsics[:, down_axis, 3] = 0.5 * (
            scene_minima[:, down_axis] + scene_maxima[:, down_axis]
        )
        extrinsics[:, look_axis, 3] = -scene_minima[:, look_axis]
        extrinsics[:, 3, 3] = 1
        move_back = torch.eye(4, dtype=torch.float32, device=extrinsics.device).repeat(b, 1, 1)
        move_back[:, 2, 3] = -distance_to_near
        extrinsics = extrinsics @ move_back

        extrinsics_list.append(extrinsics)
        intrinsics_list.append(intrinsics)
        nears.append(near)
        fars.append(far)

    extrinsics_list = torch.stack(extrinsics_list, dim=1)
    intrinsics_list = torch.stack(intrinsics_list, dim=1)
    near = torch.stack(nears, dim=1)
    far = torch.stack(fars, dim=1)

    projections = decoder(
        spatial_latent,
        extrinsics_list,
        intrinsics_list,
        near,
        far,
        (resolution, resolution),
    ).color

    if draw_label:
        proj_l = list(projections.unbind(dim=1))
        for look_axis in range(3):
            right_axis = (look_axis + 1) % 3
            down_axis = (look_axis + 2) % 3
            right_axis_name = "XYZ"[right_axis]
            down_axis_name = "XYZ"[down_axis]
            label = f"{right_axis_name}{down_axis_name} Projection {extra_label}"

            proj_l[look_axis] = torch.stack([add_label(x, label) for x in proj_l[look_axis]], dim=0)
        projections = torch.stack(pad(proj_l), dim=1)
    return projections


def render_cameras(batch: dict, resolution: int) -> Tensor:
    """Draw the first example's context (white) and target (red) cameras, near and far
    planes included, as ``(3, 3, height, width)``; ``batch`` is a ``BatchedTempExample``."""
    # Define colors for context and target views.
    num_context_views = batch["context"]["extrinsics"].shape[1]
    num_target_views = batch["target"]["extrinsics"].shape[1]
    color = torch.ones(
        (num_target_views + num_context_views, 3),
        dtype=torch.float32,
        device=batch["target"]["extrinsics"].device,
    )
    color[num_context_views:, 1:] = 0

    return draw_cameras(
        resolution,
        torch.cat((batch["context"]["extrinsics"][0], batch["target"]["extrinsics"][0])),
        torch.cat((batch["context"]["intrinsics"][0], batch["target"]["intrinsics"][0])),
        color,
        torch.cat((batch["context"]["near"][0], batch["target"]["near"][0])),
        torch.cat((batch["context"]["far"][0], batch["target"]["far"][0])),
    )
