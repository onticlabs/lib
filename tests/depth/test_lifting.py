"""Standalone tests for ontic_lib.depth.lifting."""

import pytest
import torch

from ontic_lib.geometry import cameras as C
from ontic_lib.depth import lifting as L


def _normalized_intrinsics():
    pixels = torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    return C.normalize_intrinsics(pixels, (480, 640))


def test_constant_depth_plane_lands_at_expected_world_plane():
    intrinsics = _normalized_intrinsics()
    image_size = (8, 10)
    depth = torch.full((*image_size, 1), 3.0)
    points = L.depth_to_world_points(
        depth, torch.eye(4), intrinsics, image_size, depth_type="z"
    )
    assert points.shape == (*image_size, 3)
    # Camera-z depth of a constant plane -> constant world z (identity pose).
    assert torch.allclose(points[..., 2], torch.full_like(points[..., 2], 3.0), atol=1e-5)


def test_z_and_ray_depth_types_differ():
    intrinsics = _normalized_intrinsics()
    image_size = (8, 10)
    depth = torch.full((*image_size, 1), 3.0)
    z_points = L.depth_to_world_points(
        depth, torch.eye(4), intrinsics, image_size, depth_type="z"
    )
    ray_points = L.depth_to_world_points(
        depth, torch.eye(4), intrinsics, image_size, depth_type="ray"
    )
    # Ray distance produces a curved (non-constant-z) surface off the optical axis.
    assert not torch.allclose(z_points, ray_points, atol=1e-3)
    ray_z = ray_points[..., 2]
    assert ray_z.max() > ray_z.min() + 0.1
    # ray distances from the camera center are all equal to the given depth.
    assert torch.allclose(ray_points.norm(dim=-1), torch.full_like(ray_z, 3.0), atol=1e-4)


def test_invalid_depth_type_raises():
    intrinsics = _normalized_intrinsics()
    depth = torch.full((4, 5, 1), 2.0)
    with pytest.raises(ValueError):
        L.depth_to_world_points(depth, torch.eye(4), intrinsics, (4, 5), depth_type="bad")


def test_return_coordinates():
    intrinsics = _normalized_intrinsics()
    image_size = (4, 5)
    depth = torch.full((*image_size, 1), 2.0)
    points, coordinates = L.depth_to_world_points(
        depth, torch.eye(4), intrinsics, image_size, return_coordinates=True
    )
    assert points.shape == (*image_size, 3)
    assert coordinates.shape[-1] == 2


def test_batched_depth_broadcasts_poses():
    intrinsics = _normalized_intrinsics()
    image_size = (4, 5)
    batch, views = 2, 3
    depth = torch.rand(batch, views, *image_size, 1) + 1.0
    camera_to_world = torch.eye(4).repeat(batch, views, 1, 1)
    intrinsics_b = intrinsics.repeat(batch, views, 1, 1)
    points = L.depth_to_world_points(depth, camera_to_world, intrinsics_b, image_size)
    assert points.shape == (batch, views, *image_size, 3)
