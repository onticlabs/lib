"""Standalone tests for ontic_lib.geometry.cameras."""

import pytest
import torch

from ontic_lib.geometry import cameras as C


def _pixel_intrinsics():
    return torch.tensor([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])


def test_normalize_denormalize_round_trip():
    image_size = (480, 640)
    intrinsics = _pixel_intrinsics()
    normalized = C.normalize_intrinsics(intrinsics, image_size)
    restored = C.denormalize_intrinsics(normalized, image_size)
    assert torch.allclose(restored, intrinsics, atol=1e-4)
    # normalized principal point sits near the image center in [0, 1].
    assert abs(normalized[0, 2].item() - 0.5) < 0.01
    assert abs(normalized[1, 2].item() - 0.5) < 0.01


def test_normalize_does_not_mutate_input():
    intrinsics = _pixel_intrinsics()
    before = intrinsics.clone()
    C.normalize_intrinsics(intrinsics, (480, 640))
    assert torch.equal(intrinsics, before)


def test_resize_intrinsics_consistency():
    intrinsics = _pixel_intrinsics()
    original = (480, 640)
    new = (240, 320)  # exactly half
    resized = C.resize_intrinsics(intrinsics, original, new)
    assert torch.allclose(resized[0, 0], intrinsics[0, 0] * 0.5)
    assert torch.allclose(resized[1, 1], intrinsics[1, 1] * 0.5)
    assert torch.allclose(resized[0, 2], intrinsics[0, 2] * 0.5)
    assert torch.allclose(resized[1, 2], intrinsics[1, 2] * 0.5)
    # resize by (1, 1) is a no-op.
    same = C.resize_intrinsics(intrinsics, original, original)
    assert torch.allclose(same, intrinsics)


def test_project_unproject_identity_at_depth():
    g = torch.Generator().manual_seed(0)
    intrinsics = _pixel_intrinsics()
    coords = torch.rand(20, 2, generator=g) * torch.tensor([640.0, 480.0])
    z = torch.rand(20, generator=g) * 5.0 + 1.0
    camera_points = C.unproject_camera_points(coords, z, intrinsics)
    # unprojected points sit at the requested camera-z depth.
    assert torch.allclose(camera_points[..., 2], z, atol=1e-4)
    reprojected = C.project_camera_points(camera_points, intrinsics)
    assert torch.allclose(reprojected, coords, atol=1e-3)


def test_world_rays_unit_norm_directions():
    intrinsics = C.normalize_intrinsics(_pixel_intrinsics(), (480, 640))
    coords, _ = C.sample_image_grid((6, 8))
    camera_to_world = torch.eye(4)
    origins, directions = C.world_rays(coords, camera_to_world, intrinsics)
    norms = directions.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    # identity pose -> origins at world origin.
    assert torch.allclose(origins, torch.zeros_like(origins), atol=1e-6)


def test_sample_image_grid_shapes_and_centers():
    coords, indices = C.sample_image_grid((2, 3))
    assert coords.shape == (2, 3, 2)
    assert indices.shape == (2, 3, 2)
    # indices are integer (row, col) ij pairs.
    assert torch.equal(indices[0, 0], torch.tensor([0, 0]))
    assert torch.equal(indices[1, 2], torch.tensor([1, 2]))
    # coordinates are normalized pixel centers ((x + 0.5) / W, (y + 0.5) / H).
    assert torch.allclose(coords[0, 0], torch.tensor([0.5 / 3, 0.5 / 2]))


def test_project_world_points_in_front_flag():
    intrinsics = _pixel_intrinsics()
    camera_to_world = torch.eye(4)
    points = torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -2.0]])
    _, in_front = C.project_world_points(points, camera_to_world, intrinsics)
    assert bool(in_front[0]) and not bool(in_front[1])


def test_unproject_batched_intrinsics_all_pairs_quirk():
    # Quirk: coords (N, 2) x broadcastable intrinsics (K, 1, 3, 3) -> (K, N, 3)
    # all-pairs product.
    coords = torch.rand(5, 2)
    z = torch.ones(5)
    intrinsics = _pixel_intrinsics().expand(3, 1, 3, 3)
    out = C.unproject_camera_points(coords, z, intrinsics)
    assert out.shape == (3, 5, 3)


def test_unproject_multidim_leading_coords_raise():
    # Quirk: multi-dim leading coords against batched intrinsics raise.
    coords = torch.rand(4, 5, 2)
    z = torch.ones(4, 5)
    intrinsics = _pixel_intrinsics().expand(3, 1, 3, 3)
    with pytest.raises(RuntimeError):
        C.unproject_camera_points(coords, z, intrinsics)


def test_world_pixel_size_positive():
    intrinsics = C.normalize_intrinsics(_pixel_intrinsics(), (480, 640))
    size = C.world_pixel_size(intrinsics, (480, 640))
    assert torch.all(size > 0)
