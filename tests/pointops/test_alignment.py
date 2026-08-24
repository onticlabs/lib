"""Standalone tests for ontic_lib.pointops.alignment."""

import pytest
import torch

from ontic_lib.pointops import alignment as A
from ontic_lib.transforms.rotations import rotation_6d_to_matrix


def _random_rotation(generator):
    return rotation_6d_to_matrix(torch.randn(6, generator=generator))


def _random_poses(batch, views, generator):
    poses = torch.eye(4).repeat(batch, views, 1, 1).clone()
    for b in range(batch):
        for v in range(views):
            poses[b, v, :3, :3] = _random_rotation(generator)
            poses[b, v, :3, 3] = torch.randn(3, generator=generator)
    return poses


def test_sim3_recovers_known_transform():
    g = torch.Generator().manual_seed(1)
    source = _random_poses(2, 6, g)
    rotation = _random_rotation(g)
    translation = torch.randn(3, generator=g)
    scale = torch.tensor(2.3)
    target = A.align_cameras_sim3(source, rotation, translation, scale)

    recovered_rotation, recovered_translation, recovered_scale = A.align_camera_poses_sim3(
        source, target
    )
    assert torch.allclose(recovered_rotation[0], rotation, atol=1e-5)
    assert torch.allclose(recovered_translation[0], translation, atol=1e-5)
    assert torch.allclose(recovered_scale, torch.full_like(recovered_scale, 2.3), atol=1e-5)


def test_se3_recovers_known_rigid_transform():
    g = torch.Generator().manual_seed(2)
    source = _random_poses(2, 5, g)
    rotation = _random_rotation(g)
    translation = torch.randn(3, generator=g)
    target = A.align_cameras_sim3(source, rotation, translation, torch.tensor(1.0))

    recovered_rotation, recovered_translation = A.align_camera_poses_se3(source, target)
    assert torch.allclose(recovered_rotation[0], rotation, atol=1e-5)
    assert torch.allclose(recovered_translation[0], translation, atol=1e-5)


def test_align_points_sim3_matches_manual():
    g = torch.Generator().manual_seed(3)
    points = torch.randn(4, 7, 3, generator=g)
    rotation = _random_rotation(g).expand(4, 3, 3)
    translation = torch.randn(4, 3, generator=g)
    scale = torch.rand(4, generator=g) + 0.5

    out = A.align_points_sim3(points, rotation, translation, scale)
    manual = scale[:, None, None] * torch.matmul(points, rotation.transpose(-1, -2)) + (
        translation[:, None]
    )
    assert torch.allclose(out, manual, atol=1e-5)


def test_clamp_scale_bounds():
    scale = torch.tensor([1e-9, 1.0, 1e9, float("nan"), float("inf"), float("-inf")])
    clamped = A.clamp_scale(scale, minimum=0.01, maximum=100.0)
    assert torch.all(clamped >= 0.01)
    assert torch.all(clamped <= 100.0)
    assert torch.isfinite(clamped).all()
    # nan maps to 1.0 then stays within bounds.
    assert clamped[3].item() == pytest.approx(1.0)
    # +inf clamps to the maximum, -inf to the minimum.
    assert clamped[4].item() == pytest.approx(100.0)
    assert clamped[5].item() == pytest.approx(0.01)


def test_anchor_transform_maps_source_to_target():
    g = torch.Generator().manual_seed(4)
    source = torch.eye(4)
    source[:3, :3] = _random_rotation(g)
    source[:3, 3] = torch.randn(3, generator=g)
    target = torch.eye(4)
    target[:3, :3] = _random_rotation(g)
    target[:3, 3] = torch.randn(3, generator=g)

    transform = A.anchor_transform(source, target)
    assert torch.allclose(transform @ source, target, atol=1e-5)
