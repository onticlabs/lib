"""Standalone tests for ontic_lib.depth.alignment."""

import pytest
import torch

from ontic_lib.depth import alignment as D
from ontic_lib.geometry.alignment import align_cameras_sim3
from ontic_lib.geometry.rotations import rotation_6d_to_matrix


def _random_rotation(generator):
    return rotation_6d_to_matrix(torch.randn(6, generator=generator))


def _random_poses(batch, views, generator):
    poses = torch.eye(4).repeat(batch, views, 1, 1).clone()
    for b in range(batch):
        for v in range(views):
            poses[b, v, :3, :3] = _random_rotation(generator)
            poses[b, v, :3, 3] = torch.randn(3, generator=generator)
    return poses


def test_fit_depth_scale_recovers_known_scale():
    g = torch.Generator().manual_seed(0)
    predicted = torch.rand(200, generator=g) + 0.1
    scale = 3.7
    target = scale * predicted
    recovered = D.fit_depth_scale(predicted, target)
    assert recovered.item() == pytest.approx(scale, abs=1e-4)


def test_fit_depth_scale_respects_mask():
    g = torch.Generator().manual_seed(1)
    predicted = torch.rand(100, generator=g) + 0.1
    target = 2.0 * predicted
    # corrupt half the entries, but mask them out.
    target = target.clone()
    target[50:] = 999.0
    mask = torch.zeros(100, dtype=torch.bool)
    mask[:50] = True
    recovered = D.fit_depth_scale(predicted, target, mask=mask)
    assert recovered.item() == pytest.approx(2.0, abs=1e-4)


def test_fit_depth_scale_empty_mask_fallback():
    predicted = torch.rand(10) + 0.1
    target = 2.0 * predicted
    mask = torch.zeros(10, dtype=torch.bool)
    recovered = D.fit_depth_scale(predicted, target, mask=mask)
    assert recovered.item() == pytest.approx(1.0)


def test_fit_depth_scale_and_shift_recovers_affine():
    g = torch.Generator().manual_seed(2)
    predicted = torch.rand(3, 500, generator=g)
    true_scale = torch.tensor([2.0, 0.5, 1.3])
    true_shift = torch.tensor([1.0, -0.3, 0.7])
    target = true_scale[:, None] * predicted + true_shift[:, None]
    scale, shift = D.fit_depth_scale_and_shift(predicted, target)
    assert torch.allclose(scale, true_scale, atol=1e-3)
    assert torch.allclose(shift, true_shift, atol=1e-3)


def test_fit_depth_scale_and_shift_shape_mismatch():
    with pytest.raises(ValueError):
        D.fit_depth_scale_and_shift(torch.rand(2, 4), torch.rand(2, 5))


def test_scale_depth_from_camera_poses_end_to_end():
    g = torch.Generator().manual_seed(3)
    source = _random_poses(2, 6, g)
    rotation = _random_rotation(g)
    translation = torch.randn(3, generator=g)
    scale = torch.tensor(2.3)
    target = align_cameras_sim3(source, rotation, translation, scale)

    depth = torch.rand(2, 4, 4, 1, generator=g) + 0.5
    scaled = D.scale_depth_from_camera_poses(depth, source, target)
    assert torch.allclose(scaled / depth, torch.full_like(depth, 2.3), atol=1e-4)


def test_scale_depth_from_camera_poses_requires_batched_poses():
    # Quirk: bare (V, 4, 4) poses raise (einsum without ellipsis).
    g = torch.Generator().manual_seed(4)
    poses = _random_poses(1, 6, g)[0]  # (V, 4, 4)
    depth = torch.rand(1, 4, 4, 1, generator=g)
    with pytest.raises(RuntimeError):
        D.scale_depth_from_camera_poses(depth, poses, poses)
