"""Property and known-value tests for ontic_lib.geometry.transforms."""

import torch

from ontic_lib.geometry import transforms as T


def _random_rigid(n, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 3, 3, generator=g, dtype=dtype)
    q, r = torch.linalg.qr(a)
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    q = q.clone()
    q[..., 0] = q[..., 0] * det.unsqueeze(-1)
    transform = torch.zeros(n, 4, 4, dtype=dtype)
    transform[..., :3, :3] = q
    transform[..., :3, 3] = torch.randn(n, 3, generator=g, dtype=dtype)
    transform[..., 3, 3] = 1.0
    return transform


def test_homogenize_points_and_vectors():
    g = torch.Generator().manual_seed(1)
    pts = torch.randn(5, 3, generator=g, dtype=torch.float64)
    hp = T.homogenize_points(pts)
    hv = T.homogenize_vectors(pts)
    assert hp.shape == (5, 4)
    torch.testing.assert_close(hp[..., 3], torch.ones(5, dtype=torch.float64))
    torch.testing.assert_close(hv[..., 3], torch.zeros(5, dtype=torch.float64))
    torch.testing.assert_close(hp[..., :3], pts)


def test_identity_transform_is_noop():
    g = torch.Generator().manual_seed(2)
    pts = torch.randn(7, 3, generator=g, dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64)
    torch.testing.assert_close(T.transform_points(identity, pts), pts)
    torch.testing.assert_close(T.transform_vectors(identity, pts), pts)


def test_invert_rigid_transform_is_inverse():
    transform = _random_rigid(4, seed=3)
    inverse = T.invert_rigid_transform(transform)
    identity = torch.eye(4, dtype=transform.dtype).expand_as(transform)
    torch.testing.assert_close(transform @ inverse, identity, atol=1e-10, rtol=0)
    torch.testing.assert_close(inverse @ transform, identity, atol=1e-10, rtol=0)


def test_transform_then_invert_round_trips_points():
    transform = _random_rigid(3, seed=4)
    g = torch.Generator().manual_seed(5)
    pts = torch.randn(3, 10, 3, generator=g, dtype=torch.float64)
    moved = T.transform_points(transform.unsqueeze(1), pts)
    back = T.transform_points(T.invert_rigid_transform(transform).unsqueeze(1), moved)
    torch.testing.assert_close(back, pts, atol=1e-10, rtol=0)


def test_apply_matrix_broadcasts_leading_dims():
    g = torch.Generator().manual_seed(6)
    matrix = torch.randn(3, 3, generator=g, dtype=torch.float64)
    vectors = torch.randn(4, 5, 3, generator=g, dtype=torch.float64)
    out = T.apply_matrix(matrix, vectors)
    assert out.shape == (4, 5, 3)
    torch.testing.assert_close(out, vectors @ matrix.transpose(-1, -2))


def test_transform_camera_to_world_se3_left_multiplies():
    cam = _random_rigid(2, seed=7)
    delta = _random_rigid(1, seed=8)[0]
    out = T.transform_camera_to_world_se3(cam, delta)
    torch.testing.assert_close(out, delta @ cam, atol=1e-10, rtol=0)


def test_transform_camera_to_world_sim3_matches_manual():
    cam = _random_rigid(4, seed=9)
    g = torch.Generator().manual_seed(10)
    rotation = _random_rigid(1, seed=11)[0, :3, :3]
    translation = torch.randn(3, generator=g, dtype=torch.float64)
    scale = torch.rand(1, generator=g, dtype=torch.float64) + 0.5
    out = T.transform_camera_to_world_sim3(cam, rotation, translation, scale)
    expected_rot = rotation @ cam[..., :3, :3]
    expected_center = scale * (cam[..., :3, 3] @ rotation.transpose(-1, -2)) + translation
    torch.testing.assert_close(out[..., :3, :3], expected_rot, atol=1e-10, rtol=0)
    torch.testing.assert_close(out[..., :3, 3], expected_center, atol=1e-10, rtol=0)
    torch.testing.assert_close(out[..., 3, :], cam[..., 3, :])
