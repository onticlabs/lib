"""Tests for ontic_lib.splats.gaussians covariance construction."""

import torch

from ontic_lib.splats import gaussians as G
from ontic_lib.transforms import rotations as R


def _random_rotations(n, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 3, 3, generator=g, dtype=dtype)
    q, r = torch.linalg.qr(a)
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    q = q.clone()
    q[..., 0] = q[..., 0] * det.unsqueeze(-1)
    return q


def test_covariance_equals_r_diag_s2_rt():
    rot = _random_rotations(12, seed=1)
    g = torch.Generator().manual_seed(2)
    scales = torch.rand(12, 3, generator=g, dtype=torch.float64) + 0.1
    cov = G.covariance_from_scale_rotation(scales, rot)
    expected = rot @ torch.diag_embed(scales**2) @ rot.transpose(-1, -2)
    torch.testing.assert_close(cov, expected)


def test_covariance_is_symmetric_and_psd():
    rot = _random_rotations(20, seed=3)
    g = torch.Generator().manual_seed(4)
    scales = torch.rand(20, 3, generator=g, dtype=torch.float64) + 0.05
    cov = G.covariance_from_scale_rotation(scales, rot)
    torch.testing.assert_close(cov, cov.transpose(-1, -2), atol=1e-12, rtol=0)
    eigvals = torch.linalg.eigvalsh(cov)
    assert bool((eigvals > 0).all())
    # eigenvalues of the covariance are exactly the squared scales
    torch.testing.assert_close(
        torch.sort(eigvals, dim=-1).values,
        torch.sort(scales**2, dim=-1).values,
        atol=1e-10,
        rtol=0,
    )


def test_covariance_from_representation_matches_matrix_path():
    rot = _random_rotations(8, seed=5)
    g = torch.Generator().manual_seed(6)
    scales = torch.rand(8, 3, generator=g, dtype=torch.float64) + 0.1
    quat = R.matrix_to_quaternion(rot)
    cov_rep = G.covariance_from_scale_rotation_representation(scales, quat)
    cov_mat = G.covariance_from_scale_rotation(scales, rot)
    torch.testing.assert_close(cov_rep, cov_mat, atol=1e-10, rtol=0)


def test_identity_rotation_gives_diagonal_covariance():
    scales = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float64)
    identity = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    cov = G.covariance_from_scale_rotation(scales, identity)
    torch.testing.assert_close(cov, torch.diag_embed(scales**2))


def test_batched_leading_dimensions():
    rot = _random_rotations(6, seed=7).reshape(2, 3, 3, 3)
    g = torch.Generator().manual_seed(8)
    scales = torch.rand(2, 3, 3, generator=g, dtype=torch.float64) + 0.1
    cov = G.covariance_from_scale_rotation(scales, rot)
    assert cov.shape == (2, 3, 3, 3)
