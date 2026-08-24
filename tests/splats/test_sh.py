"""Tests for ontic_lib.splats.sh (e3nn-backed)."""

import torch

from ontic_lib.splats import sh as SH


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


def test_rotate_sh_is_alias():
    assert SH.rotate_sh is SH.rotate_spherical_harmonics


def test_safe_acos_matches_acos_in_interior_and_is_finite_at_endpoints():
    x = torch.linspace(-0.9, 0.9, 19, dtype=torch.float64)
    torch.testing.assert_close(SH._safe_acos(x), torch.acos(x), atol=1e-6, rtol=0)
    endpoints = torch.tensor([-1.0, 1.0], dtype=torch.float64)
    out = SH._safe_acos(endpoints)
    assert bool(torch.isfinite(out).all())


def test_direction_to_angles_reconstructs_direction():
    from e3nn.o3 import angles_to_matrix

    g = torch.Generator().manual_seed(1)
    direction = torch.randn(8, 3, generator=g, dtype=torch.float64)
    unit = torch.nn.functional.normalize(direction, p=2, dim=-1)
    alpha, beta = SH.direction_to_angles(unit)
    rebuilt = angles_to_matrix(
        alpha, beta, torch.zeros_like(alpha)
    ) @ unit.new_tensor([0.0, 1.0, 0.0])
    torch.testing.assert_close(rebuilt, unit, atol=1e-6, rtol=0)


def test_band0_is_rotation_invariant():
    g = torch.Generator().manual_seed(2)
    coeffs = torch.randn(5, 1, generator=g, dtype=torch.float64)
    rot = _random_rotations(5, seed=3)
    out = SH.rotate_spherical_harmonics(coeffs, rot)
    torch.testing.assert_close(out, coeffs, atol=1e-6, rtol=0)


def test_identity_rotation_is_noop():
    g = torch.Generator().manual_seed(4)
    coeffs = torch.randn(3, 9, generator=g, dtype=torch.float64)  # bands l=0,1,2
    identity = torch.eye(3, dtype=torch.float64).expand(3, 3, 3)
    out = SH.rotate_spherical_harmonics(coeffs, identity)
    torch.testing.assert_close(out, coeffs, atol=1e-6, rtol=0)


def test_rotation_preserves_shape_and_norm():
    g = torch.Generator().manual_seed(5)
    coeffs = torch.randn(4, 4, generator=g, dtype=torch.float64)  # bands l=0,1
    rot = _random_rotations(4, seed=6)
    out = SH.rotate_spherical_harmonics(coeffs, rot)
    assert out.shape == coeffs.shape
    # each complete band is rotated by an orthogonal matrix -> band norm preserved
    torch.testing.assert_close(out[..., :1].norm(dim=-1), coeffs[..., :1].norm(dim=-1))
    torch.testing.assert_close(
        out[..., 1:4].norm(dim=-1), coeffs[..., 1:4].norm(dim=-1), atol=1e-6, rtol=0
    )


def test_incomplete_bands_raise():
    coeffs = torch.randn(3, dtype=torch.float64)  # 3 is not a perfect square
    rot = _random_rotations(1, seed=7)[0]
    try:
        SH.rotate_spherical_harmonics(coeffs, rot)
    except ValueError as exc:
        assert "complete bands" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_batched_rotations():
    g = torch.Generator().manual_seed(8)
    coeffs = torch.randn(2, 6, 4, generator=g, dtype=torch.float64)
    rot = _random_rotations(12, seed=9).reshape(2, 6, 3, 3)
    out = SH.rotate_spherical_harmonics(coeffs, rot)
    assert out.shape == (2, 6, 4)
