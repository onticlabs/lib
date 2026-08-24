"""Property and known-value tests for ontic_lib.geometry.rotations."""

import torch

from ontic_lib.geometry import rotations as R


def _random_rotations(n, seed=0, dtype=torch.float64):
    """Proper (det=+1) rotation matrices via QR of a Gaussian matrix."""
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 3, 3, generator=g, dtype=dtype)
    q, r = torch.linalg.qr(a)
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    q = q * sign.unsqueeze(-2)
    det = torch.linalg.det(q)
    q = q.clone()
    q[..., 0] = q[..., 0] * det.unsqueeze(-1)
    return q


def test_matrices_are_orthonormal():
    rot = _random_rotations(16, seed=1)
    identity = torch.eye(3, dtype=rot.dtype).expand_as(rot)
    torch.testing.assert_close(rot @ rot.transpose(-1, -2), identity, atol=1e-10, rtol=0)
    torch.testing.assert_close(torch.linalg.det(rot), torch.ones(16, dtype=rot.dtype))


def test_matrix_quaternion_matrix_round_trip():
    rot = _random_rotations(32, seed=2)
    quat = R.matrix_to_quaternion(rot)
    assert quat.shape == (32, 4)
    torch.testing.assert_close(R.quaternion_to_matrix(quat), rot, atol=1e-10, rtol=0)


def test_matrix_6d_matrix_round_trip():
    rot = _random_rotations(32, seed=3)
    six = R.matrix_to_rotation_6d(rot)
    assert six.shape == (32, 6)
    torch.testing.assert_close(R.rotation_6d_to_matrix(six), rot, atol=1e-10, rtol=0)


def test_matrix_procrustes_matrix_round_trip():
    rot = _random_rotations(32, seed=4)
    proc = R.matrix_to_procrustes(rot)
    assert proc.shape == (32, 9)
    torch.testing.assert_close(R.procrustes_to_matrix(proc), rot, atol=1e-10, rtol=0)


def test_quaternion_order_round_trip():
    g = torch.Generator().manual_seed(5)
    wxyz = torch.randn(10, 4, generator=g, dtype=torch.float64)
    xyzw = R.quaternion_wxyz_to_xyzw(wxyz)
    torch.testing.assert_close(R.quaternion_xyzw_to_wxyz(xyzw), wxyz)
    # explicit layout: wxyz -> xyzw moves the real part to the end
    torch.testing.assert_close(xyzw[..., -1], wxyz[..., 0])
    torch.testing.assert_close(xyzw[..., :3], wxyz[..., 1:])


def test_identity_rotation_known_values():
    identity = torch.eye(3, dtype=torch.float64)
    quat = R.matrix_to_quaternion(identity)
    torch.testing.assert_close(
        quat, torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64), atol=1e-10, rtol=0
    )
    six = R.matrix_to_rotation_6d(identity)
    torch.testing.assert_close(
        six, torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64)
    )
    proc = R.matrix_to_procrustes(identity)
    torch.testing.assert_close(proc, identity.reshape(9))


def test_representation_dispatch_matches_specific_converters():
    rot = _random_rotations(8, seed=6)
    for dim, direct in (
        (4, R.matrix_to_quaternion),
        (6, R.matrix_to_rotation_6d),
        (9, R.matrix_to_procrustes),
    ):
        rep = R.matrix_to_rotation_representation(rot, dim)
        torch.testing.assert_close(rep, direct(rot))
        torch.testing.assert_close(
            R.rotation_representation_to_matrix(rep), rot, atol=1e-10, rtol=0
        )


def test_matrix_to_rotation_representation_rejects_bad_dimension():
    rot = _random_rotations(2, seed=7)
    try:
        R.matrix_to_rotation_representation(rot, 5)
    except ValueError as exc:
        assert "4, 6, or 9" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_normalize_rotation_representation_is_projection():
    rot = _random_rotations(8, seed=8)
    for dim in (4, 6, 9):
        rep = R.matrix_to_rotation_representation(rot, dim)
        normalized = R.normalize_rotation_representation(rep)
        again = R.normalize_rotation_representation(normalized)
        torch.testing.assert_close(again, normalized, atol=1e-10, rtol=0)


def test_batched_leading_dimensions_are_preserved():
    rot = _random_rotations(6, seed=9).reshape(2, 3, 3, 3)
    quat = R.matrix_to_quaternion(rot)
    assert quat.shape == (2, 3, 4)
    torch.testing.assert_close(R.quaternion_to_matrix(quat), rot, atol=1e-10, rtol=0)


def test_accumulate_rotation_vectors_composes_left():
    g = torch.Generator().manual_seed(10)
    deltas = torch.randn(1, 4, 3, generator=g, dtype=torch.float64) * 0.1
    acc = R.accumulate_rotation_vectors(deltas)
    assert acc.shape == (1, 4, 3, 3)
    import roma

    step = roma.rotvec_to_rotmat(deltas, epsilon=1e-07)
    expected = step[:, 0]
    torch.testing.assert_close(acc[:, 0], expected, atol=1e-10, rtol=0)
    expected = step[:, 1] @ expected
    torch.testing.assert_close(acc[:, 1], expected, atol=1e-10, rtol=0)
