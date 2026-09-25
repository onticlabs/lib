"""Tests for ontic_lib.structures.Gaussians."""

import pytest
import roma
import torch

from ontic_lib.structures import Gaussians


def _make(batch=(), n=5, *, extras=True, mask=True, seed=0, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.rand(*batch, n, *shape, generator=g, dtype=dtype)

    return Gaussians(
        means=rand(3),
        scales=rand(3) + 0.1,
        rotations=rand(4) - 0.5,
        opacities=rand(1),
        harmonics=rand(3, 4),
        mask=(rand(1) > 0.3) if mask else None,
        extras={"feat": rand(7)} if extras else {},
    )


def test_construction_and_properties():
    gs = _make(batch=(2, 3), n=5)
    assert gs.batch_shape == (2, 3)
    assert gs.num_gaussians == 5
    assert gs.dtype == torch.float32
    assert gs.device == torch.device("cpu")
    assert "means=(2, 3, 5, 3)" in repr(gs) and "feat=(2, 3, 5, 7)" in repr(gs)


@pytest.mark.parametrize(
    "bad",
    [
        {"means": torch.zeros(5, 2)},
        {"scales": torch.zeros(4, 3)},
        {"rotations": torch.zeros(5, 3)},
        {"harmonics": torch.zeros(5, 4, 2)},
        {"mask": torch.zeros(5, 1)},  # not bool
        {"extras": {"f": torch.zeros(6, 2)}},
        {"extras": {"means": torch.zeros(5, 2)}},  # collides with a field
    ],
)
def test_validation_rejects_bad_shapes(bad):
    fields = {"means": torch.zeros(5, 3)}
    fields.update(bad)
    with pytest.raises(ValueError):
        Gaussians(**fields)


def test_items_includes_extras_and_skips_none():
    gs = _make(n=4)
    names = [name for name, _ in gs.items()]
    assert names == ["means", "scales", "rotations", "opacities", "harmonics", "mask", "feat"]
    assert all(isinstance(t, torch.Tensor) for _, t in gs.items())


def test_getitem_indexes_batch_dims():
    gs = _make(batch=(2, 3), n=5)
    sub = gs[1]
    assert sub.batch_shape == (3,)
    assert torch.equal(sub.means, gs.means[1])
    assert torch.equal(sub.extras["feat"], gs.extras["feat"][1])
    assert torch.equal(sub.mask, gs.mask[1])
    assert gs[1, 2].batch_shape == ()
    assert gs[:, 0].batch_shape == (2,)
    with pytest.raises(IndexError):
        gs[0, 0, 0]


def test_getitem_on_unbatched_raises():
    with pytest.raises(IndexError):
        _make(n=3)[0]


def test_flatten_unflatten_round_trip():
    gs = _make(batch=(2, 3), n=5)
    flat = gs.flatten_batch()
    assert flat.batch_shape == (6,)
    assert flat.harmonics.shape == (6, 5, 3, 4)
    assert flat.flatten_batch() is flat
    back = flat.unflatten_batch((2, 3))
    for (name, a), (_, b) in zip(gs.items(), back.items()):
        assert torch.equal(a, b), name
    assert _make(n=3).flatten_batch().batch_shape == (1,)
    with pytest.raises(ValueError):
        gs.unflatten_batch((2, 3))


def _frontier_build_covariance(scales, rotations_wxyz, eps=1e-8):
    """Frontier ``build_covariance(normalize=True, eps)`` on ``wxyz`` input, formula only.

    Frontier divides by ``norm + eps`` (twice, in ``to_rot_mat`` then ``quat2rotmat``); the
    normalised-quaternion limit of that is the unit quaternion, used here.
    """
    q = rotations_wxyz / rotations_wxyz.norm(dim=-1, keepdim=True).clamp_min(eps)
    rot = roma.unitquat_to_rotmat(torch.cat([q[..., 1:], q[..., :1]], dim=-1))
    s = scales.diag_embed()
    return rot @ s @ s.transpose(-1, -2) @ rot.transpose(-1, -2)


def test_covariance_matches_frontier_formula():
    gs = _make(batch=(2,), n=16, dtype=torch.float64, seed=3)
    expected = _frontier_build_covariance(gs.scales, gs.rotations)
    torch.testing.assert_close(gs.covariance(), expected, atol=1e-10, rtol=0)
    # Frontier's literal additive-eps normalisation differs by a relative O(eps) only.
    q = gs.rotations / (gs.rotations.norm(dim=-1, keepdim=True) + 1e-8)
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
    literal = _frontier_build_covariance(gs.scales, q, eps=0.0)
    torch.testing.assert_close(gs.covariance(), literal, atol=0, rtol=1e-7)


def test_covariance_prefers_explicit_and_requires_inputs():
    cov = torch.eye(3).expand(4, 3, 3).clone()
    gs = Gaussians(means=torch.zeros(4, 3), covariances=cov)
    assert gs.covariance() is cov
    with pytest.raises(ValueError):
        Gaussians(means=torch.zeros(4, 3)).covariance()


def test_to_float_detach_clone_replace():
    gs = _make(n=3, dtype=torch.float64)
    half = gs.to(torch.float16)
    assert half.means.dtype == torch.float16 and half.mask.dtype == torch.bool
    assert gs.float().dtype == torch.float32 and gs.float().mask.dtype == torch.bool
    cloned = gs.clone()
    assert cloned.means is not gs.means and torch.equal(cloned.means, gs.means)
    assert gs.detach().means.requires_grad is False
    replaced = gs.replace(mask=None, extras={})
    assert replaced.mask is None and replaced.extras == {}
    with pytest.raises(ValueError):
        gs.replace(scales=torch.zeros(2, 3, dtype=torch.float64))
