"""Tests for splats.rendering (gsplat-backed rasterization)."""

import pytest
import torch

from ontic_lib.splats.rendering import render_gaussians


def _has_gsplat():
    try:
        import gsplat  # noqa: F401
    except ImportError:
        return False
    return True


def _scene(g=200, v=3, seed=0):
    gen = torch.Generator().manual_seed(seed)
    means = torch.rand(g, 3, generator=gen) - 0.5
    scales = torch.rand(g, 3, generator=gen) * 0.02 + 0.005
    covariances = torch.diag_embed(scales**2)
    opacities = torch.rand(g, generator=gen) * 0.8 + 0.2
    c2w = torch.eye(4).repeat(v, 1, 1)
    c2w[:, 2, 3] = -2.0  # cameras at z=-2 looking +z (OpenCV convention)
    c2w[:, 0, 3] = torch.linspace(-0.2, 0.2, v)
    intr = torch.tensor([[1.2, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]]).repeat(v, 1, 1)
    return means, covariances, opacities, c2w, intr


def test_color_argument_validation():
    means, cov, op, c2w, intr = _scene()
    with pytest.raises(ValueError, match="exactly one"):
        render_gaussians(
            means, cov, op, camera_to_world=c2w, intrinsics_normalized=intr, image_size=(8, 8)
        )
    with pytest.raises(ValueError, match="exactly one"):
        render_gaussians(
            means, cov, op,
            sh_coefficients=torch.rand(means.shape[0], 4, 3),
            colors=torch.rand(means.shape[0], 3),
            camera_to_world=c2w, intrinsics_normalized=intr, image_size=(8, 8),
        )


def test_bad_sh_band_count_rejected():
    if not _has_gsplat():
        pytest.skip("needs gsplat for the post-validation path")
    means, cov, op, c2w, intr = _scene()
    with pytest.raises(ValueError, match="bands"):
        render_gaussians(
            means, cov, op,
            sh_coefficients=torch.rand(means.shape[0], 5, 3),
            camera_to_world=c2w, intrinsics_normalized=intr, image_size=(8, 8),
        )


@pytest.mark.skipif(_has_gsplat(), reason="gsplat installed")
def test_missing_gsplat_message():
    means, cov, op, c2w, intr = _scene()
    with pytest.raises(RuntimeError, match=r"ontic-lib\[gsplat\]"):
        render_gaussians(
            means, cov, op,
            colors=torch.rand(means.shape[0], 3),
            camera_to_world=c2w, intrinsics_normalized=intr, image_size=(8, 8),
        )


needs_gpu = pytest.mark.skipif(
    not (torch.cuda.is_available() and _has_gsplat()), reason="needs CUDA + gsplat"
)


@needs_gpu
def test_render_shapes_and_ranges():
    means, cov, op, c2w, intr = (t.cuda() for t in _scene())
    out = render_gaussians(
        means, cov, op,
        sh_coefficients=torch.rand(means.shape[0], 4, 3, device="cuda"),
        camera_to_world=c2w, intrinsics_normalized=intr, image_size=(32, 48),
        render_mode="RGB+D",
    )
    assert out.rgb.shape == (3, 3, 32, 48)
    assert out.alpha.shape == (3, 1, 32, 48) and out.depth.shape == (3, 1, 32, 48)
    assert 0.0 <= out.alpha.min() and out.alpha.max() <= 1.0
    assert (out.depth >= 0).all()


@needs_gpu
def test_single_gaussian_projects_to_image_center():
    g = torch.Generator().manual_seed(0)
    means = torch.zeros(1, 3).cuda()
    cov = (torch.eye(3) * 0.02**2).unsqueeze(0).cuda()
    op = torch.ones(1).cuda()
    c2w = torch.eye(4).unsqueeze(0).cuda()
    c2w[0, 2, 3] = -2.0
    intr = torch.tensor([[[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]]).cuda()
    out = render_gaussians(
        means, cov, op, colors=torch.rand(1, 3, generator=g).cuda(),
        camera_to_world=c2w, intrinsics_normalized=intr, image_size=(33, 33),
        render_mode="RGB+D",
    )
    alpha = out.alpha[0, 0]
    peak = (alpha == alpha.max()).nonzero()[0]
    assert abs(peak[0].item() - 16) <= 1 and abs(peak[1].item() - 16) <= 1
    center_depth = out.depth[0, 0, 16, 16] / out.alpha[0, 0, 16, 16].clamp_min(1e-6)
    assert abs(center_depth.item() - 2.0) < 0.05


@needs_gpu
def test_mask_and_per_view_near_far_grouping():
    means, cov, op, c2w, intr = (t.cuda() for t in _scene(v=4))
    colors = torch.rand(means.shape[0], 3, device="cuda")
    mask = torch.zeros(means.shape[0], dtype=torch.bool, device="cuda")
    mask[::2] = True
    full = render_gaussians(
        means[mask], cov[mask], op[mask], colors=colors[mask],
        camera_to_world=c2w, intrinsics_normalized=intr, image_size=(16, 16),
        near=torch.tensor([0.01, 0.01, 0.5, 0.5]), far=1e3,
    )
    masked = render_gaussians(
        means, cov, op, colors=colors, mask=mask,
        camera_to_world=c2w, intrinsics_normalized=intr, image_size=(16, 16),
        near=torch.tensor([0.01, 0.01, 0.5, 0.5]), far=1e3,
    )
    assert torch.equal(full.rgb, masked.rgb) and torch.equal(full.alpha, masked.alpha)
