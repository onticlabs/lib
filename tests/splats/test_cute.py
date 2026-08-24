"""Tests for the CuTeDSL rasterizer path (splats.cute + rasterizer="cute")."""

import pytest
import torch

from ontic_lib.splats.rendering import render_gaussians


def _has(mod):
    try:
        __import__(mod)
    except ImportError:
        return False
    return True


def _scene(g=300, v=3, seed=0, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    means = (torch.rand(g, 3, generator=gen) - 0.5).to(device)
    scales = torch.rand(g, 3, generator=gen) * 0.02 + 0.005
    covariances = torch.diag_embed(scales**2).to(device)
    opacities = (torch.rand(g, generator=gen) * 0.8 + 0.2).to(device)
    colors = torch.rand(g, 3, generator=gen).to(device)
    c2w = torch.eye(4).repeat(v, 1, 1).to(device)
    c2w[:, 2, 3] = -2.0
    c2w[:, 0, 3] = torch.linspace(-0.2, 0.2, v).to(device)
    intr = (
        torch.tensor([[1.2, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]]).repeat(v, 1, 1).to(device)
    )
    return means, covariances, opacities, colors, c2w, intr


def test_cute_rejects_sh_and_nonuniform_planes():
    means, cov, op, colors, c2w, intr = _scene()
    with pytest.raises(ValueError, match="post-activation"):
        render_gaussians(
            means, cov, op, sh_coefficients=torch.rand(means.shape[0], 4, 3),
            camera_to_world=c2w, intrinsics_normalized=intr, image_size=(8, 8),
            rasterizer="cute",
        )
    with pytest.raises(ValueError, match="uniform near/far"):
        render_gaussians(
            means, cov, op, colors=colors,
            camera_to_world=c2w, intrinsics_normalized=intr, image_size=(8, 8),
            near=torch.tensor([0.01, 0.1, 0.01]), rasterizer="cute",
        )


def test_unknown_rasterizer_rejected():
    means, cov, op, colors, c2w, intr = _scene()
    with pytest.raises(ValueError, match="rasterizer must be"):
        render_gaussians(
            means, cov, op, colors=colors,
            camera_to_world=c2w, intrinsics_normalized=intr, image_size=(8, 8),
            rasterizer="fast",
        )


@pytest.mark.skipif(_has("cutlass"), reason="cutlass installed")
def test_missing_cutlass_message():
    means, cov, op, colors, c2w, intr = _scene()
    with pytest.raises(RuntimeError, match=r"ontic-lib\[cute\]"):
        render_gaussians(
            means, cov, op, colors=colors,
            camera_to_world=c2w, intrinsics_normalized=intr, image_size=(8, 8),
            rasterizer="cute",
        )


def _cute_executable():
    """True when the CuTeDSL can actually drive this GPU (its bundled CUDA
    bindings may need a newer driver than torch does)."""
    if not (torch.cuda.is_available() and _has("gsplat") and _has("cutlass")):
        return False
    try:
        from cuda.bindings import runtime as cuda_runtime

        error, count = cuda_runtime.cudaGetDeviceCount()
        return int(getattr(error, "value", error)) == 0 and count > 0
    except Exception:
        return False


needs_cute = pytest.mark.skipif(
    not _cute_executable(),
    reason="needs CUDA + gsplat + CuTeDSL with a driver its CUDA bindings support",
)


@needs_cute
def test_cute_forward_matches_gsplat():
    means, cov, op, colors, c2w, intr = _scene(device="cuda")
    kwargs = dict(
        camera_to_world=c2w, intrinsics_normalized=intr, image_size=(24, 32),
        near=0.01, far=1e3, render_mode="RGB+D",
    )
    ref = render_gaussians(means, cov, op, colors=colors, **kwargs)
    out = render_gaussians(means, cov, op, colors=colors, rasterizer="cute", **kwargs)
    torch.testing.assert_close(out.rgb, ref.rgb, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out.alpha, ref.alpha, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(out.depth, ref.depth, rtol=1e-5, atol=1e-4)


@needs_cute
def test_cute_backward_matches_gsplat():
    means, cov, op, colors, c2w, intr = _scene(device="cuda")
    kwargs = dict(
        camera_to_world=c2w, intrinsics_normalized=intr, image_size=(16, 16),
        near=0.01, far=1e3, render_mode="RGB",
    )
    grads = {}
    for name in ("gsplat", "cute"):
        m = means.clone().requires_grad_()
        c = colors.clone().requires_grad_()
        o = op.clone().requires_grad_()
        out = render_gaussians(m, cov, o, colors=c, rasterizer=name, **kwargs)
        out.rgb.square().mean().backward()
        grads[name] = (m.grad, c.grad, o.grad)
    for g_ref, g_cute in zip(grads["gsplat"], grads["cute"]):
        torch.testing.assert_close(g_cute, g_ref, rtol=1e-4, atol=1e-6)


@needs_cute
def test_batched_render_multi_scene():
    from ontic_lib.camera.intrinsics import denormalize_intrinsics
    from ontic_lib.splats.cute import batched_render
    from ontic_lib.transforms.rigid import invert_rigid_transform

    h, w, b = 16, 16, 2
    outs = []
    scenes = [_scene(seed=s, device="cuda") for s in range(b)]
    for means, cov, op, colors, c2w, intr in scenes:
        outs.append(
            render_gaussians(
                means, cov, op, colors=colors, camera_to_world=c2w,
                intrinsics_normalized=intr, image_size=(h, w), near=0.01, far=1e3,
            )
        )
    stack = lambda i: torch.stack([s[i] for s in scenes])  # noqa: E731
    colors_b = torch.stack([s[3].unsqueeze(0).expand(3, -1, -1) for s in scenes])
    rc, ra = batched_render(
        stack(0), stack(1), stack(2), colors_b,
        invert_rigid_transform(stack(4)), denormalize_intrinsics(stack(5), (h, w)),
        0.01, 1e3, torch.zeros(b, 3, 3, device="cuda"), w, h,
    )
    for i, ref in enumerate(outs):
        torch.testing.assert_close(
            rc[i, :, :, :, :3].permute(0, 3, 1, 2), ref.rgb, rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(ra[i].permute(0, 3, 1, 2), ref.alpha, rtol=1e-5, atol=1e-5)
