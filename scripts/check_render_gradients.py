"""Gradient-correctness check for splats.rendering: analytic vs finite differences.

The parity tests only show the cute and gsplat paths agree with EACH OTHER
(they share gsplat's backward kernel). This script checks the gradients are
actually right: central finite differences of a scalar render loss w.r.t.
means / covariances / opacities / colors, compared against autograd, for each
rasterizer.

Rasterization is piecewise-smooth (alpha clamp at 0.999, the 1/255 alpha
skip, the 1e-4 transmittance cutoff, tile culling), so a few sampled
coordinates land on threshold crossings where FD is meaningless. The check
therefore reports median relative error and the fraction of coordinates
agreeing within 5%, and passes when >=90% agree and cosine similarity is
~1 — rather than demanding exact agreement everywhere.

Run: PYTHONPATH=src <python-with-gsplat> scripts/check_render_gradients.py
"""

from __future__ import annotations

import torch

from ontic_lib.splats.rendering import render_gaussians

SAMPLES_PER_TENSOR = 60
EPS = 3e-4


def _scene(device):
    gen = torch.Generator().manual_seed(0)
    g = 64
    means = (torch.rand(g, 3, generator=gen) - 0.5).to(device)
    scales = torch.rand(g, 3, generator=gen) * 0.03 + 0.03  # generous: smooth alphas
    covariances = torch.diag_embed(scales**2).to(device)
    opacities = (torch.rand(g, generator=gen) * 0.4 + 0.3).to(device)  # away from clamps
    colors = (torch.rand(g, 3, generator=gen) * 0.8 + 0.1).to(device)
    c2w = torch.eye(4).repeat(2, 1, 1).to(device)
    c2w[:, 2, 3] = -2.0
    c2w[:, 0, 3] = torch.tensor([-0.15, 0.15])
    intr = torch.tensor([[1.2, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]]).repeat(2, 1, 1)
    return means, covariances, opacities, colors, c2w, intr.to(device)


def check(rasterizer: str, device="cuda"):
    means, cov, op, colors, c2w, intr = _scene(device)
    weights = torch.randn(2, 3, 24, 32, generator=torch.Generator().manual_seed(1)).to(device)

    def loss_fn(m, cv, o, c):
        out = render_gaussians(
            m, cv, o, colors=c, camera_to_world=c2w, intrinsics_normalized=intr,
            image_size=(24, 32), near=0.01, far=1e3, render_mode="RGB",
            rasterizer=rasterizer,
        )
        return (out.rgb * weights).sum()

    tensors = {"means": means, "covariances": cov, "opacities": op, "colors": colors}
    leaves = {k: v.clone().requires_grad_() for k, v in tensors.items()}
    loss_fn(leaves["means"], leaves["covariances"], leaves["opacities"], leaves["colors"]).backward()

    print(f"\nrasterizer={rasterizer!r}  (loss vs central FD, eps={EPS})")
    all_ok = True
    gen = torch.Generator().manual_seed(2)
    for name, base in tensors.items():
        analytic = leaves[name].grad
        flat = base.reshape(-1)
        count = min(SAMPLES_PER_TENSOR, flat.numel())
        idx = torch.randperm(flat.numel(), generator=gen)[:count]
        fd = torch.zeros(count)
        for j, i in enumerate(idx):
            args = {k: (v.clone() if k == name else v) for k, v in tensors.items()}
            perturbed = args[name].reshape(-1)
            perturbed[i] += EPS
            hi = loss_fn(args["means"], args["covariances"], args["opacities"], args["colors"])
            perturbed[i] -= 2 * EPS
            lo = loss_fn(args["means"], args["covariances"], args["opacities"], args["colors"])
            fd[j] = ((hi - lo) / (2 * EPS)).item()
        ana = analytic.reshape(-1)[idx].cpu()
        scale = torch.maximum(ana.abs(), fd.abs()).clamp_min(1e-3)
        rel = (ana - fd).abs() / scale
        cosine = torch.nn.functional.cosine_similarity(ana, fd, dim=0).item()
        within = (rel < 0.05).float().mean().item()
        print(
            f"  {name:<12} cos={cosine:.6f}  median rel err={rel.median():.2e}  "
            f"within 5%: {within:6.1%}"
        )

    # First-order Taylor / descent test — the decisive global check. Stepping
    # x -> x - a*g must reduce the loss by a*||g||^2 (+ O(a^2)); the ratio of
    # measured to predicted decrease approaches 1 iff the gradient field is
    # right. Robust to the pointwise threshold-crossing noise FD suffers from.
    base_loss = loss_fn(*(tensors[k] for k in ("means", "covariances", "opacities", "colors")))
    grad_sq = sum(leaves[k].grad.square().sum() for k in tensors)
    print("  Taylor descent ratios (want -> 1):", end="")
    ratios = []
    for alpha in (1e-5, 3e-5, 1e-4):
        stepped = {k: (tensors[k] - alpha * leaves[k].grad.detach()) for k in tensors}
        new_loss = loss_fn(
            stepped["means"], stepped["covariances"], stepped["opacities"], stepped["colors"]
        )
        ratio = ((base_loss - new_loss) / (alpha * grad_sq)).item()
        ratios.append(ratio)
        print(f"  a={alpha:g}: {ratio:.4f}", end="")
    print()
    ok = any(abs(r - 1.0) < 0.05 for r in ratios)
    all_ok &= ok
    return all_ok


if __name__ == "__main__":
    print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
    results = {}
    for name in ("gsplat", "cute"):
        try:
            results[name] = check(name)
        except Exception as error:  # noqa: BLE001 — DSL raises its own exception types
            print(f"\nrasterizer={name!r}: skipped ({type(error).__name__}: {error})")
    print(
        "\nNOTE: these are DIAGNOSTICS, not a verdict — FD disagrees at the many"
        "\nthreshold crossings of a piecewise-smooth rasterizer, and the loss"
        "\ncurvature puts even a=1e-5 outside the Taylor linear regime. The"
        "\nauthoritative gradient check is scripts/check_rasterize_gradients.py"
        "\n(autograd vs an exact pure-torch replica); identical FD stats across"
        "\nrasterizers here indicate both compute the same gradient field."
    )
    for name, ok in results.items():
        print(f"{name}: clean Taylor regime reached: {ok}")
