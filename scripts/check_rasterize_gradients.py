"""Definitive gradient check for the rasterize stage: kernels vs a pure-torch replica.

Finite differences are inconclusive for 3DGS (threshold crossings + extreme
curvature), so this compares autograd against autograd: a pure-torch
re-implementation of gsplat's exact classic compositing (same alpha clamp,
1/255 skip, 1e-4 exclusive termination, background blend, per-image depth
order) whose gradients are correct by construction, versus

  * gsplat's ``rasterize_to_pixels`` CUDA op, and
  * ``ontic_lib.splats.cute.cute_rasterize`` (CuTeDSL fwd + gsplat bwd).

A single-tile image (16x16, tile_size 16) makes tile culling trivial so the
replica is exact. Both forward outputs and input gradients must match to
float32 noise. Run with a gsplat(+cutlass) python:

  PYTHONPATH=src <python> scripts/check_rasterize_gradients.py
"""

from __future__ import annotations

import torch

W = H = TILE = 16
N, IMAGES, CHANNELS = 48, 2, 3
ALPHA_SKIP = 1.0 / 255.0
T_STOP = 1e-4


def torch_composite(means2d, conics, colors, opacities, backgrounds, order):
    """Exact pure-torch replica of gsplat's classic 3DGS forward compositing."""
    device = means2d.device
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    px = xs.reshape(-1).float() + 0.5
    py = ys.reshape(-1).float() + 0.5
    outs, alphas = [], []
    for i in range(means2d.shape[0]):
        T = torch.ones(H * W, device=device)
        acc = torch.zeros(H * W, colors.shape[-1], device=device)
        active = torch.ones(H * W, dtype=torch.bool, device=device)
        for g in order[i]:
            dx = means2d[i, g, 0] - px
            dy = means2d[i, g, 1] - py
            c0, c1, c2 = conics[i, g, 0], conics[i, g, 1], conics[i, g, 2]
            sigma = 0.5 * (c0 * dx * dx + c2 * dy * dy) + c1 * dx * dy
            alpha = (opacities[i, g] * torch.exp(-sigma)).clamp_max(0.999)
            valid = (sigma >= 0) & (alpha >= ALPHA_SKIP) & active
            next_t = T * (1.0 - alpha)
            # exclusive termination: no accumulation on the crossing gaussian
            contribute = valid & (next_t > T_STOP)
            vis = torch.where(contribute, alpha * T, torch.zeros_like(T))
            acc = acc + vis[:, None] * colors[i, g]
            T = torch.where(contribute, next_t, T)
            active = active & ~(valid & (next_t <= T_STOP))
        outs.append((acc + T[:, None] * backgrounds[i]).reshape(H, W, -1))
        alphas.append((1.0 - T).reshape(H, W, 1))
    return torch.stack(outs), torch.stack(alphas)


def make_inputs(device):
    gen = torch.Generator().manual_seed(0)
    means2d = torch.rand(IMAGES, N, 2, generator=gen) * torch.tensor([W - 2.0, H - 2.0]) + 1.0
    # random PSD conics with moderate extent
    a = torch.rand(IMAGES, N, generator=gen) * 0.25 + 0.05
    c = torch.rand(IMAGES, N, generator=gen) * 0.25 + 0.05
    b = (torch.rand(IMAGES, N, generator=gen) - 0.5) * 0.5 * (a * c).sqrt()
    conics = torch.stack([a, b, c], dim=-1)
    colors = torch.rand(IMAGES, N, CHANNELS, generator=gen)
    opacities = torch.rand(IMAGES, N, generator=gen) * 0.85 + 0.1
    backgrounds = torch.rand(IMAGES, CHANNELS, generator=gen)
    depths = torch.rand(IMAGES, N, generator=gen)
    order = torch.argsort(depths, dim=1)
    return tuple(t.to(device) for t in (means2d, conics, colors, opacities, backgrounds, order))


def isect_single_tile(order, device):
    flatten_ids = torch.cat([order[i] + i * N for i in range(IMAGES)]).to(torch.int32)
    offsets = (torch.arange(IMAGES, dtype=torch.int32, device=device) * N).reshape(IMAGES, 1, 1)
    return offsets, flatten_ids.to(device)


def run(kernel_name, kernel_fn, inputs, reference):
    means2d, conics, colors, opacities, backgrounds, order = inputs
    leaves = [t.clone().requires_grad_() for t in (means2d, conics, colors, opacities)]
    rc, ra = kernel_fn(*leaves, backgrounds)
    weights_c = torch.randn_like(rc)
    weights_a = torch.randn_like(ra)
    ((rc * weights_c).sum() + (ra * weights_a).sum()).backward()

    ref_leaves, (ref_rc, ref_ra) = reference
    ok = True
    fwd_err = max((rc - ref_rc).abs().max().item(), (ra - ref_ra).abs().max().item())
    print(f"  {kernel_name}: forward max abs err {fwd_err:.2e}")
    ok &= fwd_err < 1e-5
    # reference backward under the SAME loss weights
    for leaf in ref_leaves:
        leaf.grad = None
    rrc, rra = torch_composite(*[leaf for leaf in ref_leaves], backgrounds, order)
    ((rrc * weights_c).sum() + (rra * weights_a).sum()).backward()
    for name, leaf, ref_leaf in zip(
        ("means2d", "conics", "colors", "opacities"), leaves, ref_leaves
    ):
        err = (leaf.grad - ref_leaf.grad).abs().max().item()
        denom = ref_leaf.grad.abs().max().item()
        print(f"    grad {name:<10} max abs err {err:.2e}  (ref max {denom:.2e})")
        # float32-noise margin: kernel exp/fma ulp differences propagate through
        # the saved alphas into the backward; real bugs are orders of magnitude off
        ok &= err <= max(2e-4, 5e-4 * denom)
    print(f"  {kernel_name}: {'GRADIENTS CORRECT vs torch replica' if ok else 'MISMATCH'}")
    return ok


def check_kernel(name: str, device: str = "cuda") -> bool:
    """Check one kernel ('gsplat' or 'cute') against the replica; True iff correct."""
    inputs = make_inputs(device)
    means2d, conics, colors, opacities, backgrounds, order = inputs
    offsets, flatten_ids = isect_single_tile(order, device)
    ref_leaves = [t.clone().requires_grad_() for t in (means2d, conics, colors, opacities)]
    ref_out = torch_composite(*ref_leaves, backgrounds, order)

    if name == "gsplat":
        from gsplat.cuda._wrapper import rasterize_to_pixels

        def kernel(m, cn, cl, op, bg):
            return rasterize_to_pixels(
                m, cn, cl, op, W, H, TILE, offsets, flatten_ids, backgrounds=bg
            )

    elif name == "cute":
        from ontic_lib.splats.cute import cute_rasterize

        def kernel(m, cn, cl, op, bg):
            return cute_rasterize(m, cn, cl, op, bg, W, H, TILE, offsets, flatten_ids)

    else:
        raise ValueError(f"unknown kernel {name!r}")
    return run(name, kernel, inputs, (ref_leaves, ref_out))


def main():
    results = {}
    for name in ("gsplat", "cute"):
        try:
            results[name] = check_kernel(name)
        except Exception as error:  # noqa: BLE001 — DSL raises its own exception types
            print(f"  {name} skipped ({type(error).__name__})")

    print()
    for name, ok in results.items():
        print(f"{name}: {'CORRECT' if ok else 'WRONG'}")
    if not all(results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    print(f"torch {torch.__version__} | cuda: {torch.cuda.is_available()}")
    main()
