"""CuTeDSL batched 3DGS rasterization: one launch for many (scene, camera) pairs.

``gsplat.rasterization`` loops per scene; this pipeline batches ``B`` scenes
x ``C`` cameras through gsplat's batched projection + tile intersection and a
CuTeDSL forward rasterize kernel, with gsplat's exact CUDA backward — so
gradients match the stock path. Forward compositing mirrors gsplat's classic
3DGS kernel (verified against it at rel ~3e-7 float noise).

Regime: post-activation colors (no SH), pinhole cameras, uniform near/far,
precomputed covariances, ``classic`` rasterize mode. Requires the ``cute``
extra (``pip install ontic-lib[cute]``: gsplat + nvidia-cutlass-dsl) and a
CUDA driver new enough for the DSL's bundled CUDA bindings (CUDA 13-era —
newer than what torch itself needs; ``cudaErrorInsufficientDriver`` at launch
means the driver, not the port). Ported from fwomo-3d's
``model/rendering/{cute_raster,cute_pipeline}.py``.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_ALPHA_THRESH = 1.0 / 255.0
_T_EPS = 1e-4
_CACHE: dict = {}


def _cutlass():
    try:
        import cutlass
        import cutlass.cute  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "splats.cute requires the CuTeDSL — install the extra: pip install ontic-lib[cute]"
        ) from error
    # The DSL resolves the kernel's (string) type annotations against this
    # module's globals, so the lazy import must publish `cute`/`cutlass` there.
    globals().setdefault("cutlass", cutlass)
    globals().setdefault("cute", cutlass.cute)
    return cutlass


def _build(W: int, H: int, D: int, TS: int, TW: int, TH: int, NI: int):
    cutlass = _cutlass()
    cute = cutlass.cute
    _F32 = cutlass.Float32

    @cute.kernel
    def raster_fwd_kernel(
        means2d: cute.Tensor,  # [I*N, 2]
        conics: cute.Tensor,  # [I*N, 3]
        colors: cute.Tensor,  # [I*N, D]
        opac: cute.Tensor,  # [I*N]
        bg: cute.Tensor,  # [I, D]
        offs: cute.Tensor,  # [I*TH*TW + 1] int32
        fids: cute.Tensor,  # [n_isects] int32
        outc: cute.Tensor,  # [I*H*W, D]
        outa: cute.Tensor,  # [I*H*W]
        outl: cute.Tensor,  # [I*H*W] int32 (last_ids)
    ):
        tx, ty, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        px = bx * TS + tx
        py = by * TS + ty
        if (px < W) & (py < H):
            tile_flat = bz * (TH * TW) + by * TW + bx
            start = offs[tile_flat]
            end = offs[tile_flat + 1]
            acc = cute.make_fragment(D, _F32)
            for ch in cutlass.range_constexpr(D):
                acc[ch] = _F32(0.0)
            T = _F32(1.0)
            pxf = _F32(px) + _F32(0.5)
            pyf = _F32(py) + _F32(0.5)
            last = cutlass.Int32(0)
            k = start
            while k < end:
                idx = fids[k]
                dx = means2d[idx, 0] - pxf
                dy = means2d[idx, 1] - pyf
                c0 = conics[idx, 0]
                c1 = conics[idx, 1]
                c2 = conics[idx, 2]
                sigma = _F32(0.5) * (c0 * dx * dx + c2 * dy * dy) + c1 * dx * dy
                alpha = opac[idx] * cute.math.exp(-sigma)
                if alpha > _F32(0.999):
                    alpha = _F32(0.999)
                if (sigma >= _F32(0.0)) & (alpha >= _F32(_ALPHA_THRESH)):
                    next_T = T * (_F32(1.0) - alpha)
                    if next_T <= _F32(_T_EPS):
                        k = end  # terminate, exclusive (no accumulate)
                    else:
                        vis = alpha * T
                        for ch in cutlass.range_constexpr(D):
                            acc[ch] = acc[ch] + vis * colors[idx, ch]
                        T = next_T
                        last = k
                        k = k + 1
                else:
                    k = k + 1
            out_pix = bz * (H * W) + py * W + px
            for ch in cutlass.range_constexpr(D):
                outc[out_pix, ch] = acc[ch] + T * bg[bz, ch]
            outa[out_pix] = _F32(1.0) - T
            outl[out_pix] = last

    @cute.jit
    def launch(means2d, conics, colors, opac, bg, offs, fids, outc, outa, outl):
        raster_fwd_kernel(
            means2d, conics, colors, opac, bg, offs, fids, outc, outa, outl
        ).launch(grid=[TW, TH, NI], block=[TS, TS, 1])

    return launch


def _fwd(
    means2d, conics, colors, opacities, backgrounds, isect_offsets, flatten_ids,
    width, height, tile_size=16,
):
    """Low-level forward. ``means2d [I, N, 2]`` etc. Returns render_colors
    ``[I, H, W, D]``, render_alphas ``[I, H, W, 1]``, last_ids ``[I, H, W]``."""
    _cutlass()
    from cutlass.cute import compile as cute_compile
    from cutlass.cute.runtime import from_dlpack

    images, N = means2d.shape[0], means2d.shape[1]
    D = colors.shape[-1]
    TW = (width + tile_size - 1) // tile_size
    TH = (height + tile_size - 1) // tile_size
    n_isects = flatten_ids.shape[0]

    # The forward kernel reads raw values; autograd is handled by the Function's
    # saved tensors + gsplat backward, so detach for the dlpack export.
    m = means2d.detach().reshape(images * N, 2).contiguous()
    c = conics.detach().reshape(images * N, 3).contiguous()
    col = colors.detach().reshape(images * N, D).contiguous()
    op = opacities.detach().reshape(images * N).contiguous()
    bg = backgrounds.detach().reshape(images, D).contiguous()
    offs = torch.cat(
        [
            isect_offsets.reshape(-1).to(torch.int32),
            torch.tensor([n_isects], dtype=torch.int32, device=means2d.device),
        ]
    ).contiguous()
    fids = flatten_ids.reshape(-1).to(torch.int32).contiguous()
    outc = torch.zeros(images * height * width, D, device=means2d.device, dtype=torch.float32)
    outa = torch.zeros(images * height * width, device=means2d.device, dtype=torch.float32)
    outl = torch.zeros(images * height * width, device=means2d.device, dtype=torch.int32)

    args = [from_dlpack(t) for t in (m, c, col, op, bg, offs, fids, outc, outa, outl)]
    key = (width, height, D, tile_size, TW, TH, images)
    if key not in _CACHE:
        _CACHE[key] = cute_compile(_build(width, height, D, tile_size, TW, TH, images), *args)
    _CACHE[key](*args)
    return (
        outc.reshape(images, height, width, D),
        outa.reshape(images, height, width, 1),
        outl.reshape(images, height, width),
    )


class _CuteRasterize(torch.autograd.Function):
    """CuTeDSL forward + gsplat CUDA backward (exact, batched)."""

    @staticmethod
    def forward(
        ctx, means2d, conics, colors, opacities, backgrounds,
        width, height, tile_size, isect_offsets, flatten_ids,
    ):
        rc, ra, last_ids = _fwd(
            means2d, conics, colors, opacities, backgrounds,
            isect_offsets, flatten_ids, width, height, tile_size,
        )
        ctx.save_for_backward(
            means2d, conics, colors, opacities, backgrounds,
            isect_offsets, flatten_ids, ra, last_ids,
        )
        ctx.width, ctx.height, ctx.tile_size = width, height, tile_size
        return rc, ra

    @staticmethod
    def backward(ctx, v_render_colors, v_render_alphas):
        from gsplat.cuda._wrapper import _make_lazy_cuda_func

        (
            means2d, conics, colors, opacities, backgrounds,
            isect_offsets, flatten_ids, render_alphas, last_ids,
        ) = ctx.saved_tensors
        _, v_means2d, v_conics, v_colors, v_opacities = _make_lazy_cuda_func(
            "rasterize_to_pixels_3dgs_bwd"
        )(
            means2d, conics, colors, opacities, backgrounds, None,
            ctx.width, ctx.height, ctx.tile_size, isect_offsets, flatten_ids,
            render_alphas, last_ids,
            v_render_colors.contiguous(), v_render_alphas.contiguous(), False,
        )
        v_backgrounds = None
        if ctx.needs_input_grad[4]:
            v_backgrounds = (v_render_colors * (1.0 - render_alphas).float()).sum(dim=(-3, -2))
        return (
            v_means2d, v_conics, v_colors, v_opacities, v_backgrounds,
            None, None, None, None, None,
        )


def cute_rasterize(
    means2d, conics, colors, opacities, backgrounds,
    width, height, tile_size, isect_offsets, flatten_ids,
):
    """Autograd rasterize: CuTeDSL forward + gsplat backward. ``means2d [I, N, 2]`` etc."""
    return _CuteRasterize.apply(
        means2d, conics, colors, opacities, backgrounds,
        width, height, tile_size, isect_offsets, flatten_ids,
    )


# Row/column indices reading the 6 unique entries of a symmetric 3x3 covariance
# in gsplat's packed order [xx, xy, xz, yy, yz, zz].
_COV6_ROWS = (0, 0, 0, 1, 1, 2)
_COV6_COLS = (0, 1, 2, 1, 2, 2)


def batched_render(
    means: Tensor,  # (B, N, 3)
    covariances: Tensor,  # (B, N, 3, 3)
    opacities: Tensor,  # (B, N)
    colors: Tensor,  # (B, C, N, Dc) post-activation
    viewmats: Tensor,  # (B, C, 4, 4) world-to-camera
    Ks: Tensor,  # (B, C, 3, 3) pixel intrinsics
    near: float,
    far: float,
    background: Tensor,  # (B, C, Dc)
    width: int,
    height: int,
    tile_size: int = 16,
    render_mode: str = "RGB+D",
) -> tuple[Tensor, Tensor]:
    """Render ``B`` scenes (each with its own Gaussians + ``C`` cameras) batched.

    Mirrors ``gsplat.rasterization`` for the post-activation regime but runs
    every stage once over all ``B x C`` images. Returns ``(render_colors
    (B, C, H, W, Dout), render_alphas (B, C, H, W, 1))``; ``Dout`` is ``Dc``
    plus one channel when ``render_mode`` appends depth.
    """
    from gsplat.cuda._wrapper import fully_fused_projection, isect_offset_encode, isect_tiles

    B, N = means.shape[0], means.shape[1]
    C = viewmats.shape[1]
    images = B * C

    covars6 = covariances[..., _COV6_ROWS, _COV6_COLS]
    radii, means2d, depths, conics, _ = fully_fused_projection(
        means, covars6, None, None, viewmats, Ks, width, height, eps2d=0.3,
        packed=False, near_plane=float(near), far_plane=float(far), radius_clip=0.0,
        sparse_grad=False, calc_compensations=False, camera_model="pinhole",
        opacities=opacities,
    )
    opacities_bc = torch.broadcast_to(opacities[:, None, :], (B, C, N))

    # Depth modes carry the projected depth as an extra zero-background channel.
    if render_mode in ("RGB+D", "RGB+ED"):
        colors = torch.cat([colors, depths[..., None]], dim=-1)
        zeros = torch.zeros(B, C, 1, device=background.device, dtype=background.dtype)
        background = torch.cat([background, zeros], dim=-1)

    tiles_w, tiles_h = math.ceil(width / tile_size), math.ceil(height / tile_size)
    _, isect_ids, flatten_ids = isect_tiles(
        means2d.reshape(images, N, 2), radii.reshape(images, N, 2), depths.reshape(images, N),
        tile_size, tiles_w, tiles_h, segmented=False, packed=False,
        n_images=images, image_ids=None, gaussian_ids=None,
    )
    isect_offsets = isect_offset_encode(isect_ids, images, tiles_w, tiles_h)  # [I, TH, TW]

    rc, ra = cute_rasterize(
        means2d.reshape(images, N, 2), conics.reshape(images, N, 3),
        colors.reshape(images, N, colors.shape[-1]), opacities_bc.reshape(images, N),
        background.reshape(images, background.shape[-1]),
        width, height, tile_size, isect_offsets, flatten_ids,
    )

    out_channels = rc.shape[-1]
    if render_mode in ("ED", "RGB+ED"):  # expected depth: normalize by accumulated alpha
        rc = torch.cat([rc[..., :-1], rc[..., -1:] / ra.clamp(min=1e-10)], dim=-1)
    return rc.reshape(B, C, height, width, out_channels), ra.reshape(B, C, height, width, 1)
