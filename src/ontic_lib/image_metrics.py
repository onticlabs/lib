"""Compatibility imports for :mod:`ontic_lib.metrics.image`."""

from .metrics.image import (
    compute_lpips_values,
    compute_psnr_values,
    compute_ssim_values,
    psnr,
    scalar_psnr,
)

__all__ = [
    "compute_lpips_values",
    "compute_psnr_values",
    "compute_ssim_values",
    "psnr",
    "scalar_psnr",
]
