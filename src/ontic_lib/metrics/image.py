from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as grad_checkpoint

__all__ = [
    "compute_lpips_values",
    "compute_psnr_values",
    "compute_ssim_values",
    "psnr",
    "scalar_psnr",
]


def _flatten_images(x: Tensor) -> tuple[Tensor, torch.Size]:
    if x.ndim < 3:
        raise ValueError(f"expected image tensor with shape (..., C, H, W), got {tuple(x.shape)}")
    batch_shape = x.shape[:-3]
    return x.reshape(-1, *x.shape[-3:]), batch_shape


def _restore_batch(values: Tensor, batch_shape: torch.Size) -> Tensor:
    return values.reshape(batch_shape)


def scalar_psnr(
    ground_truth,
    predicted,
    *,
    max_val: float = 1.0,
) -> Tensor:
    gt = torch.as_tensor(ground_truth, dtype=torch.float64)
    pred = torch.as_tensor(predicted, dtype=torch.float64, device=gt.device)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {pred.shape}")

    mse = torch.mean((gt - pred) ** 2)
    if bool(mse == 0):
        return torch.tensor(float("inf"), dtype=gt.dtype, device=gt.device)
    return 20 * torch.log10(torch.as_tensor(max_val, dtype=gt.dtype, device=gt.device)) - 10 * torch.log10(mse)


def psnr(ground_truth, predicted, *, max_val: float = 1.0) -> float:
    """Scalar PSNR compatibility API returning a Python float."""
    return float(scalar_psnr(ground_truth, predicted, max_val=max_val).item())


def compute_psnr_values(
    ground_truth: Tensor,
    predicted: Tensor,
    masks: Tensor | None = None,
    *,
    max_val: float = 1.0,
    clip: bool = False,
) -> Tensor:
    gt = torch.as_tensor(ground_truth)
    pred = torch.as_tensor(predicted)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {pred.shape}")
    if gt.ndim < 3:
        raise ValueError(f"expected image tensor with shape (..., C, H, W), got {tuple(gt.shape)}")

    gt = gt.to(dtype=torch.float32)
    pred = pred.to(dtype=torch.float32, device=gt.device)
    if clip:
        gt = gt.clip(0.0, max_val)
        pred = pred.clip(0.0, max_val)

    if masks is not None:
        mask = torch.as_tensor(masks, dtype=gt.dtype, device=gt.device)
        if mask.shape[-3] != 1:
            raise ValueError(f"expected mask channel dimension to be 1, got {tuple(mask.shape)}")
        gt = gt * mask
        pred = pred * mask

    mse = ((gt - pred) ** 2).flatten(-3).mean(dim=-1)
    max_val_t = torch.as_tensor(max_val, dtype=gt.dtype, device=gt.device)
    psnr = 20 * torch.log10(max_val_t) - 10 * torch.log10(mse)
    psnr = torch.where(mse == 0, torch.full_like(psnr, float("inf")), psnr)

    if masks is not None:
        n_total = mask.shape[-2] * mask.shape[-1]
        correction_db = 10.0 * torch.log10(n_total / mask.sum(dim=(-1, -2)).clamp_min(1.0))
        psnr = psnr - correction_db.squeeze(-1)

    return psnr


def compute_ssim_values(
    ground_truth: Tensor,
    predicted: Tensor,
    ssim_model: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    gt = torch.as_tensor(ground_truth)
    pred = torch.as_tensor(predicted, device=gt.device)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {pred.shape}")

    gt_flat, batch_shape = _flatten_images(gt)
    pred_flat, _ = _flatten_images(pred)

    if ssim_model is None:
        from torchmetrics.functional.image import structural_similarity_index_measure

        # SSIM variance terms (conv(x^2) - mu^2) are numerically unstable in bf16,
        # so force float32 even under autocast.
        with torch.amp.autocast("cuda", enabled=False):
            ssim = structural_similarity_index_measure(
                pred_flat.float(),
                gt_flat.float(),
                data_range=1.0,
                reduction="none",
            )
    else:
        ssim = ssim_model(pred_flat, gt_flat)
        if ssim.ndim == 0:
            ssim = ssim[None]

    return _restore_batch(ssim, batch_shape)


def compute_lpips_values(
    lpips_model,
    ground_truth: Tensor,
    predicted: Tensor,
    *,
    chunk_size: int = -1,
    downsample: int = 1,
    gradient_checkpointing: bool = False,
) -> Tensor:
    gt = torch.as_tensor(ground_truth)
    pred = torch.as_tensor(predicted, device=gt.device)
    if gt.shape != pred.shape:
        raise ValueError(f"shape mismatch: {gt.shape} vs {pred.shape}")

    gt_flat, batch_shape = _flatten_images(gt)
    pred_flat, _ = _flatten_images(pred)

    if downsample > 1:
        gt_flat = F.interpolate(
            gt_flat,
            scale_factor=1.0 / downsample,
            mode="bilinear",
            align_corners=False,
        )
        pred_flat = F.interpolate(
            pred_flat,
            scale_factor=1.0 / downsample,
            mode="bilinear",
            align_corners=False,
        )

    def _lpips_forward(gt_chunk: Tensor, pred_chunk: Tensor) -> Tensor:
        return lpips_model.net(gt_chunk, pred_chunk, normalize=True)

    if chunk_size < 0:
        if gradient_checkpointing:
            values = grad_checkpoint(_lpips_forward, gt_flat, pred_flat, use_reentrant=False)
        else:
            values = _lpips_forward(gt_flat, pred_flat)
    else:
        chunks = []
        for i in range(0, gt_flat.shape[0], chunk_size):
            gt_chunk = gt_flat[i : i + chunk_size]
            pred_chunk = pred_flat[i : i + chunk_size]
            if gradient_checkpointing:
                value_chunk = grad_checkpoint(_lpips_forward, gt_chunk, pred_chunk, use_reentrant=False)
            else:
                value_chunk = _lpips_forward(gt_chunk, pred_chunk)
            chunks.append(value_chunk)
        values = torch.cat(chunks, dim=0)

    return _restore_batch(values[:, 0, 0, 0], batch_shape)
