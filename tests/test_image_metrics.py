from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from ontic_lib.metrics.image import (
    compute_lpips_values,
    compute_psnr_values,
    compute_ssim_values,
    scalar_psnr,
)


def test_legacy_image_metrics_imports_are_preserved():
    from ontic_lib.image_metrics import compute_psnr_values as legacy

    assert legacy is compute_psnr_values


class FakeLpips:
    def net(self, gt, pred, normalize=True):
        assert normalize is True
        return ((gt - pred) ** 2).mean(dim=(1, 2, 3), keepdim=True).view(-1, 1, 1, 1)


def _old_lpips_like(fake_model, gt, pred, *, chunk_size=-1, downsample=1):
    batch_shape = gt.shape[:-3]
    gt_flat = gt.reshape(-1, *gt.shape[-3:])
    pred_flat = pred.reshape(-1, *pred.shape[-3:])

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

    if chunk_size < 0:
        values = fake_model.net(gt_flat, pred_flat, normalize=True)
    else:
        values = torch.cat(
            [
                fake_model.net(
                    gt_flat[i : i + chunk_size],
                    pred_flat[i : i + chunk_size],
                    normalize=True,
                )
                for i in range(0, gt_flat.shape[0], chunk_size)
            ],
            dim=0,
        )
    return values[:, 0, 0, 0].reshape(batch_shape)


def test_scalar_psnr_matches_closed_form():
    assert scalar_psnr([0.0, 0.0], [0.5, 0.5]).item() == pytest.approx(6.0205999)
    assert scalar_psnr([1.0, 2.0], [1.0, 2.0]).item() == float("inf")


def test_compute_psnr_values_returns_per_image_values():
    gt = torch.zeros(2, 3, 4, 4)
    pred = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])

    psnr = compute_psnr_values(gt, pred)

    assert psnr.shape == (2,)
    assert psnr[0] == float("inf")
    assert psnr[1] == pytest.approx(0.0)


def test_compute_psnr_values_supports_masks():
    gt = torch.zeros(1, 1, 2, 2)
    pred = torch.ones_like(gt)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    psnr = compute_psnr_values(gt, pred, mask)

    assert psnr.shape == (1,)
    assert psnr[0] == pytest.approx(0.0)


def test_compute_ssim_values_keeps_batch_shape_and_gradients():
    torch.manual_seed(0)
    gt = torch.rand(2, 3, 32, 32)
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)

    ssim = compute_ssim_values(gt, pred)
    (1 - ssim).mean().backward()

    assert ssim.shape == (2,)
    assert pred.grad is not None
    assert bool(pred.grad.abs().sum() > 0)


def test_compute_lpips_values_matches_chunked_reference():
    torch.manual_seed(0)
    gt = torch.rand(2, 3, 3, 16, 16)
    pred = torch.rand(2, 3, 3, 16, 16, requires_grad=True)
    fake_model = FakeLpips()

    for chunk_size in (-1, 1, 4):
        for downsample in (1, 2):
            current = compute_lpips_values(
                fake_model,
                gt,
                pred,
                chunk_size=chunk_size,
                downsample=downsample,
            )
            expected = _old_lpips_like(
                fake_model,
                gt,
                pred,
                chunk_size=chunk_size,
                downsample=downsample,
            )
            assert torch.equal(current, expected)

    compute_lpips_values(fake_model, gt, pred, chunk_size=2).mean().backward()
    assert pred.grad is not None
    assert bool(pred.grad.abs().sum() > 0)
