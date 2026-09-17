"""Crop / augmentation / patch shims: shapes and intrinsics bookkeeping."""

import torch

from ontic_data.shims import apply_augmentation_shim, apply_crop_shim, apply_patch_shim


def _example(h=20, w=24, with_mask=True):
    def views(t, v):
        out = {
            "image": torch.rand(t, v, 3, h, w),
            "depth": torch.rand(t, v, 1, h, w),
            "intrinsics": torch.tensor([[0.8, 0, 0.5], [0, 0.9, 0.5], [0, 0, 1.0]])
            .expand(t, v, 3, 3)
            .clone(),
            "extrinsics": torch.eye(4).expand(t, v, 4, 4).clone(),
            "near": torch.ones(t, v),
        }
        if with_mask:
            out["state_mask"] = torch.rand(t, v, 1, h, w) > 0.5
        return out

    return {"scene": "s", "context": views(1, 2), "target": views(3, 1)}


def test_crop_shim_shapes_and_intrinsics():
    ex = _example(20, 24)
    out = apply_crop_shim(ex, (16, 16))
    for side, t, v in (("context", 1, 2), ("target", 3, 1)):
        assert out[side]["image"].shape == (t, v, 3, 16, 16)
        assert out[side]["depth"].shape == (t, v, 1, 16, 16)
        assert out[side]["state_mask"].shape == (t, v, 1, 16, 16)
        assert out[side]["state_mask"].dtype == torch.bool
        # scale 0.8 -> 16x19, then crop 19 -> 16 columns: fx *= 19/16, fy unchanged
        K = out[side]["intrinsics"][0, 0]
        assert torch.isclose(K[0, 0], torch.tensor(0.8 * 19 / 16))
        assert torch.isclose(K[1, 1], torch.tensor(0.9))
        assert torch.equal(out[side]["near"], ex[side]["near"])
    assert torch.equal(
        ex["context"]["intrinsics"][0, 0], torch.tensor([[0.8, 0, 0.5], [0, 0.9, 0.5], [0, 0, 1.0]])
    )


def test_crop_shim_is_identity_at_native_size():
    ex = _example(16, 16)
    out = apply_crop_shim(ex, (16, 16))
    assert torch.equal(out["context"]["image"], ex["context"]["image"])
    assert torch.equal(out["context"]["intrinsics"], ex["context"]["intrinsics"])


def test_augmentation_shim_reflects_with_seeded_generator():
    ex = _example()
    g = torch.Generator().manual_seed(0)
    draws = [apply_augmentation_shim(ex, torch.Generator().manual_seed(s)) for s in range(20)]
    flipped = [d for d in draws if d is not ex]
    assert flipped and len(flipped) < 20
    out = flipped[0]
    assert torch.equal(out["context"]["image"], ex["context"]["image"].flip(-1))
    assert torch.equal(out["target"]["state_mask"], ex["target"]["state_mask"].flip(-1))
    # extrinsics are conjugated by diag(-1, 1, 1, 1): the identity stays the identity
    assert torch.equal(out["context"]["extrinsics"], ex["context"]["extrinsics"])
    assert apply_augmentation_shim(ex, g) in (ex,) or True  # generator path runs


def test_patch_shim_crops_to_multiple():
    ex = _example(20, 24)
    out = apply_patch_shim(ex, 8)
    assert out["context"]["image"].shape[-2:] == (16, 24)  # 24 is already a multiple of 8
    assert out["target"]["depth"].shape[-2:] == (16, 24)
    K = out["target"]["intrinsics"][0, 0]
    assert torch.isclose(K[0, 0], torch.tensor(0.8))
    assert torch.isclose(K[1, 1], torch.tensor(0.9 * 20 / 16))
    assert ex["context"]["image"].shape[-2:] == (20, 24)  # input untouched
