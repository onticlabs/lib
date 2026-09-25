"""Parity against the frontier originals (runs only where ``fwomo_3d`` imports)."""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
import torch

f_view_sampler = pytest.importorskip("fwomo_3d.data.view_sampler")
f_collate = pytest.importorskip("fwomo_3d.data.collate")
f_shim = pytest.importorskip("fwomo_3d.data.data_shim")

from ontic_data.collate import collate_examples  # noqa: E402
from ontic_data.shims import apply_augmentation_shim, apply_crop_shim  # noqa: E402
from ontic_data.view_sampler import ViewSampler, ViewSamplerCfg  # noqa: E402

RIG = [
    "1-1",
    "1-2",
    "2-1",
    "2-2",
    "3-1",
    "3-2",
    "wrist_camera_r",
    "wrist_camera_l",
    "scene_camera",
    "birds_eye",
]


def _assert_same(a, b, path="", atol=1e-6):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), f"{path}: keys {set(a)} != {set(b)}"
        for k in a:
            _assert_same(a[k], b[k], f"{path}.{k}", atol)
    elif isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor), f"{path}: {type(b)}"
        assert a.shape == b.shape and a.dtype == b.dtype, (
            f"{path}: {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}"
        )
        if a.is_floating_point():
            assert torch.allclose(a, b, atol=atol, rtol=1e-5), (
                f"{path}: max diff {(a - b).abs().max()}"
            )
        else:
            assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.allclose(a, b, atol=atol), path
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{path}[{i}]", atol)
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


# --------------------------------------------------------------------------- #
# view sampler
# --------------------------------------------------------------------------- #

SAMPLER_CASES = {
    "default": dict(),
    "includes_context": dict(num_context_views=2, num_target_views=2, target_includes_context=True),
    "all_targets": dict(num_context_views=2, num_target_views=-1),
    "all_views": dict(num_context_views=-1, num_target_views=-1),
    "fixed_groups": dict(
        num_context_views=2, num_target_views=1, context_views=[[0, 1, 2], [3, 4, 5]]
    ),
    "fixed_superset": dict(num_context_views=2, num_target_views=2, context_views=[0, 2, 4, 5]),
    "paired": dict(num_context_views=3, num_target_views=3, paired_views=True),
}


@pytest.mark.parametrize("name", list(SAMPLER_CASES))
def test_view_sampler_parity(name):
    kw = SAMPLER_CASES[name]
    theirs = f_view_sampler.ViewSamplerCfg(**kw).build("train")
    ours = ViewSampler(ViewSamplerCfg(**kw), "train")  # global RNG, like frontier
    extr, intr = torch.eye(4).expand(10, 4, 4), torch.eye(3).expand(10, 3, 3)
    for t in range(6):
        torch.manual_seed(100 + t)
        a = theirs.sample("scene", extr, intr, camera_group_ix=t, cam_names=RIG)
        torch.manual_seed(100 + t)
        b = ours.sample("scene", extr, intr, camera_group_ix=t, cam_names=RIG)
        _assert_same(list(a), list(b), name)


# --------------------------------------------------------------------------- #
# collate + shims
# --------------------------------------------------------------------------- #


def _sample(T, seed):
    g = torch.Generator().manual_seed(seed)
    return {
        "scene": f"s{seed}",
        "context": {
            "image": torch.rand(1, 2, 3, 6, 8, generator=g),
            "index": torch.arange(2)[None],
        },
        "target": {"image": torch.rand(T, 1, 3, 6, 8, generator=g), "near": torch.ones(T, 1)},
        "workspace_min": torch.zeros(3),
        "actions": {
            "hand": torch.rand(T, 1, 21, 4, generator=g),
            **({"obj": torch.rand(T, 1, 1, 4)} if seed % 2 else {}),
        },
    }


def test_collate_parity_ragged_horizons():
    batch = [_sample(5, 0), _sample(7, 1), _sample(6, 2)]
    _assert_same(
        f_collate.my_custom_collate_fn(copy.deepcopy(batch)), collate_examples(copy.deepcopy(batch))
    )
    same = [_sample(4, 3), _sample(4, 4)]
    _assert_same(
        f_collate.my_custom_collate_fn(copy.deepcopy(same)), collate_examples(copy.deepcopy(same))
    )


def _shim_example(h=20, w=24):
    g = torch.Generator().manual_seed(0)

    def views(t, v):
        return {
            "image": torch.rand(t, v, 3, h, w, generator=g),
            "depth": torch.rand(t, v, 1, h, w, generator=g),
            "state_mask": torch.rand(t, v, 1, h, w, generator=g),
            "intrinsics": torch.rand(t, v, 3, 3, generator=g),
            "extrinsics": torch.rand(t, v, 4, 4, generator=g),
            "near": torch.ones(t, v),
        }

    return {"scene": "s", "context": views(1, 2), "target": views(3, 1)}


@pytest.mark.parametrize("shape", [(16, 16), (20, 24), (10, 20)])
def test_crop_shim_parity(shape):
    ex = _shim_example()
    _assert_same(
        f_shim.apply_crop_shim(copy.deepcopy(ex), shape), apply_crop_shim(copy.deepcopy(ex), shape)
    )


def test_augmentation_shim_parity():
    ex = _shim_example()
    for seed in range(6):
        torch.manual_seed(seed)
        a = f_shim.apply_augmentation_shim(copy.deepcopy(ex))
        torch.manual_seed(seed)
        b = apply_augmentation_shim(copy.deepcopy(ex))
        _assert_same(a, b, f"seed{seed}")


# --------------------------------------------------------------------------- #
# datasets (need the data roots on this machine)
# --------------------------------------------------------------------------- #


def _dataset_pair(name):
    from ontic_data import DATASETS

    f_data = pytest.importorskip("fwomo_3d.data")
    theirs_cls = {
        "genesis": f_data.DatasetGenesisCfg,
        "hocap": f_data.HocapDatasetCfg,
        "physinone": f_data.PhysInOneDatasetCfg,
        "synthrobot": f_data.SynthRobotDatasetCfg,
    }[name]
    ours_cls = DATASETS[name]
    root = ours_cls().roots[0] if name == "genesis" else ours_cls().root
    if not (os.path.isdir(root) and os.access(root, os.R_OK)):
        pytest.skip(f"{name} root not readable: {root}")
    return theirs_cls(), ours_cls()


@pytest.mark.parametrize("name", ["genesis", "hocap", "physinone", "synthrobot"])
def test_dataset_parity_val(name):
    theirs_cfg, ours_cfg = _dataset_pair(name)
    theirs = theirs_cfg.build("val")
    ours = ours_cfg.build("val")
    assert len(theirs) == len(ours)
    assert theirs.record_labels() == ours.record_labels()
    torch.manual_seed(0)
    a = theirs[0]
    torch.manual_seed(0)
    b = ours[0]
    _assert_same(a, b, name, atol=1e-5)
