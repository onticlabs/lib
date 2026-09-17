"""Numerical parity of the ontic_nn wrappers against the fwomo (frontier) originals.

Runs only where ``fwomo_3d`` imports (the frontier venv) on CUDA:

    cd <frontier_world_model> && PYTHONPATH=<lib>/src:<lib>/packages/ontic-nn/src \\
        .venv/bin/python -m pytest --import-mode=importlib --rootdir=<lib> -c <lib>/pyproject.toml \\
        <lib>/packages/ontic-nn/tests/test_wrappers_parity.py -s

Every case needs the upstream package to import and its weights to sit in the local HF /
torch hub cache (``allow_download=False``); otherwise it is skipped with the reason.
"""

from __future__ import annotations

import gc
import importlib
import os

import pytest
import torch

from ontic_nn.metric_depth import METRIC_MODELS
from ontic_nn.wrappers import BACKBONES


def _has(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not _has("fwomo_3d"), reason="needs fwomo_3d"),
]

DEV = "cuda"
LONG_SIDE = 224  # multiple of 14 and 16; a 224x224 input is not resized
B, V = 1, 2
ATOL = 1e-4


def _cached(repo_id: str, filename: str | None = None) -> str:
    """Local HF cache path of a repo (snapshot dir) or a file; skips on a cache miss.

    Besides the default cache, the colon-separated directories in ``ONTIC_PARITY_HF_CACHES``
    (HF-layout ``models--org--name`` caches) are searched.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    caches = [None] + [d for d in os.environ.get("ONTIC_PARITY_HF_CACHES", "").split(":") if d]
    for cache_dir in caches:
        try:
            if filename is None:
                return snapshot_download(repo_id, cache_dir=cache_dir, local_files_only=True)
            return hf_hub_download(repo_id, filename, cache_dir=cache_dir, local_files_only=True)
        except Exception:  # noqa: BLE001
            continue
    pytest.skip(f"{repo_id} not in the local HF cache(s) {caches}")


def _inputs(seed: int = 0):
    torch.manual_seed(seed)
    images = torch.rand(B, V, 3, LONG_SIDE, LONG_SIDE, device=DEV)
    ext = torch.eye(4, device=DEV).expand(B, V, 4, 4).clone()
    ext[:, 1, 0, 3] = 0.2
    k = torch.tensor([[0.6, 0.0, 0.5], [0.0, 0.6, 0.5], [0.0, 0.0, 1.0]], device=DEV)
    return images, ext, k.expand(B, V, 3, 3).clone()


def _run(build, fn):
    """Build a model, run ``fn(model)``, move results to CPU and free the GPU."""
    try:
        model = build().to(DEV).eval()
    except RuntimeError as e:
        pytest.skip(f"cache miss: {str(e)[:200]}")
    try:
        torch.manual_seed(1)
        with torch.inference_mode():
            out = fn(model)
        torch.cuda.synchronize()
        return {k: (v.detach().float().cpu() if torch.is_tensor(v) else v) for k, v in out.items()}
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def _compare(ours: dict, theirs: dict):
    assert set(ours) == set(theirs), (sorted(ours), sorted(theirs))
    for k in sorted(theirs):
        a, b = ours[k], theirs[k]
        if not torch.is_tensor(a):
            assert a == b, k
            continue
        assert a.shape == b.shape, (k, a.shape, b.shape)
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), k
        err = (a - b).abs().max().item()
        assert torch.allclose(a, b, atol=ATOL, rtol=1e-4), f"{k}: max |diff| = {err:.3e}"
    print(f"  compared {len(theirs)} tensors OK")


def _backbone_outputs(model, with_cameras: bool):
    images, ext, k = _inputs()
    kwargs = dict(extrinsics=ext, intrinsics=k) if with_cameras else {}
    out = model(images, **kwargs)
    data = dict(out.data)
    data["input_resolution"] = out.input_resolution
    data["dpt_resolution"] = out.dpt_resolution
    data["patch_resolution"] = out.patch_resolution
    return data


# -- backbones --------------------------------------------------------------------------
# name -> (research module, weights (repo, file), frontier kwargs, our kwargs, GT cameras)
BACKBONE_CASES = {
    "da3": ("depth_anything_3", ("depth-anything/DA3NESTED-GIANT-LARGE-1.1", None), True),
    "ma": ("mapanything", ("facebook/map-anything", None), True),
    "vggt": ("vggt_omega", ("facebook/VGGT-Omega", "vggt_omega_1b_512.pt"), False),
    "pi3x": ("pi3", ("yyfz233/Pi3X", None), True),
    "dvlt": ("dvlt", ("nvidia/dvlt", "model.safetensors"), False),
    "moge3": ("moge", ("Ruicheng/moge-3-vitl", "model.pt"), False),
}


@pytest.mark.parametrize("name", sorted(BACKBONE_CASES))
@pytest.mark.parametrize("with_cameras", [False, True])
def test_backbone_parity(name, with_cameras):
    module, (repo, filename), accepts_cameras = BACKBONE_CASES[name]
    if with_cameras and not accepts_cameras:
        pytest.skip(f"{name} is pose-free")
    if not _has(module):
        pytest.skip(f"{module} does not import here")
    import fwomo_3d.model.backbone as fb

    path = _cached(repo, filename)
    theirs_cfg = fb.BackboneConfig.get_known_choices()[name](model_dir=path, long_side=LONG_SIDE)
    ours_cfg = BACKBONES[name](checkpoint_path=path, long_side=LONG_SIDE, allow_download=False)
    print(f"\n[{name} cameras={with_cameras}] weights: {path}")
    theirs = _run(theirs_cfg.build, lambda m: _backbone_outputs(m, with_cameras))
    ours = _run(ours_cfg.build, lambda m: _backbone_outputs(m, with_cameras))
    _compare(ours, theirs)


def test_gtdepth_parity():
    import fwomo_3d.model.backbone as fb

    def outputs(model):
        images, ext, k = _inputs()
        depth = torch.rand(B, V, 1, LONG_SIDE, LONG_SIDE, device=DEV) + 1.0
        depth[:, 0, :, :20] = 0.0
        return _backbone_outputs_with_depth(model, images, ext, k, depth)

    theirs_cfg = fb.GTDepthBackboneConfig(long_side=LONG_SIDE)
    ours_cfg = BACKBONES["gtdepth"](long_side=LONG_SIDE, allow_download=False)
    theirs = _run(theirs_cfg.build, outputs)
    ours = _run(ours_cfg.build, outputs)
    _compare(ours, theirs)


def _backbone_outputs_with_depth(model, images, ext, k, depth):
    torch.manual_seed(1)
    out = model(images, ext, k, depth)
    data = dict(out.data)
    data["input_resolution"] = out.input_resolution
    data["patch_resolution"] = out.patch_resolution
    return data


# -- metric models ----------------------------------------------------------------------
METRIC_CASES = {
    "da3": ("depth_anything_3", ("depth-anything/DA3NESTED-GIANT-LARGE-1.1", None), 224),
    "unidepth": ("unidepth", ("lpiccinelli/unidepth-v2-vitl14", None), 224),
    "depthpro": ("depth_pro", ("apple/DepthPro", "depth_pro.pt"), 224),
    "metric3d": (None, None, 224),
}


def _metric_outputs(model):
    images, _ext, k = _inputs()
    out = model(images, intrinsics=k if model.requires_intrinsics else None)
    return {"depth": out.depth, "conf": out.conf, "intrinsics": out.intrinsics}


@pytest.mark.parametrize("name", sorted(METRIC_CASES))
def test_metric_parity(name):
    module, weights, long_side = METRIC_CASES[name]
    if module is not None and not _has(module):
        pytest.skip(f"{module} does not import here")
    import fwomo_3d.model.metric as fm

    theirs_cls = fm.MetricDepthConfig.get_known_choices()[name]
    if weights is not None:
        path = _cached(*weights)
        theirs_cfg = theirs_cls(checkpoint_path=path, long_side=long_side, allow_download=False)
        ours_cfg = METRIC_MODELS[name](
            checkpoint_path=path, long_side=long_side, allow_download=False
        )
    else:  # metric3d: torch.hub cache only
        theirs_cfg = theirs_cls(long_side=long_side, allow_download=False)
        ours_cfg = METRIC_MODELS[name](long_side=long_side, allow_download=False)
    print(f"\n[metric {name}]")
    theirs = _run(theirs_cfg.build, _metric_outputs)
    ours = _run(ours_cfg.build, _metric_outputs)
    _compare(ours, theirs)
