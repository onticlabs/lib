"""Grad-mode contract of the VGGT wrapper: an outer ``no_grad`` is never overridden.

Runs a random-init VGGT-Omega (``embed_dim=64``) on CPU; skipped without the research package.
"""

from __future__ import annotations

import sys
from typing import Dict

import pytest
import torch

from ontic_nn.wrappers import vggt as vggt_mod
from ontic_nn.wrappers.vggt import VGGTBackbone, VGGTBackboneConfig



def _require(module: str, top: str) -> None:
    """Skip the module when the research package is missing, leaving sys.modules as it was
    (a failed import of a namespace package would otherwise stay cached)."""
    try:
        __import__(module)
    except ImportError:
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            del sys.modules[key]
        pytest.skip(f"needs the {top} research package", allow_module_level=True)

_require("vggt_omega", "vggt_omega")

SUBMODULES = ("aggregator", "dense_head", "camera_head")


def tiny_backbone(monkeypatch, **freeze) -> VGGTBackbone:
    monkeypatch.setattr(vggt_mod, "amp_dtype", lambda: torch.bfloat16)  # no CUDA context needed
    torch.manual_seed(0)
    cfg = VGGTBackboneConfig(model_dir="", embed_dim=64, long_side=64, **freeze)
    return cfg.build().eval()


def forward_grad_modes(bb: VGGTBackbone, images: torch.Tensor):
    """Forward ``bb``; also the autograd mode seen inside each sub-module call."""
    modes: Dict[str, bool] = {}
    handles = [
        getattr(bb.vggt, name).register_forward_hook(
            lambda m, i, o, name=name: modes.__setitem__(name, torch.is_grad_enabled())
        )
        for name in SUBMODULES
    ]
    try:
        out = bb(images)
    finally:
        for h in handles:
            h.remove()
    assert set(modes) == set(SUBMODULES)
    return out, modes


def test_no_grad_forward_builds_no_graph_when_trainable(monkeypatch):
    bb = tiny_backbone(
        monkeypatch, freeze_backbone=False, freeze_dpt_head=False, freeze_cam_dec=False
    )
    assert all(p.requires_grad for p in bb.parameters())
    with torch.no_grad():
        out, modes = forward_grad_modes(bb, torch.rand(1, 2, 3, 48, 64))
    assert modes == {name: False for name in SUBMODULES}
    for key, value in out.data.items():
        assert not value.requires_grad and value.grad_fn is None, key
    assert out.patch_resolution == (3, 4) and out.data["depth"].shape == (1, 2, 48, 64)


@pytest.mark.parametrize(
    "freeze, trainable, grad_keys",
    [
        (
            dict(freeze_backbone=False, freeze_dpt_head=False, freeze_cam_dec=False),
            set(SUBMODULES),
            {"depth", "depth_conf", "extrinsics_pred", "intrinsics_pred"}
            | {f"patch_feat_{i}" for i in range(4)},
        ),
        (
            dict(freeze_backbone=True, freeze_dpt_head=False, freeze_cam_dec=True),
            {"dense_head"},
            {"depth", "depth_conf"},
        ),
    ],
)
def test_grad_mode_follows_freeze_flags(monkeypatch, freeze, trainable, grad_keys):
    bb = tiny_backbone(monkeypatch, **freeze)
    out, modes = forward_grad_modes(bb, torch.rand(1, 2, 3, 48, 64))
    assert modes == {name: name in trainable for name in SUBMODULES}
    assert {k for k, v in out.data.items() if v.requires_grad} == grad_keys
    out.data["depth"].sum().backward()
    for name in SUBMODULES:  # depth reaches the aggregator and dense head, never the camera head
        has_grad = any(p.grad is not None for p in getattr(bb.vggt, name).parameters())
        assert has_grad == (name in trainable and name != "camera_head"), name
