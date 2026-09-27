"""Grad-mode and memory behaviour of the Pi3X wrapper (random init on CPU, no weights).

A trainable, gradient-checkpointed Pi3X must not build a graph inside ``torch.no_grad()``
(validation otherwise keeps gigabytes alive per forward), must recompute its blocks in the
backward, and must keep a frozen camera branch out of the graph.
"""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn as nn

from ontic_nn.wrappers import pi3x as pi3x_mod
from ontic_nn.wrappers.pi3x import Pi3XBackboneConfig



def _require(module: str, top: str) -> None:
    """Skip the module when the research package is missing, leaving sys.modules as it was
    (a failed import of a namespace package would otherwise stay cached)."""
    try:
        __import__(module)
    except ImportError:
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            del sys.modules[key]
        pytest.skip(f"needs the {top} research package", allow_module_level=True)

_require("pi3.models.pi3x", "pi3")

B, V, H, W = 1, 2, 84, 112


@pytest.fixture(scope="module")
def backbone():
    torch.manual_seed(0)
    cfg = Pi3XBackboneConfig(
        model_dir="",  # random init
        long_side=112,
        freeze_backbone=False,
        freeze_dpt_head=False,
        gradient_checkpointing=True,
    )
    return cfg.build().eval()


def _inputs():
    torch.manual_seed(0)
    images = torch.rand(B, V, 3, H, W)
    ext = torch.eye(4).expand(B, V, 4, 4).clone()
    k = torch.tensor([[0.8, 0.0, 0.5], [0.0, 0.8, 0.5], [0.0, 0.0, 1.0]])
    return images, ext, k.expand(B, V, 3, 3).clone()


def _count_calls(module: nn.Module, counter: dict):
    def hook(*_):
        counter[module] = counter.get(module, 0) + 1

    return module.register_forward_hook(hook)


def test_cpu_forward_is_float32_and_finite(backbone):
    out = backbone(*_inputs())
    assert out.input_resolution == (H, W) and out.patch_resolution == (6, 8)
    assert out.data["depth"].shape == (B, V, H, W)
    assert [f.shape for f in out.patch_features] == [(B, V, 6, 8, 1024)] * 4
    assert out.data["extrinsics_pred"].shape == (B, V, 4, 4)
    assert out.data["intrinsics_pred"].shape == (B, V, 3, 3)
    for k, t in out.data.items():
        assert t.dtype == torch.float32 and torch.isfinite(t).all(), k
    assert out.data["depth"].requires_grad  # trainable trunk, caller allows autograd


def test_no_graph_under_no_grad(backbone):
    with torch.no_grad():
        out = backbone(*_inputs())
    for k, t in out.data.items():
        assert not t.requires_grad and t.grad_fn is None, k


def test_blocks_are_checkpointed(backbone, monkeypatch):
    m = backbone.pi3x
    blocks = [*m.decoder, *m.encoder.blocks]
    assert all("forward" in vars(blk) for blk in blocks)  # wrapped per instance

    real = pi3x_mod.checkpoint
    checkpointed = []

    def counting(fn, *args, **kwargs):
        checkpointed.append(fn)
        return real(fn, *args, **kwargs)

    monkeypatch.setattr(pi3x_mod, "checkpoint", counting)
    calls: dict = {}
    handles = [_count_calls(m.decoder[0].mlp, calls), _count_calls(m.encoder.blocks[0].mlp, calls)]
    try:
        with torch.no_grad():
            backbone(*_inputs())
        assert checkpointed == [] and set(calls.values()) == {1}

        calls.clear()
        out = backbone(*_inputs())
        assert len(checkpointed) == len(blocks)
        assert set(calls.values()) == {1}
        (out.data["depth"].mean() + sum(f.mean() for f in out.patch_features)).backward()
        assert set(calls.values()) == {2}  # recomputed in the backward
        assert m.decoder[0].mlp.fc1.weight.grad is not None
        assert m.encoder.blocks[0].mlp.fc1.weight.grad is not None
    finally:
        for h in handles:
            h.remove()
        backbone.zero_grad(set_to_none=True)


def test_frozen_camera_branch_is_detached(backbone):
    cam = backbone.pi3x.camera_decoder
    assert pi3x_mod._detach_inputs in cam._forward_pre_hooks.values()
    seen: dict = {}

    def pre_hook(_module, args, kwargs):  # runs after the wrapper's hook, sees its outputs
        seen["inputs"] = [t.requires_grad for t in (*args, *kwargs.values()) if torch.is_tensor(t)]

    def hook(_module, _args, output):
        seen["output"] = output.requires_grad

    handles = [
        cam.register_forward_pre_hook(pre_hook, with_kwargs=True),
        cam.register_forward_hook(hook),
    ]
    try:
        out = backbone(*_inputs())
    finally:
        for h in handles:
            h.remove()
    assert out.data["depth"].requires_grad  # the trunk is in the graph ...
    assert seen["inputs"] and not any(seen["inputs"])  # ... but the camera decoder is not
    assert seen["output"] is False


def test_checkpoint_block_only_when_trainable_and_grad_enabled(monkeypatch):
    torch.manual_seed(0)
    block = nn.Linear(3, 2)
    x = torch.randn(4, 3)
    expected = block(x)
    pi3x_mod._checkpoint_block(block)

    real = pi3x_mod.checkpoint
    n = []
    monkeypatch.setattr(
        pi3x_mod, "checkpoint", lambda fn, *a, **kw: n.append(1) or real(fn, *a, **kw)
    )

    assert torch.equal(block(x), expected) and len(n) == 1
    with torch.no_grad():
        assert torch.equal(block(x), expected) and len(n) == 1
    block.requires_grad_(False)
    assert torch.equal(block(x), expected) and len(n) == 1
    block.requires_grad_(True)
    block(x).sum().backward()
    assert len(n) == 2 and block.weight.grad is not None


def test_detach_inputs_hook():
    x = torch.ones(2, requires_grad=True)
    args, kwargs = pi3x_mod._detach_inputs(nn.Identity(), (x, 3), {"xpos": x, "flag": None})
    assert args[1] == 3 and kwargs["flag"] is None
    assert not args[0].requires_grad and not kwargs["xpos"].requires_grad
    assert torch.equal(args[0], x) and torch.equal(kwargs["xpos"], x)
