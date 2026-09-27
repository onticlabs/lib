"""Grad-mode contract of the DA3 wrapper: a trainable camera encoder must not re-enable
autograd under an outer ``torch.no_grad()`` (validation) and must still train outside it.

Runs on the random-init ``da3-small`` preset (needs ``depth_anything_3``, no weights).
"""

from __future__ import annotations

import gc
import sys

import pytest
import torch

from ontic_nn.wrappers import BACKBONES



def _require(module: str, top: str) -> None:
    """Skip the module when the research package is missing, leaving sys.modules as it was
    (a failed import of a namespace package would otherwise stay cached)."""
    try:
        __import__(module)
    except ImportError:
        for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
            del sys.modules[key]
        pytest.skip(f"needs the {top} research package", allow_module_level=True)

_require("depth_anything_3.api", "depth_anything_3")

B, V, SIDE = 1, 2, 56  # 4x4 patches: fast on CPU
DEVICES = [
    "cpu",
    pytest.param(
        "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    ),
]


def _build(device: str, **overrides):
    torch.manual_seed(0)
    cfg = BACKBONES["da3"](model_dir="", model_name="da3-small", long_side=SIDE, **overrides)
    return cfg.build().to(device).eval()


def _inputs(device: str):
    torch.manual_seed(1)
    images = torch.rand(B, V, 3, SIDE, SIDE, device=device)
    ext = torch.eye(4, device=device).expand(B, V, 4, 4).clone()
    ext[:, 1, 0, 3] = 0.2
    k = torch.tensor([[0.6, 0.0, 0.5], [0.0, 0.6, 0.5], [0.0, 0.0, 1.0]], device=device)
    return images, ext, k.expand(B, V, 3, 3).clone()


def _cam_enc(bb):
    return next(m for name, m in bb.named_modules() if name.endswith("cam_enc"))


def _watch(module):
    """Forward hook recording the grad mode ``module`` ran under and whether it built a graph."""
    seen = {}

    def hook(_module, _inputs, output):
        seen["grad_enabled"] = torch.is_grad_enabled()
        seen["graph"] = output.grad_fn is not None

    return seen, module.register_forward_hook(hook)


@pytest.mark.parametrize("device", DEVICES)
def test_no_grad_builds_no_graph_with_trainable_cam_enc(device):
    bb = _build(device, freeze_cam_enc=False)
    cam_enc = _cam_enc(bb)
    assert all(p.requires_grad for p in cam_enc.parameters())
    images, ext, k = _inputs(device)

    with torch.no_grad():  # warm-up so persistent workspaces (cuBLAS) are in the baseline
        bb(images, ext, k)
    gc.collect()
    if device == "cuda":
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()

    seen, handle = _watch(cam_enc)
    with torch.no_grad():
        out = bb(images, ext, k)
    handle.remove()

    assert seen, "camera encoder did not run"
    assert not seen["grad_enabled"] and not seen["graph"], seen
    for name, t in out.data.items():
        assert not t.requires_grad and t.grad_fn is None, name
    if device == "cuda":
        del out
        gc.collect()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() - before < 2**20  # a retained graph would be 100s of MB


def test_grad_mode_follows_freeze_cam_enc():
    frozen = _build("cpu")  # freeze_cam_enc=True
    seen, handle = _watch(_cam_enc(frozen))
    frozen(*_inputs("cpu"))
    handle.remove()
    assert not seen["grad_enabled"] and not seen["graph"], seen

    bb = _build("cpu", freeze_cam_enc=False, freeze_backbone=False)
    cam_enc = _cam_enc(bb)
    seen, handle = _watch(cam_enc)
    out = bb(*_inputs("cpu"))
    handle.remove()
    assert seen["grad_enabled"] and seen["graph"], seen
    feat = out.data["patch_feat_3"]  # the ViT attends to the camera token
    assert feat.requires_grad
    feat.sum().backward()
    assert all(p.grad is not None for p in cam_enc.parameters())
