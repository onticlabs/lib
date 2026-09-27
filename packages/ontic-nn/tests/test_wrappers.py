"""Contract tests for ontic_nn.wrappers (no research packages, no weights)."""

from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest
import torch
import torch.nn as nn

from ontic_nn.dinov2 import DinoVisionTransformer
from ontic_nn.wrappers import (
    BACKBONES,
    BackboneConfig,
    BackboneOutput,
    GTDepthBackboneConfig,
    buffers_to_params,
    convert_to_buffer,
    extract_weights,
    offline_guard,
    resize_to_long_side,
    resolve_checkpoint,
)
from ontic_nn.wrappers import dvlt
from ontic_nn.wrappers import gtdepth as gtdepth_mod

# backbone key -> top-level research module its build() imports first
RESEARCH_MODULES = {
    "da3": "depth_anything_3",
    "ma": "mapanything",
    "vggt": "vggt_omega",
    "pi3x": "pi3",
    "dvlt": "dvlt",
    "moge3": "moge",
}


def test_registry_keys():
    assert set(BACKBONES) == {"da3", "ma", "vggt", "pi3x", "dvlt", "moge3", "gtdepth"}
    for cls in BACKBONES.values():
        assert issubclass(cls, BackboneConfig)


@pytest.mark.parametrize("name", sorted(BACKBONES))
def test_config_defaults(name):
    cfg = BACKBONES[name]()
    assert cfg.long_side % 14 == 0 or cfg.long_side % 16 == 0
    assert cfg.checkpoint_path is None and cfg.cache_dir is None and cfg.allow_download
    assert cfg.freeze_dpt_head and cfg.freeze_cam_dec and cfg.freeze_cam_enc


@pytest.mark.parametrize("name", sorted(RESEARCH_MODULES))
def test_build_without_research_repo_names_extra(name, monkeypatch):
    # Earlier tests may have imported the package: drop its cached submodules too, or the
    # import inside build() is served from sys.modules and never fails.
    top = RESEARCH_MODULES[name]
    for key in [k for k in sys.modules if k == top or k.startswith(top + ".")]:
        monkeypatch.delitem(sys.modules, key)
    monkeypatch.setitem(sys.modules, top, None)
    with pytest.raises(ImportError, match=rf"ontic-nn\[{name}\]"):
        BACKBONES[name]().build()


def test_base_config_build_not_implemented():
    with pytest.raises(NotImplementedError):
        BackboneConfig().build()


# -- resize_to_long_side ----------------------------------------------------------------
def test_resize_to_long_side_snaps_to_patch():
    x = torch.rand(2, 3, 3, 480, 640)
    y = resize_to_long_side(x, 14, 518)
    assert y.shape == (2, 3, 3, 392, 518)  # 480 * 518/640 = 388.5 -> 28 patches
    y16 = resize_to_long_side(x, 16, 512)
    assert y16.shape == (2, 3, 3, 384, 512)
    tiny = resize_to_long_side(torch.rand(1, 3, 4, 100), 14, 28)
    assert tiny.shape == (1, 3, 14, 28)  # never below one patch


def test_resize_to_long_side_noop_returns_input():
    x = torch.rand(1, 2, 3, 224, 224)
    assert resize_to_long_side(x, 14, 224) is x
    assert resize_to_long_side(x, 16, 224) is x


# -- BackboneOutput ---------------------------------------------------------------------
def test_backbone_output_camera_fallback_and_features():
    gt_e, pred_e = torch.eye(4)[None, None], 2 * torch.eye(4)[None, None]
    gt_k, pred_k = torch.eye(3)[None, None], 2 * torch.eye(3)[None, None]
    out = BackboneOutput(data={"depth": torch.ones(1, 1, 4, 6)})
    assert out.get_extrinsics() is None and out.get_intrinsics() is None
    assert out.patch_features == []
    assert out.resolution == (4, 6)

    out.data["extrinsics_pred"] = pred_e
    out.data["intrinsics_pred"] = pred_k
    assert out.get_extrinsics() is pred_e and out.get_intrinsics() is pred_k
    out.data["extrinsics"] = gt_e
    out.data["intrinsics"] = gt_k
    assert out.get_extrinsics() is gt_e and out.get_intrinsics() is gt_k

    feats = [torch.zeros(1, 1, 2, 2, 3) for _ in range(4)]
    for i, f in enumerate(feats):
        out.data[f"patch_feat_{i}"] = f
    assert [f is g for f, g in zip(out.patch_features, feats)] == [True] * 4


# -- parameter freezing helpers ---------------------------------------------------------
def test_convert_to_buffer_round_trip():
    torch.manual_seed(0)
    module = nn.Sequential(nn.Linear(3, 4), nn.LayerNorm(4)).eval()
    module.register_buffer("steps", torch.tensor(3))
    reference = {k: v.clone() for k, v in module.state_dict().items()}
    x = torch.randn(5, 3)
    expected = module(x)

    convert_to_buffer(module, persistent=False)
    assert list(module.parameters()) == []
    assert module.state_dict() == {}  # non-persistent buffers
    assert {k for k, _ in module.named_buffers()} == set(reference)
    assert torch.equal(module(x), expected)

    buffers_to_params(module)
    params = dict(module.named_parameters())
    assert set(params) == set(reference) - {"steps"}  # integer buffers stay buffers
    assert dict(module.named_buffers()).keys() == {"steps"}
    for k, v in params.items():
        assert torch.equal(v, reference[k]) and v.requires_grad
    assert torch.equal(module(x), expected)


def test_extract_weights_strips_prefix():
    sd = {"model.head.a": 1, "model.head.b": 2, "model.other": 3}
    assert extract_weights(sd, "model.head.") == {"a": 1, "b": 2}


# -- offline_guard / resolve_checkpoint -------------------------------------------------
def test_offline_guard_restores_env_and_hub(monkeypatch):
    import torch.hub

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    real = torch.hub.download_url_to_file
    with offline_guard(True):
        assert os.environ["HF_HUB_OFFLINE"] == "0"
        assert torch.hub.download_url_to_file is real
    with offline_guard(False):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        with pytest.raises(RuntimeError, match="allow_download=False"):
            torch.hub.download_url_to_file("http://x/y.pth", "/dev/null")
    assert os.environ["HF_HUB_OFFLINE"] == "0"
    assert "TRANSFORMERS_OFFLINE" not in os.environ
    assert torch.hub.download_url_to_file is real


def test_resolve_checkpoint_local_paths(tmp_path):
    f = tmp_path / "model.pt"
    f.write_bytes(b"x")
    assert resolve_checkpoint(str(f), "org/repo") == str(f)
    assert resolve_checkpoint(str(tmp_path), "org/repo") == str(tmp_path)
    assert resolve_checkpoint(str(tmp_path), "org/repo", filename="model.pt") == str(f)
    assert resolve_checkpoint(None, None) is None
    assert resolve_checkpoint("", "") is None
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path / "missing"), "org/repo")
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path), "org/repo", filename="other.pt")


def test_resolve_checkpoint_without_hub_names_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    with pytest.raises(ImportError, match=r"ontic-nn\[vggt\]"):
        resolve_checkpoint(None, "org/repo", extra="vggt")


# -- gtdepth on a tiny DINOv2 ----------------------------------------------------------
def test_gtdepth_forward_with_tiny_dinov2(monkeypatch):
    def tiny_dinov2(variant, use_reg=False, use_checkpointing=False):
        torch.manual_seed(0)
        return DinoVisionTransformer(
            img_size=56, patch_size=14, embed_dim=16, depth=4, num_heads=2, block_chunks=0
        )

    monkeypatch.setattr(gtdepth_mod, "load_pretrained_dinov2", tiny_dinov2)
    monkeypatch.setitem(gtdepth_mod.EMBED_DIM, "vits", 16)
    monkeypatch.setitem(gtdepth_mod.VIT_DEPTH, "vits", 4)
    cfg = GTDepthBackboneConfig(variant="vits", long_side=56, freeze_backbone=True)
    bb = cfg.build().eval()
    assert bb.tap_layers == [0, 1, 2, 3]
    assert bb.encoder_dim == 16 and bb.patch_size == 14 and bb.accepts_gt_cameras
    assert all(not p.requires_grad for p in bb.parameters())

    b, v = 1, 2
    images = torch.rand(b, v, 3, 60, 80)
    depth = torch.rand(b, v, 1, 60, 80) + 0.5
    depth[0, 0, 0, :10] = 0.0  # invalid GT pixels
    extr = torch.eye(4).expand(b, v, 4, 4)
    intr = torch.eye(3).expand(b, v, 3, 3)
    out = bb(images, extr, intr, depth)
    assert out.input_resolution == (42, 56) and out.patch_resolution == (3, 4)
    assert out.data["depth"].shape == (b, v, 42, 56)
    assert torch.equal(out.data["depth_conf"], torch.ones(b, v, 42, 56))
    assert out.data["sky_mask"][0, 0, :6].all() and not out.data["sky_mask"][0, 1].any()
    assert [f.shape for f in out.patch_features] == [(b, v, 3, 4, 16)] * 4
    assert out.get_extrinsics() is extr and out.get_intrinsics() is intr

    out2 = bb(images)
    assert torch.equal(out2.data["depth"], torch.ones(b, v, 42, 56))
    assert "sky_mask" not in out2.data and out2.get_extrinsics() is None


def test_gtdepth_missing_local_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        GTDepthBackboneConfig(checkpoint_path="/nonexistent/dinov2.pth").build()


def test_dvlt_does_not_shadow_an_unloaded_modern_torch_module(monkeypatch):
    name = "torch.nn.attention.flex_attention"
    monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delattr(torch.nn.attention, "flex_attention", raising=False)
    actual = ModuleType(name)
    actual._Backend = object()

    def import_module(module):
        assert module == name
        monkeypatch.setitem(sys.modules, name, actual)
        return actual

    monkeypatch.setattr(dvlt.importlib, "import_module", import_module)
    dvlt.install_flex_attention_shim()
    assert sys.modules[name] is actual


def test_dvlt_still_supports_torch_without_flex_attention(monkeypatch):
    name = "torch.nn.attention.flex_attention"
    monkeypatch.delitem(sys.modules, name, raising=False)

    def import_module(module):
        raise ModuleNotFoundError(name=module)

    monkeypatch.setattr(dvlt.importlib, "import_module", import_module)
    try:
        dvlt.install_flex_attention_shim()
        with pytest.raises(RuntimeError, match="torch >= 2.5"):
            sys.modules[name].flex_attention()
    finally:
        sys.modules.pop(name, None)


def test_dvlt_does_not_hide_a_broken_modern_torch_install(monkeypatch):
    name = "torch.nn.attention.flex_attention"
    monkeypatch.delitem(sys.modules, name, raising=False)

    def import_module(module):
        raise ModuleNotFoundError(name="some_torch_dependency")

    monkeypatch.setattr(dvlt.importlib, "import_module", import_module)
    with pytest.raises(ModuleNotFoundError):
        dvlt.install_flex_attention_shim()
    assert name not in sys.modules
