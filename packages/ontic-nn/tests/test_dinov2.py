"""Tests for ontic_nn.dinov2 (tiny random models; no weights downloaded)."""

import sys

import pytest
import torch

from ontic_nn.dinov2 import DinoVisionTransformer, StandaloneDinoExtractor
from ontic_nn.dinov2 import pretrained as pretrained_mod


def _tiny(**kwargs) -> DinoVisionTransformer:
    torch.manual_seed(0)
    return DinoVisionTransformer(
        img_size=56, patch_size=14, embed_dim=32, depth=2, num_heads=2, block_chunks=0, **kwargs
    ).eval()


def test_forward_features_shapes():
    model = _tiny(num_register_tokens=2, init_values=1.0)
    x = torch.randn(3, 3, 56, 56)
    out = model(x)
    assert out["x_norm_clstoken"].shape == (3, 32)
    assert out["x_norm_regtokens"].shape == (3, 2, 32)
    assert out["x_norm_patchtokens"].shape == (3, 16, 32)
    assert out["x_prenorm"].shape == (3, 1 + 2 + 16, 32)


def test_pos_embed_interpolation_for_other_resolution():
    model = _tiny()
    out = model(torch.randn(1, 3, 28, 42))
    assert out["x_norm_patchtokens"].shape == (1, 2 * 3, 32)


def test_masks_replace_patch_tokens():
    model = _tiny()
    x = torch.randn(1, 3, 56, 56)
    masks = torch.zeros(1, 16, dtype=torch.bool)
    masks[0, 3] = True
    tokens = model.prepare_tokens_with_masks(x, masks)
    expected = model.mask_token[0] + model.pos_embed[0, 1 + 3]
    assert torch.allclose(tokens[0, 1 + 3], expected)


def test_get_intermediate_layers_reshape_and_cls():
    model = _tiny()
    x = torch.randn(2, 3, 56, 56)
    outs = model.get_intermediate_layers(x, n=2, reshape=True, return_class_token=True)
    assert len(outs) == 2
    patches, cls = outs[-1]
    assert patches.shape == (2, 32, 4, 4)
    assert cls.shape == (2, 32)
    flat = model.get_intermediate_layers(x, n=[0])
    assert flat[0].shape == (2, 16, 32)


def test_chunked_blocks_match_flat():
    torch.manual_seed(0)
    flat = DinoVisionTransformer(
        img_size=56, patch_size=14, embed_dim=32, depth=4, num_heads=2, block_chunks=0
    ).eval()
    chunked = DinoVisionTransformer(
        img_size=56, patch_size=14, embed_dim=32, depth=4, num_heads=2, block_chunks=2
    ).eval()
    sd = flat.state_dict()
    remapped = {}
    for k, v in sd.items():
        if k.startswith("blocks."):
            i = int(k.split(".")[1])
            k = f"blocks.{i // 2}.{i}." + ".".join(k.split(".")[2:])
        remapped[k] = v
    chunked.load_state_dict(remapped, strict=True)
    x = torch.randn(1, 3, 56, 56)
    a = flat.get_intermediate_layers(x, n=[1, 3])
    b = chunked.get_intermediate_layers(x, n=[1, 3])
    for ta, tb in zip(a, b):
        assert torch.allclose(ta, tb, atol=1e-6)


def test_extractor_arbitrary_batch_dims_and_pixel_mask():
    model = _tiny()
    ext = StandaloneDinoExtractor(model, freeze=True)
    ext.train()
    assert not ext.model.training
    assert all(not p.requires_grad for p in ext.parameters())
    images = torch.randn(2, 3, 3, 28, 42)
    mask = torch.zeros(2, 3, 28, 42, dtype=torch.bool)
    mask[0, 0, 0, 0] = True
    feats = ext(images, mask)
    assert feats.shape == (2, 3, 2, 3, 32)
    plain = ext(images)
    assert not torch.allclose(feats[0, 0], plain[0, 0])
    assert torch.allclose(feats[1], plain[1])


def test_pretrained_loader_is_offline_until_called(monkeypatch):
    calls = {}

    def fake_load(url, **kwargs):
        calls["url"] = url
        return pretrained_mod.DINOv2("vits", num_register_tokens=4).state_dict()

    monkeypatch.setattr("torch.hub.load_state_dict_from_url", fake_load)
    model = pretrained_mod.load_pretrained_dinov2("vits", use_reg=True)
    assert model.embed_dim == pretrained_mod.EMBED_DIM["vits"]
    assert model.num_register_tokens == 4
    assert calls["url"].endswith("dinov2_vits14_reg4_pretrain.pth")
    with pytest.raises(ValueError, match="variant"):
        pretrained_mod.load_pretrained_dinov2("vitx")


def test_module_imports_without_network(monkeypatch):
    assert "ontic_nn.dinov2.pretrained" in sys.modules
    assert hasattr(pretrained_mod, "load_pretrained_dinov2")
