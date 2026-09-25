"""End-to-end tests of PointTransformerV3 on tiny configurations (CPU)."""

import pytest
import torch

from ontic_lib.structures import PointBatch
from ontic_nn.ptv3 import (
    AttentionCfg,
    PointTransformerV3,
    PoolingCfg,
    StageCfg,
    TemporalCfg,
    load_fwomo_state_dict,
    presets,
)


def tiny_cfg(**overrides):
    values = dict(
        encoder=(
            StageCfg(1, 12, 2, patch_size=8),
            StageCfg(1, 24, 2, patch_size=8),
            StageCfg(1, 24, 2, patch_size=8),
        ),
        decoder=(StageCfg(1, 12, 2, patch_size=8), StageCfg(1, 24, 2, patch_size=8)),
        pooling=PoolingCfg(strides=(2, 2)),
        temporal=TemporalCfg(merge_window=(1, 2)),
        drop_path=0.0,
    )
    values.update(overrides)
    return presets.base(**values)


def random_points(seed=0, temporal=True):
    g = torch.Generator().manual_seed(seed)
    shape = (2, 4, 40) if temporal else (2, 60)
    coord = torch.rand(*shape, 3, generator=g)
    feat = torch.randn(*shape, 6, generator=g)
    mask = torch.rand(shape, generator=g) > 0.2
    extras = {"flag": torch.rand(shape, generator=g) > 0.5}
    return PointBatch.from_padded(coord, feat, mask, extras=extras)


@pytest.mark.parametrize("temporal", [True, False])
def test_forward_backward_keeps_order(temporal):
    cfg = tiny_cfg() if temporal else tiny_cfg(temporal=None)
    model = PointTransformerV3(cfg)
    points = random_points(temporal=temporal)
    out = model(points, generator=torch.Generator().manual_seed(0))
    assert out.feat.shape == (len(points), cfg.out_channels)
    assert torch.equal(out.coord, points.coord)
    assert torch.equal(out.batch, points.batch)
    assert "flag" in out.extras
    out.feat.sum().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_temporal_model_requires_time():
    model = PointTransformerV3(tiny_cfg())
    with pytest.raises(ValueError, match="time"):
        model(random_points(temporal=False))


def test_feature_pyramid_and_cls_mode():
    points = random_points()
    out = PointTransformerV3(tiny_cfg(feature_pyramid=True))(points)
    assert out.feat.shape == (len(points), 3)
    coarse = PointTransformerV3(tiny_cfg(cls_mode=True, decoder=()))(points)
    assert coarse.feat.shape[-1] == 24
    assert len(coarse) < len(points)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(pooling=PoolingCfg(kind="grid", strides=(2, 2))),
        dict(pooling=PoolingCfg(strides=(2, 2), neighbor_attender=True, norm="bn")),
        dict(attention=AttentionCfg(rpe="v2")),
        dict(attention=AttentionCfg(rpe="v1", upcast_attention=True, upcast_softmax=True)),
        dict(attention=AttentionCfg(rope=True, rope_coord_source="grid_coord")),
        dict(
            encoder=(
                StageCfg(1, 12, 2, 8, conv=True, attn=False),
                StageCfg(1, 24, 2, 8),
                StageCfg(1, 24, 2, 8, conv=False, attn=True),
            )
        ),
    ],
)
def test_variants_run(overrides):
    model = PointTransformerV3(tiny_cfg(**overrides))
    out = model(random_points())
    assert out.feat.shape[-1] == 3 and torch.isfinite(out.feat).all()


def test_gradient_checkpointing_matches():
    points = random_points()
    model = PointTransformerV3(tiny_cfg(gradient_checkpointing=True)).train()
    reference = PointTransformerV3(tiny_cfg()).train()
    reference.load_state_dict(model.state_dict())
    g = torch.Generator().manual_seed(3)
    a = model(points, generator=g)
    b = reference(points, generator=torch.Generator().manual_seed(3))
    assert torch.allclose(a.feat, b.feat, atol=1e-6)
    a.feat.sum().backward()
    b.feat.sum().backward()
    for (name, p), (_, q) in zip(model.named_parameters(), reference.named_parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-5), name


def test_validate_rejects_bad_lengths():
    with pytest.raises(ValueError):
        PointTransformerV3(tiny_cfg(pooling=PoolingCfg(strides=(2,))))
    with pytest.raises(ValueError):
        PointTransformerV3(tiny_cfg(temporal=TemporalCfg(merge_window=(2,))))
    with pytest.raises(ValueError):
        PointTransformerV3(tiny_cfg(decoder=(StageCfg(1, 12, 2, 8),)))
    with pytest.raises(ValueError):
        PointTransformerV3(tiny_cfg(attention=AttentionCfg(backend="flash", rpe="v2")))


def test_presets_shapes():
    for preset in (presets.base, presets.medium, presets.large, presets.fwomo_legacy):
        cfg = preset()
        cfg.validate()
        assert len(cfg.encoder) == 5 and len(cfg.decoder) == 4
    assert presets.fwomo_legacy().conv_impl == "spconv"
    assert presets.fwomo_legacy().attention.backend == "flash"
    assert presets.base().encoder[3].depth == 6 and presets.large().encoder[3].depth == 12


def test_load_fwomo_state_dict_renames():
    model = PointTransformerV3(tiny_cfg())
    sd = model.state_dict()
    legacy = {}
    for key, value in sd.items():
        k = key.replace(".cpe.conv.", ".cpe.0.").replace(".cpe.linear.", ".cpe.1.")
        k = k.replace(".cpe.norm.", ".cpe.2.").replace(".norm1.", ".norm1.0.")
        k = k.replace(".norm2.", ".norm2.0.").replace(".mlp.", ".mlp.0.")
        k = k.replace(".down.norm.", ".down.norm.0.")
        k = k.replace(".up.proj.linear.", ".up.proj.0.").replace(".up.proj.norm.", ".up.proj.1.")
        k = k.replace(".up.proj_skip.linear.", ".up.proj_skip.0.")
        k = k.replace(".up.proj_skip.norm.", ".up.proj_skip.1.")
        k = k.replace("dec.head.", "dec.2.")
        if k.endswith("cpe.0.weight") or k.endswith("stem.conv.weight"):
            volume, cin, cout = value.shape
            ks = round(volume ** (1 / 3))
            value = value.permute(2, 0, 1).reshape(cout, ks, ks, ks, cin)
        legacy[k] = value
    assert set(legacy) != set(sd)
    fresh = PointTransformerV3(tiny_cfg())
    load_fwomo_state_dict(fresh, legacy, strict=True)
    for key in sd:
        assert torch.equal(fresh.state_dict()[key], sd[key]), key
