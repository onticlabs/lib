"""Optional accelerators raise informative ImportErrors when absent."""

import builtins
import importlib

import pytest

from ontic_nn.ptv3 import AttentionCfg, PointTransformerV3, PoolingCfg, StageCfg, presets


def _has(name):
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def _blocked(monkeypatch, *names):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in names:
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def _cfg(**overrides):
    values = dict(
        encoder=(StageCfg(1, 12, 2, 8), StageCfg(1, 24, 2, 8)),
        decoder=(StageCfg(1, 12, 2, 8),),
        pooling=PoolingCfg(strides=(2,)),
        temporal=None,
    )
    values.update(overrides)
    return presets.base(**values)


def test_spconv_missing_names_extra(monkeypatch):
    _blocked(monkeypatch, "spconv")
    with pytest.raises(ImportError, match=r"ontic-nn\[spconv\]"):
        PointTransformerV3(_cfg(conv_impl="spconv"))


def test_flash_missing_names_package(monkeypatch):
    _blocked(monkeypatch, "flash_attn")
    with pytest.raises(ImportError, match="flash-attn"):
        PointTransformerV3(_cfg(attention=AttentionCfg(backend="flash")))


def test_rope_cuda_missing_names_script(monkeypatch):
    _blocked(monkeypatch, "point_rope_cuda")
    with pytest.raises(ImportError, match="install_cuda_ext.sh point_rope"):
        PointTransformerV3(_cfg(attention=AttentionCfg(rope=True, rope_impl="cuda")))


@pytest.mark.skipif(not _has("spconv"), reason="spconv not installed")
def test_spconv_constructs():
    model = PointTransformerV3(_cfg(conv_impl="spconv"))
    assert model.embedding.stem.conv.impl == "spconv"
