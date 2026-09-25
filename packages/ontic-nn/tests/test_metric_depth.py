"""Contract tests for ontic_nn.metric_depth (no research packages, no weights)."""

from __future__ import annotations

import os
import sys

import pytest
import torch

from ontic_nn.metric_depth import (
    METRIC_MODELS,
    Metric3DMetricConfig,
    Metric3DMetricModel,
    MetricDepthConfig,
    MetricDepthModel,
    MetricDepthOutput,
    offline_guard,
)

RESEARCH_MODULES = {"da3": "depth_anything_3", "depthpro": "depth_pro", "unidepth": "unidepth"}


def test_registry_keys():
    assert set(METRIC_MODELS) == {"da3", "depthpro", "metric3d", "unidepth"}
    for cls in METRIC_MODELS.values():
        assert issubclass(cls, MetricDepthConfig)


@pytest.mark.parametrize("name", sorted(METRIC_MODELS))
def test_config_defaults(name):
    cfg = METRIC_MODELS[name]()
    assert cfg.checkpoint_path is None and cfg.cache_dir is None and cfg.allow_download
    assert cfg.long_side > 0


@pytest.mark.parametrize("name", sorted(RESEARCH_MODULES))
def test_build_without_research_repo_names_extra(name, monkeypatch):
    monkeypatch.setitem(sys.modules, RESEARCH_MODULES[name], None)
    with pytest.raises(ImportError, match=rf"ontic-nn\[{name}\]"):
        METRIC_MODELS[name]().build()


def test_metric3d_offline_cache_miss_raises(tmp_path):
    import torch.hub

    hub_dir = torch.hub.get_dir()
    try:
        cfg = Metric3DMetricConfig(cache_dir=str(tmp_path), allow_download=False)
        with pytest.raises(RuntimeError, match="allow_download"):
            cfg.build()
        assert torch.hub.get_dir() == str(tmp_path / "hub")
    finally:
        torch.hub.set_dir(hub_dir)
    with pytest.raises(ValueError, match="variant"):
        Metric3DMetricConfig(variant="vit_huge").build()


def test_requires_intrinsics_flags():
    assert Metric3DMetricModel.REQUIRES_INTRINSICS is True
    assert MetricDepthModel.REQUIRES_INTRINSICS is False
    for name, cfg_cls in METRIC_MODELS.items():
        assert cfg_cls.__module__.endswith(name)


def test_base_config_and_output():
    with pytest.raises(NotImplementedError):
        MetricDepthConfig().build()
    out = MetricDepthOutput(depth=torch.ones(1, 1, 4, 4))
    assert out.conf is None and out.intrinsics is None


def test_offline_guard_restores_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "0")
    with offline_guard(False):
        assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["HF_DATASETS_OFFLINE"] == "0"
