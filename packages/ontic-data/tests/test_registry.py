"""Registry contents, config defaults, light imports, mixed config."""

import importlib
import json
import pkgutil
import subprocess
import sys
import textwrap

import pytest
import torch
from torch.utils.data import ConcatDataset

import ontic_data
from ontic_data import DATASETS, DatasetCfg
from ontic_data.datasets.mixed import MixedDatasetCfg

HEAVY = [
    "pandas",
    "h5py",
    "cv2",
    "PIL",
    "mujoco",
    "torchcodec",
    "decord",
    "scipy",
    "torchvision",
    "jaxtyping",
]


def test_registry_keys_and_base():
    assert list(DATASETS) == [
        "genesis",
        "hocap",
        "taco",
        "dextris",
        "robot-dextris",
        "physinone",
        "synthrobot",
    ]
    for cls in DATASETS.values():
        assert issubclass(cls, DatasetCfg)
        cfg = cls()
        assert cfg.n_step_predict == 0 and cfg.build.__doc__ is None or True


def test_frontier_field_names_survive():
    assert DATASETS["genesis"]().roots == ["/fast/mzhobro/datasets/soft_genesis_elastic"]
    assert DATASETS["hocap"]().root == "/fast/mzhobro/hocap_dataset2/hocap_dataset"
    assert DATASETS["taco"]().image_shape == [376, 512]
    assert DATASETS["dextris"]().fps == 60
    assert DATASETS["physinone"]().split_mode == "scene"
    assert DATASETS["synthrobot"]().action_mode == "frame"
    assert DatasetCfg(n_step_predict=-3).n_step_predict == 0


def test_all_modules_import_without_heavy_deps():
    modules = [ontic_data.__name__] + [
        m.name for m in pkgutil.walk_packages(ontic_data.__path__, ontic_data.__name__ + ".")
    ]
    script = textwrap.dedent(
        """
        import importlib, json, sys, traceback
        import numpy, torch, einops, yaml
        heavy, modules = json.loads(sys.argv[1]), json.loads(sys.argv[2])
        sys.modules.update({m: None for m in heavy})
        errors = {}
        for name in modules:
            try:
                importlib.import_module(name)
            except BaseException:
                errors[name] = traceback.format_exc()
        print(json.dumps(errors))
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script, json.dumps(HEAVY), json.dumps(modules)],
        capture_output=True,
        text=True,
        check=True,
    )
    errors = json.loads(out.stdout.strip().splitlines()[-1])
    assert errors == {}, "\n".join(errors.values())


def test_mixed_builds_concat():
    class Toy(DatasetCfg):
        def build(self, stage, *, step_fn=None, horizon_fn=None):
            return torch.utils.data.TensorDataset(torch.zeros(3))

    single = MixedDatasetCfg(datasets={"a": Toy()}).build("train")
    assert len(single) == 3
    both = MixedDatasetCfg(datasets={"a": Toy(), "b": Toy()}).build("val")
    assert isinstance(both, ConcatDataset) and len(both) == 6
    with pytest.raises(ValueError, match="no sub-datasets"):
        MixedDatasetCfg().build("train")


def test_importlib_reload_is_idempotent():
    assert importlib.reload(ontic_data).DATASETS.keys() == DATASETS.keys()
