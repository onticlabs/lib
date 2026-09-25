"""ontic_data: dataset loaders for temporal multi-view scenes.

Every module imports with torch + numpy + einops + pyyaml alone; format-specific readers
(video decoders, h5py, pandas, PIL, cv2, mujoco) are imported lazily and raise an
``ImportError`` naming the extra to install. ``DATASETS`` maps registry keys to config
classes; ``cfg.build(stage, step_fn=..., horizon_fn=...)`` returns the dataset.
"""

from .config import DatasetCfg, HorizonFn, StepFn
from .datasets.dextris import DextrisDatasetCfg, RobotDextrisDatasetCfg
from .datasets.genesis import DatasetGenesisCfg
from .datasets.hocap import HocapDatasetCfg
from .datasets.physinone import PhysInOneDatasetCfg
from .datasets.synthrobot import SynthRobotDatasetCfg
from .datasets.taco import TacoDatasetCfg

__version__ = "0.5.0"

DATASETS: dict[str, type[DatasetCfg]] = {
    "genesis": DatasetGenesisCfg,
    "hocap": HocapDatasetCfg,
    "taco": TacoDatasetCfg,
    "dextris": DextrisDatasetCfg,
    "robot-dextris": RobotDextrisDatasetCfg,
    "physinone": PhysInOneDatasetCfg,
    "synthrobot": SynthRobotDatasetCfg,
}

__all__ = ["DATASETS", "DatasetCfg", "HorizonFn", "StepFn", "__version__"]
