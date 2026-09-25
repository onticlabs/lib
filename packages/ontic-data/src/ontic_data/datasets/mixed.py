"""Train on several datasets at once via ``ConcatDataset``."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch.utils.data import ConcatDataset, Dataset

from ..config import DatasetCfg, HorizonFn, Stage, StepFn


@dataclass(kw_only=True)
class MixedDatasetCfg(DatasetCfg):
    """Sub-datasets keyed by a user label; nothing here mutates the sub-configs, so
    shared fields (``image_shape``, ``n_step_*``, view sampler counts) must be set on
    each of them by the caller."""

    datasets: dict[str, DatasetCfg] = field(default_factory=dict)

    def build(
        self,
        stage: Stage,
        *,
        step_fn: StepFn | None = None,
        horizon_fn: HorizonFn | None = None,
    ) -> Dataset:
        built = [
            sub.build(stage, step_fn=step_fn, horizon_fn=horizon_fn)
            for sub in self.datasets.values()
        ]
        if not built:
            raise ValueError("MixedDatasetCfg: no sub-datasets configured")
        if len(built) == 1:
            return built[0]
        return ConcatDataset(built)
