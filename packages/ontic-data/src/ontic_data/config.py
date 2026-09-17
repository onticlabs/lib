"""Stage-independent dataset configuration shared by every dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from torch.utils.data import Dataset

from .view_sampler import ViewSamplerCfg

Stage = Literal["train", "val", "test"]

#: ``() -> global training step``. ``None`` means step 0.
StepFn = Callable[[], int]
#: ``(global_step, n_pred_full) -> n_pred`` prediction steps worth decoding at that step
#: (the training loop's horizon schedule). ``None`` decodes the full horizon.
HorizonFn = Callable[[int, int], int]


@dataclass(kw_only=True)
class DatasetCfg:
    """Base dataset config; subclasses add their roots and implement :meth:`build`.

    ``build(stage, *, step_fn=None, horizon_fn=None)`` replaces the frontier
    ``build(stage, step_tracker)``: the step counter and the horizon curriculum stay
    with the trainer, which passes them in as callables. Datasets only ever read
    ``step_fn()`` (an int) and ``horizon_fn(step, n_pred_full)`` (an int).
    """

    # Loading + render/photometric-loss resolution (H, W). Backbones resize internally.
    image_shape: list[int] = field(default_factory=lambda: [128, 128])
    # World-space workspace AABB (metres), emitted as ``workspace_min``/``workspace_max``.
    workspace_min: list[float] | None = None
    workspace_max: list[float] | None = None
    view_sampler: ViewSamplerCfg = field(default_factory=ViewSamplerCfg)
    n_step_state: int = 1
    n_step_predict: int = 0
    val_n_step_predict: int = 100  # validation horizon (only when n_step_predict > 0)
    near: float = 0.1
    far: float = 1000.0
    augment: bool = False
    speedup: int = 1
    camera_ixs_allowed: Optional[list[int]] = None
    consistent_cameras: bool = False  # same camera indices for all timesteps
    n_step_state_predict: int = 1  # dynamic-model steps per rollout iteration
    fps: int = 30  # native FPS (used by MixedDatasetCfg to normalise temporal resolution)

    # Horizon-aware loading (train stage only): decode only
    # ``n_step_state + horizon_fn(step_fn() + margin, n_pred_full)`` frames. The view
    # sampler still runs for every timestep of the full horizon so batches stay bit-exact
    # with full-horizon loading. The margin covers dataloader prefetch lag.
    horizon_aware_loading: bool = False
    horizon_load_margin_steps: int = 256

    def __post_init__(self) -> None:
        self.n_step_predict = max(0, self.n_step_predict)

    def build_view_sampler(self, stage: Stage, generator=None):
        return self.view_sampler.build(stage, generator)

    def build(
        self,
        stage: Stage,
        *,
        step_fn: StepFn | None = None,
        horizon_fn: HorizonFn | None = None,
    ) -> Dataset:
        raise NotImplementedError
