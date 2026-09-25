"""Context/target view sampling without replacement.

Semantics:
- ``num_context_views`` context views drawn uniformly at random (or fixed via
  ``context_views``).
- ``num_target_views`` target views drawn from the remaining cameras.
- ``num_target_views == -1`` -> use all remaining views as target.
- ``num_context_views == -1`` -> use all views as context; if ``num_target_views == -1``
  too, target = all views (identical set).
- ``target_includes_context=True`` -> concatenate context into target.

Randomness comes from an optional ``torch.Generator``; ``None`` uses torch's global RNG
(seed it per dataloader worker with :func:`ontic_data.collate.worker_init_fn`).
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Literal, Optional, Sequence

import torch
from torch import Tensor

Stage = Literal["train", "val", "test"]


@lru_cache(maxsize=32)
def build_camera_pairs(cam_names: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """Name-derived (context, target) camera pairs for paired sampling.

    Numbered rigs pair ``N-1`` with ``N-2``; rigs without numbered pairs fall back to
    consecutive rings ``ring_2k -> ring_2k+1``. Indices are positions in ``cam_names``.
    """
    ix = {n: i for i, n in enumerate(cam_names)}
    pairs = []
    for name in cam_names:
        m = re.fullmatch(r"(\d+)-1", name)
        if m and f"{m.group(1)}-2" in ix:
            pairs.append((ix[name], ix[f"{m.group(1)}-2"]))
    if pairs:
        return tuple(pairs)
    rings = sorted(n for n in cam_names if n.startswith("ring_"))
    return tuple((ix[rings[k]], ix[rings[k + 1]]) for k in range(0, len(rings) - 1, 2))


@dataclass(kw_only=True)
class ViewSamplerCfg:
    num_context_views: int = 3
    num_target_views: int = 2
    target_includes_context: bool = False  # if True, context views are added to target
    context_views: Optional[list] = None  # fixed indices, or a list of index groups

    # Name-paired mode (see build_camera_pairs). Each draw samples
    # ``num_context_views - num_extra_context`` pairs (context takes the first camera of
    # each pair, target the second) plus distinct extras from ``pair_extra_pool`` (when
    # the pool runs dry, any still-unused camera). Needs ``cam_names`` in ``sample``.
    paired_views: bool = False
    pair_extra_pool: list[str] = field(
        default_factory=lambda: ["wrist_camera_r", "wrist_camera_l", "scene_camera"]
    )
    num_extra_context: int = 1
    num_extra_target: int = 1

    def build(self, stage: Stage, generator: torch.Generator | None = None) -> ViewSampler:
        return ViewSampler(self, stage, generator)


class ViewSampler:
    """Samples sorted context / target camera indices for one timestep."""

    def __init__(
        self,
        cfg: ViewSamplerCfg,
        stage: Stage,
        generator: torch.Generator | None = None,
    ) -> None:
        self.cfg = cfg
        self.stage = stage
        self.generator = generator
        self.context_views = copy.deepcopy(cfg.context_views)
        self.updated = False

    def sample(
        self,
        scene: str,
        extrinsics: Tensor,
        intrinsics: Tensor,
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Sorted ``(context (Vc,), target (Vt,))`` int64 indices into the ``V`` cameras."""
        index_context, index_target = self._sample(
            scene, extrinsics, intrinsics, device=device, **kwargs
        )
        if self.cfg.target_includes_context:
            index_target = torch.cat([index_context, index_target])
        return index_context.sort().values, index_target.sort().values

    def update_camera_indices(self, ixs: list[int] | None) -> None:
        """Remap configured (fixed) views after ``camera_ixs_allowed`` filtering."""
        if ixs is None or self.updated:
            return
        if isinstance(self.context_views, list):
            if isinstance(self.context_views[0], list):
                self.context_views = [[ixs.index(i) for i in g] for g in self.context_views]
            else:
                self.context_views = [ixs.index(i) for i in self.context_views]
        self.updated = True

    def _randperm(self, n: int, device: torch.device) -> Tensor:
        return torch.randperm(n, generator=self.generator, device=device)

    def _sample(
        self,
        scene: str,
        extrinsics: Tensor,
        intrinsics: Tensor,
        camera_group_ix: int | None = None,
        cam_names: Sequence[str] | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[Tensor, Tensor]:
        num_views = extrinsics.shape[0]

        if self.cfg.paired_views:
            if cam_names is None:
                raise ValueError("paired_views needs cam_names from the dataset")
            return self._sample_paired(list(cam_names), device)

        if self.num_context_views == -1 and self.num_target_views == -1:
            all_frames = torch.arange(num_views, device=device)
            return all_frames, all_frames

        if self.num_context_views == -1:
            raise ValueError("num_context_views can only be -1 if num_target_views is -1 too")
        num_total = self.num_context_views + self.num_target_views
        if num_total > num_views:
            raise ValueError(f"need {num_total} views but the scene has {num_views}")

        # 1. Context views.
        context_views = self.context_views
        rand_order = self._randperm(num_views, device)
        index_context = rand_order[: self.num_context_views]

        if context_views is not None:
            if isinstance(context_views[0], list):
                if camera_group_ix is None:
                    pick = torch.randint(len(context_views), (1,), generator=self.generator)
                    context_views = context_views[int(pick)]
                else:
                    context_views = context_views[camera_group_ix % len(context_views)]
            if len(context_views) < self.num_context_views:
                raise ValueError("fewer fixed context_views than num_context_views")
            index_context = torch.tensor(context_views, dtype=torch.int64, device=device)
            if len(context_views) > self.num_context_views:
                rand_order = self._randperm(len(index_context), device)
                index_context = index_context[rand_order[: self.num_context_views]]

        # 2. Target views from the rest.
        if self.num_target_views == -1:
            num_targets = num_views - self.num_context_views
        else:
            num_targets = self.num_target_views
        chosen = set(index_context.tolist())
        rest_views = torch.tensor(
            [i for i in range(num_views) if i not in chosen], dtype=torch.int64, device=device
        )
        rand_order = self._randperm(len(rest_views), device)
        index_target = rest_views[rand_order[:num_targets]].detach().clone()
        return index_context, index_target

    def _sample_paired(self, cam_names: list[str], device: torch.device) -> tuple[Tensor, Tensor]:
        cfg = self.cfg
        n_pairs = cfg.num_context_views - cfg.num_extra_context
        if cfg.num_target_views != n_pairs + cfg.num_extra_target:
            raise ValueError(
                f"paired_views: num_target_views ({cfg.num_target_views}) must be "
                f"pairs ({n_pairs}) + num_extra_target ({cfg.num_extra_target})"
            )
        pairs = build_camera_pairs(tuple(cam_names))
        if len(pairs) < n_pairs:
            raise ValueError(f"rig {cam_names} has {len(pairs)} pairs, need {n_pairs}")

        chosen = [
            pairs[i] for i in self._randperm(len(pairs), torch.device("cpu"))[:n_pairs].tolist()
        ]
        context = [c for c, _ in chosen]
        target = [t for _, t in chosen]

        # Extras: the wrist/scene pool first (shuffled), then any unused camera.
        used = set(context) | set(target)
        pool = [i for i, n in enumerate(cam_names) if n in cfg.pair_extra_pool and i not in used]
        spare = [i for i in range(len(cam_names)) if i not in used and i not in pool]
        cpu = torch.device("cpu")
        avail = [pool[i] for i in self._randperm(len(pool), cpu).tolist()]
        avail += [spare[i] for i in self._randperm(len(spare), cpu).tolist()]
        need = cfg.num_extra_context + cfg.num_extra_target
        if len(avail) < need:
            raise ValueError(f"rig {cam_names}: not enough cameras for {need} extras")
        context += avail[: cfg.num_extra_context]
        target += avail[cfg.num_extra_context : need]
        return (
            torch.tensor(context, dtype=torch.int64, device=device),
            torch.tensor(target, dtype=torch.int64, device=device),
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
