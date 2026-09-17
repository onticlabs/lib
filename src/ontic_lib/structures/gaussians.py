"""Batched container for anisotropic 3D Gaussians."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from ..splats.gaussians import covariance_from_scale_rotation_representation

# Trailing (per-Gaussian) shape of every tensor field; ``None`` entries are free.
_TRAILING_SHAPES: dict[str, tuple[int | None, ...]] = {
    "means": (3,),
    "scales": (3,),
    "rotations": (4,),
    "opacities": (1,),
    "harmonics": (3, None),
    "covariances": (3, 3),
    "mask": (1,),
}


@dataclass
class Gaussians:
    """Gaussians with arbitrary leading batch dims ``*batch`` and ``N`` Gaussians each.

    Shapes: ``means (*batch, N, 3)``, ``scales (*batch, N, 3)`` (post-activation, positive),
    ``rotations (*batch, N, 4)`` real-first ``wxyz`` quaternions, ``opacities (*batch, N, 1)``,
    ``harmonics (*batch, N, 3, d_sh)``, ``covariances (*batch, N, 3, 3)`` (explicit input only,
    never derived or cached), ``mask (*batch, N, 1)`` bool, ``extras`` ``(*batch, N, ...)``.
    Every tensor shares ``(*batch, N)``; ``__post_init__`` validates this.
    """

    means: Tensor
    scales: Tensor | None = None
    rotations: Tensor | None = None
    opacities: Tensor | None = None
    harmonics: Tensor | None = None
    covariances: Tensor | None = None
    mask: Tensor | None = None
    extras: dict[str, Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.means.ndim < 2 or self.means.shape[-1] != 3:
            raise ValueError(f"means must have shape (*batch, N, 3), got {tuple(self.means.shape)}")
        lead = self.means.shape[:-1]
        for name, trailing in _TRAILING_SHAPES.items():
            value = getattr(self, name)
            if value is None:
                continue
            expected = tuple(lead) + trailing
            actual = tuple(value.shape)
            ok = len(actual) == len(expected) and all(
                e is None or a == e for a, e in zip(actual, expected)
            )
            if not ok:
                shown = tuple("?" if e is None else e for e in expected)
                raise ValueError(f"{name} must have shape {shown}, got {actual}")
        if self.mask is not None and self.mask.dtype != torch.bool:
            raise ValueError(f"mask must be bool, got {self.mask.dtype}")
        for name, value in self.extras.items():
            if name in _TRAILING_SHAPES:
                raise ValueError(f"extras key {name!r} collides with a Gaussians field")
            if tuple(value.shape[: len(lead)]) != tuple(lead):
                raise ValueError(
                    f"extras[{name!r}] must have leading shape {tuple(lead)}, "
                    f"got {tuple(value.shape)}"
                )

    # ------------------------------------------------------------------ properties
    @property
    def batch_shape(self) -> tuple[int, ...]:
        return tuple(self.means.shape[:-2])

    @property
    def num_gaussians(self) -> int:
        return int(self.means.shape[-2])

    @property
    def device(self) -> torch.device:
        return self.means.device

    @property
    def dtype(self) -> torch.dtype:
        return self.means.dtype

    # ------------------------------------------------------------------ access
    def items(self) -> Iterator[tuple[str, Tensor]]:
        """Yield ``(name, tensor)`` for every non-``None`` field, then every extra."""
        for f in dataclasses.fields(self):
            if f.name == "extras":
                continue
            value = getattr(self, f.name)
            if value is not None:
                yield f.name, value
        yield from self.extras.items()

    def covariance(self) -> Tensor:
        """Return ``(*batch, N, 3, 3)``: ``covariances`` if given, else ``R diag(s^2) R^T``.

        Quaternions are normalized (``eps=1e-8``) before conversion.
        """
        if self.covariances is not None:
            return self.covariances
        if self.scales is None or self.rotations is None:
            raise ValueError("covariance() needs scales and rotations (or explicit covariances)")
        return covariance_from_scale_rotation_representation(
            self.scales, self.rotations, quaternion_order="wxyz", normalize=True, eps=1e-8
        )

    # ------------------------------------------------------------------ transforms
    def _map(self, fn: Callable[[Tensor], Tensor]) -> Gaussians:
        kwargs: dict[str, Any] = {name: fn(value) for name, value in self.items()}
        extras = {name: kwargs.pop(name) for name in self.extras}
        return Gaussians(**kwargs, extras=extras)

    def __getitem__(self, index: Any) -> Gaussians:
        """Index every field along the batch dims. Raises on unbatched Gaussians."""
        batch_dims = len(self.batch_shape)
        if batch_dims == 0:
            raise IndexError("cannot index unbatched Gaussians (no batch dims)")
        if isinstance(index, tuple) and len(index) > batch_dims:
            raise IndexError(
                f"index has {len(index)} entries but there are {batch_dims} batch dims"
            )
        return self._map(lambda v: v[index])

    def flatten_batch(self) -> Gaussians:
        """Merge all batch dims into one: ``(*batch, N, ...) -> (prod(batch), N, ...)``.

        Unbatched Gaussians gain a batch dim of size 1.
        """
        batch_dims = len(self.batch_shape)
        if batch_dims == 1:
            return self
        return self._map(lambda v: v.reshape(-1, *v.shape[batch_dims:]))

    def unflatten_batch(self, shape: tuple[int, ...]) -> Gaussians:
        """Split the single batch dim into ``shape``: ``(B, N, ...) -> (*shape, N, ...)``."""
        if len(self.batch_shape) != 1:
            raise ValueError(f"unflatten_batch needs exactly one batch dim, got {self.batch_shape}")
        shape = tuple(shape)
        return self._map(lambda v: v.reshape(*shape, *v.shape[1:]))

    def to(self, *args: Any, **kwargs: Any) -> Gaussians:
        """``Tensor.to`` on every field; non-floating tensors (``mask``) keep their dtype."""

        def convert(v: Tensor) -> Tensor:
            out = v.to(*args, **kwargs)
            return out if v.is_floating_point() else out.to(v.dtype)

        return self._map(convert)

    def float(self) -> Gaussians:
        return self._map(lambda v: v.float() if v.is_floating_point() else v)

    def detach(self) -> Gaussians:
        return self._map(lambda v: v.detach())

    def clone(self) -> Gaussians:
        return self._map(lambda v: v.clone())

    def replace(self, **changes: Any) -> Gaussians:
        """Return a copy with the given fields replaced (re-validated)."""
        return dataclasses.replace(self, **changes)

    def __repr__(self) -> str:
        parts = [f"{name}={tuple(value.shape)}" for name, value in self.items()]
        return f"{type(self).__name__}({', '.join(parts)})"
