"""Loose scalar / vector / pair inputs and the tensors they are sanitized into."""

from __future__ import annotations

from typing import Iterable, Union

import torch
from einops import repeat
from torch import Tensor

Real = Union[float, int]

#: A real, an iterable of reals, or a ``(3,)`` / ``(batch, 3)`` / ``(2,)`` / ``(batch, 2)`` tensor.
Vector = Union[Real, Iterable[Real], Tensor]


def sanitize_vector(vector: Vector, dim: int, device: torch.device) -> Tensor:
    """``vector`` as a float32 ``(#batch, dim)`` tensor (a scalar is repeated along ``dim``)."""
    if isinstance(vector, Tensor):
        vector = vector.type(torch.float32).to(device)
    else:
        vector = torch.tensor(vector, dtype=torch.float32, device=device)
    while vector.ndim < 2:
        vector = vector[None]
    if vector.shape[-1] == 1:
        vector = repeat(vector, "... () -> ... c", c=dim)
    assert vector.shape[-1] == dim
    assert vector.ndim == 2
    return vector


#: A real, an iterable of reals, or a ``()`` / ``(batch,)`` tensor.
Scalar = Union[Real, Iterable[Real], Tensor]


def sanitize_scalar(scalar: Scalar, device: torch.device) -> Tensor:
    """``scalar`` as a float32 ``(#batch,)`` tensor."""
    if isinstance(scalar, Tensor):
        scalar = scalar.type(torch.float32).to(device)
    else:
        scalar = torch.tensor(scalar, dtype=torch.float32, device=device)
    while scalar.ndim < 1:
        scalar = scalar[None]
    assert scalar.ndim == 1
    return scalar


#: An iterable of two reals or a ``(2,)`` tensor.
Pair = Union[Iterable[Real], Tensor]


def sanitize_pair(pair: Pair, device: torch.device) -> Tensor:
    """``pair`` as a float32 ``(2,)`` tensor."""
    if isinstance(pair, Tensor):
        pair = pair.type(torch.float32).to(device)
    else:
        pair = torch.tensor(pair, dtype=torch.float32, device=device)
    assert pair.shape == (2,)
    return pair
