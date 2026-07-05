from __future__ import annotations

import math

import numpy as np


def psnr(a, b, *, max_val: float = 1.0) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)


class MetricsAccumulator:
    def __init__(self):
        self._sums: dict[str, float] = {}
        self._weights: dict[str, float] = {}

    def add(self, metrics: dict[str, float], n: int = 1) -> None:
        for key, value in metrics.items():
            self._sums[key] = self._sums.get(key, 0.0) + value * n
            self._weights[key] = self._weights.get(key, 0.0) + n

    def mean(self) -> dict[str, float]:
        return {key: self._sums[key] / self._weights[key] for key in self._sums}

    def reset(self) -> None:
        self._sums.clear()
        self._weights.clear()
