from __future__ import annotations

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
