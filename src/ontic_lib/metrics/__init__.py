"""Evaluation metrics and aggregation.

Metric protocols live separately from geometry operations and losses. The
package-level exports preserve the original ``ontic_lib.metrics`` API.
"""

from .aggregate import MetricsAccumulator
from .image import psnr

__all__ = ["MetricsAccumulator", "psnr"]
