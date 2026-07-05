from __future__ import annotations

import math

import numpy as np
import pytest

from ontic_lib.metrics import MetricsAccumulator, psnr


def test_psnr_of_identical_arrays_is_inf():
    a = np.array([0.1, 0.5, 0.9])

    assert psnr(a, a) == float("inf")


def test_psnr_matches_closed_form_on_known_example():
    a = np.array([0.0, 0.0, 0.0, 0.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    # mse = 1.0, max_val = 1.0 -> 20*log10(1) - 10*log10(1) = 0
    assert psnr(a, b, max_val=1.0) == pytest.approx(0.0, abs=1e-9)

    c = np.array([0.0, 0.0])
    d = np.array([0.5, 0.5])
    # mse = 0.25 -> 20*log10(1) - 10*log10(0.25) = 6.0206 dB
    expected = 20 * math.log10(1.0) - 10 * math.log10(0.25)
    assert psnr(c, d, max_val=1.0) == pytest.approx(expected, abs=1e-9)


def test_psnr_shape_mismatch_raises_value_error():
    a = np.zeros((2, 2))
    b = np.zeros((3,))

    with pytest.raises(ValueError):
        psnr(a, b)


def test_psnr_accepts_array_likes():
    assert psnr([1, 2, 3], [1, 2, 3]) == float("inf")


def test_accumulator_weighted_mean_across_multiple_adds():
    acc = MetricsAccumulator()
    acc.add({"loss": 1.0}, n=1)
    acc.add({"loss": 3.0}, n=3)
    # weighted mean = (1*1 + 3*3) / (1+3) = 10/4 = 2.5
    assert acc.mean()["loss"] == pytest.approx(2.5)


def test_accumulator_default_n_is_one():
    acc = MetricsAccumulator()
    acc.add({"loss": 2.0})
    acc.add({"loss": 4.0})

    assert acc.mean()["loss"] == pytest.approx(3.0)


def test_accumulator_handles_missing_keys_across_adds():
    acc = MetricsAccumulator()
    acc.add({"loss": 1.0, "psnr": 30.0}, n=2)
    acc.add({"loss": 3.0}, n=2)  # psnr missing this time

    result = acc.mean()

    # loss: (1*2 + 3*2) / (2+2) = 8/4 = 2.0
    assert result["loss"] == pytest.approx(2.0)
    # psnr: only present in the first add -> (30*2)/2 = 30.0
    assert result["psnr"] == pytest.approx(30.0)


def test_accumulator_mean_on_empty_is_empty_dict():
    acc = MetricsAccumulator()

    assert acc.mean() == {}


def test_accumulator_reset_clears_state():
    acc = MetricsAccumulator()
    acc.add({"loss": 1.0}, n=5)
    acc.reset()

    assert acc.mean() == {}

    acc.add({"loss": 10.0}, n=1)
    assert acc.mean()["loss"] == pytest.approx(10.0)
