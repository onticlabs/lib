"""Unit tests for ontic_lib.distributed."""

import torch

from ontic_lib.distributed import _classify, _is_distributed, avg_log_dict_across_ranks


def test_not_distributed_in_test_env():
    # These tests run single-process, so the helper must report non-distributed.
    assert _is_distributed() is False


def test_passthrough_returns_new_dict_with_same_values():
    log_dict = {
        "loss": 1.5,
        "step": 3,
        "grad_norm": torch.tensor([2.0, 4.0]),
        "name": "train",
        "flag": True,
        "image": object(),
    }
    original = dict(log_dict)
    out = avg_log_dict_across_ranks(log_dict)

    # A new dict object, not the same instance...
    assert out is not log_dict
    # ...but the incoming dict is not mutated.
    assert log_dict == original
    assert log_dict["grad_norm"] is original["grad_norm"]

    # Same keys and identical (unreduced) values on a single-rank run.
    assert out.keys() == log_dict.keys()
    assert out["loss"] == 1.5
    assert out["step"] == 3
    assert out["name"] == "train"
    assert out["flag"] is True
    assert out["image"] is log_dict["image"]
    assert torch.equal(out["grad_norm"], log_dict["grad_norm"])


def test_classify_splits_reducible_from_passthrough():
    log_dict = {
        "scalar_int": 2,
        "scalar_float": 0.5,
        "float_tensor": torch.randn(3, 4),
        "int_tensor": torch.tensor([1, 2, 3]),
        "bool_flag": True,
        "text": "hello",
        "nothing": None,
    }
    scalars, tensor_shapes = _classify(log_dict)
    assert scalars == {"scalar_int", "scalar_float"}
    assert tensor_shapes == {"float_tensor": (3, 4)}


def test_passthrough_preserves_empty_dict():
    out = avg_log_dict_across_ranks({})
    assert out == {}
