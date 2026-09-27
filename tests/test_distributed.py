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


# --- two processes over gloo on the CPU: the collectives themselves ---------------------

import datetime  # noqa: E402
import os  # noqa: E402
import socket  # noqa: E402

import torch.multiprocessing as mp  # noqa: E402

from ontic_lib import distributed  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, world: int, port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank)
    )
    assert distributed.setup(backend="gloo", timeout=datetime.timedelta(seconds=120)) == (rank, world)
    assert distributed.is_distributed()
    assert distributed.is_main() == (rank == 0)
    assert (distributed.rank(), distributed.world_size()) == (rank, world)
    assert distributed.dist_device().type == "cpu"

    torch.manual_seed(0)
    net = torch.nn.Linear(4, 2)
    torch.manual_seed(0)
    reference = torch.nn.Linear(4, 2)  # what rank 0 holds
    if rank != 0:
        with torch.no_grad():
            net.weight.add_(1.0)
    distributed.broadcast_module(net)
    torch.testing.assert_close(net.weight, reference.weight)

    net.weight.grad = torch.full_like(net.weight, float(rank + 1))  # ranks 1, 2, ... -> mean 1.5 for two
    net.bias.grad = torch.ones_like(net.bias) if rank == 0 else None  # a missing grad counts as zeros
    distributed.sync_gradients(net)
    torch.testing.assert_close(net.weight.grad, torch.full_like(net.weight, (world + 1) / 2))
    torch.testing.assert_close(net.bias.grad, torch.full_like(net.bias, 1.0 / world))

    assert distributed.broadcast_flag(rank == 0) is True
    assert distributed.broadcast_flag(rank != 0) is False

    log = {"loss": float(rank), "t": torch.tensor([float(rank)]), "name": "train", "n": rank, "ok": True}
    if rank == 0:
        log["img"] = "rank-0 only"
    out = distributed.avg_log_dict_across_ranks(log)
    expected = (world - 1) / 2
    assert out["loss"] == expected and out["n"] == expected
    assert out["name"] == "train" and out["ok"] is True
    torch.testing.assert_close(out["t"], torch.tensor([expected]))
    assert ("img" in out) == (rank == 0)
    assert log["loss"] == float(rank)  # the input is not mutated

    distributed.teardown()
    assert not distributed.is_distributed()


def test_two_process_gloo():
    mp.spawn(_worker, args=(2, _free_port()), nprocs=2, join=True)
