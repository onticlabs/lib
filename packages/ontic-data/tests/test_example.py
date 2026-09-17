"""Example schema helpers fold ``batch *time`` into one axis."""

import torch

from ontic_data.example import to_batched_example, to_batched_views


def _views(b, t, v):
    return {
        "image": torch.rand(b, t, v, 3, 4, 4),
        "intrinsics": torch.rand(b, t, v, 3, 3),
        "extrinsics": torch.rand(b, t, v, 4, 4),
        "near": torch.ones(b, t, v),
        "index": torch.zeros(b, t, v, dtype=torch.long),
    }


def test_to_batched_views_folds_time():
    out = to_batched_views(_views(2, 3, 4))
    assert out["image"].shape == (6, 4, 3, 4, 4)
    assert out["intrinsics"].shape == (6, 4, 3, 3)
    assert out["near"].shape == (6, 4) and out["index"].shape == (6, 4)
    assert to_batched_views(out) is out  # already batched


def test_to_batched_example_folds_actions_and_string_index():
    ex = {
        "context": _views(2, 3, 1),
        "target": {**_views(2, 3, 2), "index": [["a", "b"]] * 6},
        "scene": ["s"] * 2,
        "actions": {"hand": torch.rand(2, 3, 1, 21, 4)},
    }
    out = to_batched_example(ex)
    assert out["actions"]["hand"].shape == (6, 1, 21, 4)
    assert out["target"]["index"] == ["a", "b"] * 6
