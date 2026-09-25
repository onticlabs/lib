"""Collation: ragged target horizons, action padding, presence-channel contract."""

import pytest
import torch

from ontic_data.collate import collate_actions, collate_examples, harmonize_target_horizon


def _sample(T, n_actions=True, extra_key=None):
    s = {
        "scene": f"scene_{T}",
        "context": {"image": torch.rand(1, 2, 3, 4, 4), "near": torch.ones(1, 2)},
        "target": {
            "image": torch.rand(T, 1, 3, 4, 4),
            "index": torch.zeros(T, 1, dtype=torch.long),
        },
    }
    if n_actions:
        s["actions"] = {"hand": torch.rand(T, 1, 5, 4)}
        if extra_key:
            s["actions"][extra_key] = torch.rand(T, 1, 2, 4)
    return s


def test_harmonize_truncates_to_min_horizon_including_actions():
    batch = [_sample(5), _sample(6), _sample(7)]
    out = harmonize_target_horizon(batch)
    for s in out:
        assert s["target"]["image"].shape[0] == 5
        assert s["target"]["index"].shape[0] == 5
        assert s["actions"]["hand"].shape[0] == 5
    # a tensor whose leading dim is not the horizon is left alone
    assert out[0]["context"]["image"].shape[0] == 1


def test_harmonize_noop_when_equal():
    batch = [_sample(4), _sample(4)]
    assert harmonize_target_horizon(batch) is batch


def test_collate_examples_stacks_and_pads_actions():
    out = collate_examples([_sample(3, extra_key="obj"), _sample(4)])
    assert out["target"]["image"].shape == (2, 3, 1, 3, 4, 4)
    assert out["scene"] == ["scene_3", "scene_4"]
    assert out["actions"]["hand"].shape == (2, 3, 1, 5, 4)
    # a key missing from one sample is zero-padded (presence 0) for that sample
    assert out["actions"]["obj"].shape == (2, 3, 1, 2, 4)
    assert torch.all(out["actions"]["obj"][1] == 0)


def test_collate_examples_without_actions():
    out = collate_examples([_sample(2, n_actions=False), _sample(2, n_actions=False)])
    assert "actions" not in out


def test_collate_actions_pads_timesteps_and_rejects_missing_presence():
    padded = collate_actions([{"a": torch.ones(2, 1, 3, 4)}, {"a": torch.ones(5, 1, 3, 4)}, None])
    assert padded["a"].shape == (3, 5, 1, 3, 4)
    assert torch.all(padded["a"][0, 2:] == 0) and torch.all(padded["a"][2] == 0)
    with pytest.raises(ValueError, match="presence"):
        collate_actions([{"a": torch.ones(2, 1, 3, 3)}])
