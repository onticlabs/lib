"""Cross-cutting adapter regressions found while reviewing the upstream APIs."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ontic_nn.trackers import GeometrySequence, MVTracker, MVTrackerConfig, PointQueries
from ontic_nn.trackers import TrackCraft3R, TrackCraft3RConfig
from ontic_nn.trackers.mvtracker import _load_checkpoint
import ontic_nn.trackers.trackcraft3r as trackcraft


def test_reverse_tracking_preserves_absolute_time_world_frame_and_query_identity():
    class ForwardOnlyPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, *, rgbs, depths, intrs, extrs, query_points_3d):
            self.calls.append((rgbs.clone(), depths.clone(), intrs.clone(), extrs.clone()))
            b, v, t, _, h, w = rgbs.shape
            assert depths.shape == (b, v, t, 1, h, w)
            absolute_time = rgbs[0, 0, :, 0, 0, 0]
            query_time = query_points_3d[0, :, 0].long()
            delta = absolute_time[:, None] - absolute_time[query_time][None]
            xyz = query_points_3d[..., 1:][:, None].expand(-1, t, -1, -1).clone()
            xyz[..., 0] += delta
            before = torch.arange(t)[:, None] < query_time[None]
            xyz[:, before] = -999  # upstream has no supported estimate before each query
            visibility = (0.2 + absolute_time * 0.05)[None, :, None].expand(1, t, len(query_time))
            return {"traj_e": xyz, "vis_e_as_prob": visibility}

    b, t, v = 2, 8, 2
    depth = (torch.arange(t).float() + 2)[None, :, None, None, None].expand(b, t, v, 16, 16)
    c2w = torch.eye(4).repeat(b, t, v, 1, 1)
    c2w[..., 0, 3] = torch.arange(t)[None, :, None] * 0.2
    k = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]).repeat(b, t, v, 1, 1)
    geometry = GeometrySequence(depth, c2w, k)
    query = PointQueries(
        ids=torch.tensor([[91, 7], [3, 82]]),
        time=torch.tensor([[0, 5], [2, 7]]),
        xyz_world=torch.tensor(
            [[[0.0, 0.0, 2.0], [10.0, 0.0, 7.0]], [[30.0, 0.0, 4.0], [40.0, 0.0, 9.0]]]
        ),
    )
    images = (torch.arange(t).float() / 255)[None, :, None, None, None, None].expand(
        b, t, v, 3, 16, 16
    )
    model = ForwardOnlyPredictor()
    tracker = MVTracker(
        MVTrackerConfig(image_size=(16, 16), scene_normalization="none", bidirectional=True),
        model=model,
    )
    result = tracker(images, query, geometry=geometry)
    expected = query.xyz_world[:, None].expand(-1, t, -1, -1).clone()
    expected[..., 0] += torch.arange(t)[None, :, None] - query.time[:, None]
    torch.testing.assert_close(result.tracks_world, expected)
    torch.testing.assert_close(
        result.visibility, (0.2 + torch.arange(t) * 0.05)[None, :, None].expand(b, t, 2)
    )
    assert torch.equal(result.ids, query.ids)
    assert result.valid.all()
    assert len(model.calls) == 4
    for forward, backward in ((model.calls[0], model.calls[1]), (model.calls[2], model.calls[3])):
        for original, reversed_value in zip(forward, backward):
            torch.testing.assert_close(reversed_value, original.flip(2))


def test_mvtracker_checkpoint_cannot_silently_leave_random_parameters(tmp_path):
    path = tmp_path / "incomplete.pth"
    torch.save({"weight": torch.ones(1, 1)}, path)
    with pytest.raises(RuntimeError, match="missing"):
        _load_checkpoint(nn.Linear(1, 1), MVTrackerConfig(checkpoint_path=str(path)))


def test_trackcraft_offline_also_covers_wan_modelscope_assets(monkeypatch):
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        seen["offline"] = os.environ.get("MODELSCOPE_OFFLINE")
        return SimpleNamespace()

    monkeypatch.setattr(
        trackcraft,
        "import_research_module",
        lambda *args, **kw: SimpleNamespace(WanSceneFlowPredictor=factory),
    )
    monkeypatch.setattr(
        trackcraft, "resolve_checkpoint", lambda *args, **kw: "/models/model.safetensors"
    )
    monkeypatch.setenv("MODELSCOPE_OFFLINE", "0")
    trackcraft.load_predictor(TrackCraft3RConfig(allow_download=False))
    assert seen["offline"] == "1"
    assert seen["apply_speed_opts"] is False
    assert os.environ["MODELSCOPE_OFFLINE"] == "0"


def test_trackcraft_module_move_updates_predictor_context_without_moving_unused_text_encoder():
    text_encoder = nn.Linear(1, 1)
    predictor = SimpleNamespace(
        device="cpu",
        pipe=SimpleNamespace(
            dit=nn.Linear(1, 1).bfloat16(), text_encoder=text_encoder, device="cpu"
        ),
        _null_context=torch.zeros(1, 2, 3, dtype=torch.bfloat16),
    )
    tracker = TrackCraft3R(TrackCraft3RConfig(device="cpu"), predictor=predictor)
    tracker.to("meta")
    assert torch.device(predictor.device).type == "meta"
    assert torch.device(predictor.pipe.device).type == "meta"
    assert predictor._null_context.device.type == "meta"
    assert predictor.pipe.dit.weight.device.type == "meta"
    assert text_encoder.weight.device.type == "cpu"


def test_trackcraft_rejects_skew_instead_of_discarding_it():
    k = torch.tensor([[1.0, 0.1, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]).repeat(1, 12, 1, 1, 1)
    geometry = GeometrySequence(torch.ones(1, 12, 1, 4, 4), torch.eye(4).repeat(1, 12, 1, 1, 1), k)
    tracker = TrackCraft3R(TrackCraft3RConfig(), predictor=SimpleNamespace())
    with pytest.raises(ValueError, match="skew"):
        tracker._clip_intrinsics(geometry)
