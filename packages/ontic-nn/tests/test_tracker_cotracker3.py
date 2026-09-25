"""CoTracker image tracks must lift to the right camera, depth pixel and identity."""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from ontic_nn.trackers import CoTracker3, CoTracker3Config, GeometrySequence, PointQueries
from ontic_nn.trackers import cotracker3 as adapter


def scene(b=1, t=4, v=1, h=8, w=8):
    c2w = torch.eye(4).repeat(b, t, v, 1, 1)
    c2w[..., 0, 3] = torch.arange(v) * 10.0
    c2w[..., 1, 3] = torch.arange(t)[None, :, None] * 0.1
    k = torch.tensor([[1.0, 0.1, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]]).repeat(b, t, v, 1, 1)
    return GeometrySequence(torch.full((b, t, v, h, w), 2.0), c2w, k)


def queries(g, *, time=None, views=None):
    b = g.depth.shape[0]
    times = torch.zeros(b, 3, dtype=torch.long) if time is None else time
    views = torch.zeros_like(times) if views is None else views
    uv = torch.tensor([[[2.5 / 8, 3.5 / 8], [4.5 / 8, 3.5 / 8], [5.5 / 8, 2.5 / 8]]]).expand(
        b, -1, -1
    )
    ids = torch.tensor([[91, 8, 72]]).expand(b, -1)
    return PointQueries.from_pixels(g, times, views, uv, ids=ids)


class ImageTracker(nn.Module):
    def __init__(self, edit=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.calls = []
        self.edit = edit

    def forward(self, video, *, queries, backward_tracking):
        self.calls.append((video.clone(), queries.clone(), backward_tracking))
        b, t = video.shape[:2]
        xy = queries[:, None, :, 1:].expand(b, t, -1, -1).clone()
        vis = torch.ones(xy.shape[:-1], dtype=torch.bool, device=video.device)
        if self.edit:
            self.edit(xy, vis)
        return xy, vis


def test_independent_views_batches_chunks_and_ids_with_moving_cameras():
    g = scene(b=2, v=3)
    views = torch.tensor([[2, 0, 2], [0, 0, 1]])
    times = torch.tensor([[0, 2, 3], [1, 0, 2]])
    q = queries(g, time=times, views=views)
    images = torch.zeros(2, 4, 3, 3, 8, 8)
    images[:, :, 0] = 0.25
    images[:, :, 1] = 0.5
    images[:, :, 2] = 0.75
    model = ImageTracker()
    tracker = CoTracker3(CoTracker3Config(query_chunk_size=1), model=model)
    out = tracker(images, q, geometry=g)
    expected = q.xyz_world[:, None].expand(-1, 4, -1, -1).clone()
    expected[..., 1] += (torch.arange(4)[None, :, None] - q.time[:, None]) * 0.1
    torch.testing.assert_close(out.tracks_world, expected)
    assert torch.equal(out.ids, q.ids) and out.valid.all()
    assert out.visibility_scope == "query_view"
    assert out.metadata["view_processing"].startswith("independent")
    assert len(model.calls) == 6  # empty cameras are not run
    assert [float(c[0].max()) for c in model.calls] == [63.75, 191.25, 191.25, 63.75, 63.75, 127.5]
    assert all(backward for _, _, backward in model.calls)
    assert not tracker.model.weight.requires_grad
    tracker.train()
    assert not tracker.model.training


def test_invalid_depth_occlusion_out_of_bounds_and_nonfinite_tracks_are_not_3d_points():
    g = scene(t=5)
    q = queries(g)
    g.depth[0, 1, 0, 3, 2] = 0
    g.depth_valid = torch.ones_like(g.depth, dtype=torch.bool)
    g.depth_valid[0, 2, 0, 3, 4] = False

    def edit(xy, vis):
        vis[0, 2, 0] = False
        xy[0, 3, 0] = torch.tensor([-0.1, 3.0])
        xy[0, 3, 1] = torch.tensor([8.0, 3.0])
        xy[0, 4, 0] = torch.nan
        xy[0, 4, 1] = torch.inf

    out = CoTracker3(CoTracker3Config(), model=ImageTracker(edit))(
        torch.zeros(1, 5, 1, 3, 8, 8), q, geometry=g
    )
    expected = torch.tensor([[[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]]).bool()
    assert torch.equal(out.valid, expected)
    assert torch.isnan(out.tracks_world[~expected]).all()
    assert out.visibility[0, 1, 0] == 1  # depth validity is distinct from image visibility
    assert out.visibility[0, 2, 0] == 0


def test_lifting_keeps_pixel_centers_when_rgb_and_depth_resolutions_differ():
    g = scene()
    q = queries(g)
    model = ImageTracker()
    out = CoTracker3(CoTracker3Config(), model=model)(
        torch.zeros(1, 4, 1, 3, 16, 16), q, geometry=g
    )
    torch.testing.assert_close(model.calls[0][1][0, :, 1:], q.source_uv[0] * 16 - 0.5)
    expected = q.xyz_world[:, None].expand(-1, 4, -1, -1).clone()
    expected[..., 1] += torch.arange(4)[None, :, None] * 0.1
    torch.testing.assert_close(out.tracks_world, expected)


def test_xyz_only_query_and_forward_only_time_mask():
    g = scene()
    q = queries(g, time=torch.full((1, 3), 2, dtype=torch.long))
    q.source_view = q.source_uv = None
    model = ImageTracker()
    out = CoTracker3(CoTracker3Config(bidirectional=False), model=model)(
        torch.zeros(1, 4, 1, 3, 8, 8), q, geometry=g
    )
    assert not out.valid[:, :2].any() and out.valid[:, 2:].all()
    assert not model.calls[0][2]


def test_motion_samples_each_frames_depth_and_rotated_camera():
    g = scene(t=3)
    q = queries(g)
    g.depth[0, 1] = 3
    g.depth[0, 2] = 4
    g.intrinsics[..., 0, 1] = 0
    q = queries(g)
    g.extrinsics[0, 2, 0, :3, :3] = torch.tensor([[0.0, -1, 0], [1, 0, 0], [0, 0, 1]])

    def edit(xy, vis):
        xy[:, 1, :, 0] += 0.6  # nearest pixel advances by one
        xy[:, 2, :, 1] += 1

    out = CoTracker3(CoTracker3Config(), model=ImageTracker(edit))(
        torch.zeros(1, 3, 1, 3, 8, 8), q, geometry=g
    )
    torch.testing.assert_close(out.tracks_world[0, 1, 0], torch.tensor([-0.1875, -0.0875, 3]))
    torch.testing.assert_close(out.tracks_world[0, 2, 0], torch.tensor([-0.25, -0.55, 4]))


def test_empty_queries_do_not_call_model():
    g = scene()
    q = PointQueries(
        torch.empty(1, 0, dtype=torch.long),
        torch.empty(1, 0, dtype=torch.long),
        torch.empty(1, 0, 3),
    )
    model = ImageTracker()
    out = CoTracker3(CoTracker3Config(), model=model)(torch.zeros(1, 4, 1, 3, 8, 8), q, geometry=g)
    assert out.tracks_world.shape == (1, 4, 0, 3) and not model.calls


def test_inconsistent_query_depth_and_missing_multiview_provenance_fail():
    g = scene(v=2)
    q = queries(g)
    tracker = CoTracker3(CoTracker3Config(), model=ImageTracker())
    images = torch.zeros(1, 4, 2, 3, 8, 8)
    q.xyz_world *= 2
    with pytest.raises(ValueError, match="visible source depth"):
        tracker(images, q, geometry=g)
    q.source_view = q.source_uv = None
    with pytest.raises(ValueError, match="source_view"):
        tracker(images, q, geometry=g)


@pytest.mark.parametrize(
    "options", [{"freeze_tracker": False}, {"window_len": 1}, {"query_chunk_size": 0}]
)
def test_invalid_configuration_rejected(options):
    with pytest.raises(ValueError, match="cotracker3"):
        CoTracker3(CoTracker3Config(**options), model=ImageTracker())


def test_loader_matches_pointworld_offline_predictor_and_explicit_checkpoint(monkeypatch, tmp_path):
    repo = tmp_path / "third_party/co-tracker"
    (repo / "cotracker").mkdir(parents=True)
    (repo / "cotracker/predictor.py").touch()
    checkpoint = tmp_path / "scaled_online.pth"
    checkpoint.touch()
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        assert sys.path[0] == str(repo)
        return ImageTracker()

    def imports(*args, **kwargs):
        assert args == ("cotracker.predictor",)
        return SimpleNamespace(CoTrackerPredictor=factory)

    monkeypatch.setattr(adapter, "import_research_module", imports)
    monkeypatch.setenv("ONTIC_COTRACKER3_REPO", str(tmp_path))
    before = list(sys.path)
    model = CoTracker3(CoTracker3Config(checkpoint_path=str(checkpoint), allow_download=False))
    assert seen == dict(checkpoint=str(checkpoint), offline=True, v2=False, window_len=16)
    assert model._checkpoint == str(checkpoint) and sys.path == before


def test_wrong_repository_and_absent_checkpoint_do_not_initialize_random_model(tmp_path):
    with pytest.raises(ImportError, match="CoTracker checkout"):
        CoTracker3(CoTracker3Config(repo_path=str(tmp_path)))
    with pytest.raises(ValueError, match="pretrained checkpoint"):
        CoTracker3(CoTracker3Config(model_dir="", allow_download=False))
