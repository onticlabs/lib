"""Contract tests for the TAPIP3D tracker wrapper (CPU, fake upstream model, no weights).

The fake model stands in for upstream ``PointTracker3D``: it records exactly what the wrapper
hands the native API and returns deterministic trajectories, so the assertions below are about
the adapter's conventions — axes, RGB range, camera direction, pixel centres, query times,
padding, batching, visibility — not about its own arithmetic mirrored back.
"""

from __future__ import annotations

import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest
import torch
import torch.nn as nn

from ontic_nn.trackers.common import GeometrySequence, PointQueries
from ontic_nn.trackers.registry import TRACKERS
from ontic_nn.trackers.tapip3d import (
    REPO_PATH_ENV,
    TAPIP3D,
    TAPIP3D_REVISION,
    UPSTREAM_TOP_LEVEL,
    TAPIP3DConfig,
    _build_from_checkpoint,
    _check_module_collisions,
    _depth_roi,
    _load_weights,
    _points_on_a_grid,
    _resolve_repo_path,
)

H, W = 8, 12  # the fake's native resolution; tests default to running 1:1


# ---------------------------------------------------------------------------
# Fake upstream model
# ---------------------------------------------------------------------------
@dataclass
class _FakePrediction:
    coords: torch.Tensor  # (B, T, N, 3)
    visibs: torch.Tensor  # (B, T, N), logits


class FakeTracker(nn.Module):
    """Upstream ``PointTracker3D`` stand-in.

    Each call returns ``query_xyz + [t + 100 * call_index, 0, 0]`` and logits
    ``t - 1 + 5 * call_index``, so a forward pass, a reverse-time pass and successive batch
    items are all distinguishable in the wrapper's output.
    """

    bidirectional = False

    def __init__(self, image_size: Tuple[int, int] = (H, W), seq_len: int = 4) -> None:
        super().__init__()
        self.image_size = tuple(image_size)
        self.seq_len = seq_len
        self.eval_mode = "local"  # upstream's constructor default; load_model switches it
        self.calls: List[Dict[str, Any]] = []
        self.weight = nn.Parameter(torch.zeros(1))

    def set_image_size(self, image_size) -> None:
        self.image_size = tuple(image_size)

    def set_eval_mode(self, eval_mode: str) -> None:
        self.eval_mode = eval_mode

    def forward(
        self,
        *,
        rgb_obs: torch.Tensor,
        depth_obs: torch.Tensor,
        num_iters: int,
        query_point: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        mode: str,
        **extra: Any,
    ):
        tag = len(self.calls)
        self.calls.append(
            {
                "rgb_obs": rgb_obs.clone(),
                "depth_obs": depth_obs.clone(),
                "num_iters": num_iters,
                "query_point": query_point.clone(),
                "intrinsics": intrinsics.clone(),
                "extrinsics": extrinsics.clone(),
                "mode": mode,
                "image_size": self.image_size,
                "extra": {k: (v.clone() if torch.is_tensor(v) else v) for k, v in extra.items()},
            }
        )
        b, t = rgb_obs.shape[:2]
        n = query_point.shape[1]
        steps = torch.arange(t, dtype=torch.float32)
        offset = torch.zeros(t, n, 3)
        offset[..., 0] = steps[:, None] + 100.0 * tag
        coords = query_point[:, None, :, 1:] + offset[None]
        visibs = (steps[:, None] - 1.0 + 5.0 * tag).expand(t, n)[None].expand(b, t, n).clone()
        return _FakePrediction(coords=coords, visibs=visibs), []


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------
def _images(b: int, t: int, h: int = H, w: int = W) -> torch.Tensor:
    n = b * t * 3 * h * w
    return torch.linspace(0.0, 1.0, n).reshape(b, t, 1, 3, h, w)


def _intrinsics(b: int, t: int) -> torch.Tensor:
    """Normalised pinhole: half-width focal, centred principal point."""
    k = torch.zeros(b, t, 1, 3, 3)
    k[..., 0, 0] = 0.5
    k[..., 1, 1] = 0.5
    k[..., 0, 2] = 0.5
    k[..., 1, 2] = 0.5
    k[..., 2, 2] = 1.0
    return k


def _geometry(
    b: int,
    t: int,
    *,
    h: int = H,
    w: int = W,
    depth: Optional[torch.Tensor] = None,
    depth_valid: Optional[torch.Tensor] = None,
    extrinsics: Optional[torch.Tensor] = None,
) -> GeometrySequence:
    if depth is None:
        depth = torch.full((b, t, 1, h, w), 2.0)
    if extrinsics is None:
        extrinsics = torch.eye(4).expand(b, t, 1, 4, 4).clone()
    if depth_valid is None:
        depth_valid = torch.ones_like(depth, dtype=torch.bool)
    return GeometrySequence(
        depth=depth,
        depth_valid=depth_valid,
        extrinsics=extrinsics,
        intrinsics=_intrinsics(b, t),
    )


def _queries(xyz: Sequence[Sequence[float]], times: Sequence[int], b: int = 1) -> PointQueries:
    xyz_t = torch.tensor(xyz, dtype=torch.float32)[None].expand(b, -1, -1).clone()
    time_t = torch.tensor(times, dtype=torch.int64)[None].expand(b, -1).clone()
    ids = torch.arange(xyz_t.shape[1], dtype=torch.int64)[None].expand(b, -1).clone()
    return PointQueries(ids=ids, time=time_t, xyz_world=xyz_t, source_view=None, source_uv=None)


def _tracker(**overrides: Any) -> Tuple[TAPIP3D, FakeTracker]:
    fake = FakeTracker(
        image_size=overrides.pop("image_size", (H, W)),
        seq_len=overrides.pop("seq_len", 4),
    )
    cfg_kwargs: Dict[str, Any] = {
        "resolution_factor": 1.0,
        "support_grid_size": 0,
        "use_depth_roi": False,
    }
    cfg_kwargs.update(overrides)
    return TAPIP3D(TAPIP3DConfig(**cfg_kwargs), model=fake), fake


# ---------------------------------------------------------------------------
# Registry / config
# ---------------------------------------------------------------------------
def test_registered_under_tapip3d():
    assert TRACKERS["tapip3d"] is TAPIP3DConfig


def test_capabilities_are_monocular_and_offline():
    caps = TAPIP3DConfig.CAPABILITIES
    assert caps is TAPIP3D.CAPABILITIES
    assert caps.multiview is False and caps.dense is False
    assert caps.arbitrary_query_times is True
    assert caps.execution_mode == "offline"
    assert caps.visibility_scope == "query_view"


def test_defaults_match_the_released_checkpoint_and_demo():
    cfg = TAPIP3DConfig()
    assert (cfg.model_dir, cfg.checkpoint_file) == ("zbww/tapip3d", "tapip3d_final.pth")
    assert cfg.eval_mode == "raw"  # "local" would cancel camera motion
    assert (cfg.num_iters, cfg.support_grid_size, cfg.resolution_factor) == (6, 16, 2.0)
    assert cfg.bidirectional and cfg.freeze_tracker and cfg.allow_download


def test_build_without_a_checkout_explains_how_to_get_one(monkeypatch):
    monkeypatch.delenv(REPO_PATH_ENV, raising=False)
    with pytest.raises(ImportError, match=r"github\.com/zbw001/TAPIP3D"):
        TAPIP3DConfig().build()


def test_rejects_unknown_eval_mode():
    with pytest.raises(ValueError, match="eval_mode"):
        TAPIP3D(TAPIP3DConfig(eval_mode="global"), model=FakeTracker())


def test_eval_mode_is_applied_to_the_model():
    tracker, fake = _tracker()
    assert fake.eval_mode == "raw"
    _, local = _tracker(eval_mode="local")
    assert local.eval_mode == "local"
    assert tracker.model is fake


def test_frozen_by_default():
    tracker, fake = _tracker()
    assert all(not p.requires_grad for p in tracker.parameters())
    assert not fake.training


def test_inference_size_follows_upstream_resolution_factor():
    tracker, _ = _tracker(image_size=(384, 512), resolution_factor=2.0)
    # inference.py: int(side * sqrt(resolution_factor)) off the training resolution
    assert tracker.inference_size() == (int(384 * math.sqrt(2)), int(512 * math.sqrt(2)))
    assert tracker.native_image_size == (384, 512)


def test_inference_size_overrides():
    explicit, _ = _tracker(image_size=(384, 512), inference_resolution=(100, 140))
    assert explicit.inference_size() == (100, 140)
    by_long_side, _ = _tracker(image_size=(384, 512), resolution_factor=None, long_side=256)
    assert by_long_side.inference_size() == (192, 256)


def test_native_image_size_is_captured_before_set_image_size():
    tracker, fake = _tracker(image_size=(384, 512), inference_resolution=(96, 128))
    fake.set_image_size((96, 128))  # what a forward pass does to the model
    assert tracker.native_image_size == (384, 512)
    assert tracker.inference_size() == (96, 128)


# ---------------------------------------------------------------------------
# Native call: axes, ranges, cameras, pixel centres
# ---------------------------------------------------------------------------
def test_native_call_axes_and_rgb_range():
    tracker, fake = _tracker()
    images = _images(1, 5)
    queries = _queries([[0.0, 0.0, 2.0], [0.3, -0.2, 2.5]], [0, 0])
    out = tracker(images, queries, geometry=_geometry(1, 5))

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["rgb_obs"].shape == (1, 5, 3, H, W)  # view axis dropped, channels first
    assert call["depth_obs"].shape == (1, 5, H, W)
    assert call["intrinsics"].shape == (1, 5, 3, 3)
    assert call["extrinsics"].shape == (1, 5, 4, 4)
    assert call["query_point"].shape == (1, 2, 4)
    assert call["mode"] == "inference" and call["num_iters"] == 6
    assert call["image_size"] == (H, W)  # set_image_size ran before the forward
    assert torch.allclose(call["rgb_obs"], images[:, :, 0])
    assert call["rgb_obs"].min() >= 0.0 and call["rgb_obs"].max() <= 1.0
    assert out.tracks_world.shape == (1, 5, 2, 3)


def test_query_point_is_time_then_world_xyz():
    tracker, fake = _tracker()
    xyz = [[0.1, -0.2, 2.0], [0.0, 0.0, 3.0]]
    tracker(_images(1, 5), _queries(xyz, [0, 2]), geometry=_geometry(1, 5))
    qp = fake.calls[0]["query_point"]
    assert torch.equal(qp[0, :, 0], torch.tensor([0.0, 2.0]))
    assert torch.allclose(qp[0, :, 1:], torch.tensor(xyz))


def test_intrinsics_are_pixel_matrices_over_integer_centres():
    tracker, fake = _tracker()
    tracker(_images(1, 5), _queries([[0.0, 0.0, 2.0]], [0]), geometry=_geometry(1, 5))
    k = fake.calls[0]["intrinsics"][0, 0]
    # normalised (0.5, 0.5) focal/centre at 8x12 -> fx = 6, cx = 0.5 * 12 - 0.5 = 5.5
    expected = torch.tensor([[6.0, 0.0, 5.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]])
    assert torch.allclose(k, expected)


def test_extrinsics_are_world_to_camera_for_a_moving_camera():
    rot = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # cam +z -> world +x
    position = torch.tensor([2.0, -1.0, 0.5])
    c2w = torch.eye(4)
    c2w[:3, :3] = rot
    c2w[:3, 3] = position
    poses = torch.eye(4).expand(1, 5, 1, 4, 4).clone()
    poses[:, 2, 0] = c2w  # only frame 2 moves, so a per-frame mix-up would show up

    tracker, fake = _tracker()
    tracker(
        _images(1, 5),
        _queries([[0.0, 0.0, 2.0]], [0]),
        geometry=_geometry(1, 5, extrinsics=poses),
    )
    ext = fake.calls[0]["extrinsics"][0]
    assert torch.allclose(ext[0], torch.eye(4), atol=1e-6)
    assert torch.allclose(ext[2], torch.linalg.inv(c2w), atol=1e-6)

    # Upstream lifts a pixel with inv(extrinsics) @ (inv(K) [u, v, 1] * depth): a point 2 m in
    # front of frame 2's camera must land at position + 2 * (camera z in world).
    k = fake.calls[0]["intrinsics"][0, 2]
    pixel = torch.tensor([5.5, 3.5, 1.0])  # the principal point
    camera = torch.linalg.inv(k) @ pixel * 2.0
    world = torch.linalg.inv(ext[2]) @ torch.cat([camera, torch.ones(1)])
    assert torch.allclose(world[:3], position + rot @ torch.tensor([0.0, 0.0, 2.0]), atol=1e-5)

    off_centre = torch.tensor([11.5, 3.5, 1.0])  # +1 focal length to the right
    camera = torch.linalg.inv(k) @ off_centre * 2.0
    assert torch.allclose(camera, torch.tensor([2.0, 0.0, 2.0]), atol=1e-5)


def test_invalid_depth_is_passed_as_zero():
    depth = torch.full((1, 4, 1, H, W), 2.0)
    depth[..., :, : W // 2] = 7.0  # would be visible if the mask were ignored
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[..., :, : W // 2] = False

    tracker, fake = _tracker()
    tracker(
        _images(1, 4),
        _queries([[0.0, 0.0, 2.0]], [0]),
        geometry=_geometry(1, 4, depth=depth, depth_valid=valid),
    )
    passed = fake.calls[0]["depth_obs"][0, 0]
    assert torch.all(passed[:, : W // 2] == 0.0)  # upstream's invalid sentinel
    assert torch.all(passed[:, W // 2 :] == 2.0)


def test_inputs_are_resized_to_the_inference_resolution():
    tracker, fake = _tracker(image_size=(4, 6), resolution_factor=1.0)
    images = _images(1, 4, h=H, w=W)
    geometry = _geometry(1, 4, h=H, w=W)
    out = tracker(images, _queries([[0.0, 0.0, 2.0]], [0]), geometry=geometry)

    call = fake.calls[0]
    assert call["rgb_obs"].shape == (1, 4, 3, 4, 6)
    assert call["depth_obs"].shape == (1, 4, 4, 6)
    assert torch.allclose(call["depth_obs"], torch.full((1, 4, 4, 6), 2.0))  # depth is not rescaled
    # normalised intrinsics are resolution independent: fx = 0.5 * 6, cx = 0.5 * 6 - 0.5
    assert torch.allclose(
        call["intrinsics"][0, 0],
        torch.tensor([[3.0, 0.0, 2.5], [0.0, 2.0, 1.5], [0.0, 0.0, 1.0]]),
    )
    assert out.metadata["inference_resolution"] == (4, 6)
    assert out.tracks_world.shape == (1, 4, 1, 3)  # public grid is unchanged


# ---------------------------------------------------------------------------
# Query times: reverse pass and unsupported prefixes
# ---------------------------------------------------------------------------
def test_backward_pass_fills_frames_before_the_query_time():
    tracker, fake = _tracker()
    xyz = [[0.0, 0.0, 2.0]]
    out = tracker(_images(1, 5), _queries(xyz, [3]), geometry=_geometry(1, 5))

    assert len(fake.calls) == 2
    forward, backward = fake.calls
    assert torch.allclose(backward["rgb_obs"], forward["rgb_obs"].flip(dims=(1,)))
    assert torch.allclose(backward["depth_obs"], forward["depth_obs"].flip(dims=(1,)))
    assert torch.allclose(backward["extrinsics"], forward["extrinsics"].flip(dims=(1,)))
    assert backward["query_point"][0, 0, 0].item() == 1.0  # T - 1 - 3
    assert torch.allclose(backward["query_point"][0, :, 1:], torch.tensor(xyz))

    base = torch.tensor(xyz)[0]
    for t in range(5):
        dx = (100.0 + (4 - t)) if t < 3 else float(t)  # backward call is tagged +100
        assert torch.allclose(out.tracks_world[0, t, 0], base + torch.tensor([dx, 0.0, 0.0]))
    expected_logits = torch.tensor([8.0, 7.0, 6.0, 2.0, 3.0])
    assert torch.allclose(out.visibility[0, :, 0], torch.sigmoid(expected_logits))
    assert bool(out.valid.all())
    assert out.metadata["backward_pass"] == (True,)


def test_no_backward_pass_when_every_query_is_at_frame_zero():
    tracker, fake = _tracker()
    out = tracker(_images(1, 5), _queries([[0.0, 0.0, 2.0]], [0]), geometry=_geometry(1, 5))
    assert len(fake.calls) == 1
    assert out.metadata["backward_pass"] == (False,)
    assert bool(out.valid.all())


def test_prefix_is_invalid_when_the_backward_pass_is_disabled():
    tracker, fake = _tracker(bidirectional=False)
    queries = _queries([[0.0, 0.0, 2.0], [0.1, 0.1, 2.0]], [2, 0])
    out = tracker(_images(1, 6), queries, geometry=_geometry(1, 6))
    assert len(fake.calls) == 1
    assert not bool(out.valid[0, :2, 0].any())  # before its query time: unsupported
    assert bool(out.valid[0, 2:, 0].all())
    assert bool(out.valid[0, :, 1].all())  # queried at frame 0
    assert torch.isfinite(out.visibility).all()


def test_no_backward_pass_for_a_natively_bidirectional_model():
    fake = FakeTracker()
    fake.bidirectional = True
    tracker = TAPIP3D(
        TAPIP3DConfig(resolution_factor=1.0, support_grid_size=0, use_depth_roi=False), model=fake
    )
    out = tracker(_images(1, 5), _queries([[0.0, 0.0, 2.0]], [3]), geometry=_geometry(1, 5))
    assert len(fake.calls) == 1
    assert bool(out.valid.all())


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------
def test_world_tracks_are_returned_unchanged():
    tracker, _ = _tracker()
    xyz = [[0.4, -0.3, 2.2]]
    out = tracker(_images(1, 5), _queries(xyz, [0]), geometry=_geometry(1, 5))
    base = torch.tensor(xyz)[0]
    for t in range(5):
        assert torch.allclose(out.tracks_world[0, t, 0], base + torch.tensor([float(t), 0.0, 0.0]))


def test_visibility_is_the_native_sigmoid_with_a_per_view_copy():
    tracker, _ = _tracker()
    out = tracker(_images(1, 5), _queries([[0.0, 0.0, 2.0]], [0]), geometry=_geometry(1, 5))
    logits = torch.arange(5, dtype=torch.float32) - 1.0
    assert torch.allclose(out.visibility[0, :, 0], torch.sigmoid(logits))
    assert out.visibility_scope == "query_view"
    assert out.visibility_per_view is not None
    assert out.visibility_per_view.shape == (1, 5, 1, 1)
    assert torch.allclose(out.visibility_per_view[:, :, 0], out.visibility)
    assert out.visibility.min() >= 0.0 and out.visibility.max() <= 1.0


def test_ids_and_metadata_are_preserved():
    tracker, _ = _tracker()
    queries = _queries([[0.0, 0.0, 2.0], [0.2, 0.1, 2.0]], [0, 1])
    out = tracker(_images(1, 5), queries, geometry=_geometry(1, 5))
    assert torch.equal(out.ids, queries.ids)
    assert out.metadata["tracker"] == "tapip3d"
    assert out.metadata["upstream_api_revision"] == TAPIP3D_REVISION
    assert out.metadata["eval_mode"] == "raw"
    assert out.metadata["native_image_size"] == (H, W)
    assert "world_to_camera" in out.metadata["extrinsics_convention"]


def test_empty_queries_short_circuit():
    tracker, fake = _tracker()
    empty = PointQueries(
        ids=torch.zeros(1, 0, dtype=torch.int64),
        time=torch.zeros(1, 0, dtype=torch.int64),
        xyz_world=torch.zeros(1, 0, 3),
        source_view=None,
        source_uv=None,
    )
    out = tracker(_images(1, 5), empty, geometry=_geometry(1, 5))
    assert fake.calls == []
    assert out.tracks_world.shape == (1, 5, 0, 3)
    assert out.ids.shape == (1, 0)


# ---------------------------------------------------------------------------
# Native restrictions: views, batch, clip length
# ---------------------------------------------------------------------------
def test_multiple_views_are_rejected():
    tracker, fake = _tracker()
    images = _images(1, 5).expand(-1, -1, 2, -1, -1, -1).contiguous()
    geometry = _geometry(1, 5)
    two_view = GeometrySequence(
        depth=geometry.depth.expand(-1, -1, 2, -1, -1).contiguous(),
        depth_valid=geometry.depth_valid.expand(-1, -1, 2, -1, -1).contiguous(),
        extrinsics=geometry.extrinsics.expand(-1, -1, 2, -1, -1).contiguous(),
        intrinsics=geometry.intrinsics.expand(-1, -1, 2, -1, -1).contiguous(),
    )
    with pytest.raises(ValueError, match="monocular"):
        tracker(images, _queries([[0.0, 0.0, 2.0]], [0]), geometry=two_view)
    assert fake.calls == []


def test_batch_is_looped_one_clip_at_a_time():
    tracker, fake = _tracker()
    queries = _queries([[0.0, 0.0, 2.0]], [0], b=2)
    queries.xyz_world[1, 0] = torch.tensor([1.0, 1.0, 3.0])
    out = tracker(_images(2, 5), queries, geometry=_geometry(2, 5))

    assert len(fake.calls) == 2  # upstream refuses B > 1
    for call in fake.calls:
        assert call["rgb_obs"].shape[0] == 1 and call["query_point"].shape[0] == 1
    assert torch.allclose(fake.calls[0]["query_point"][0, 0, 1:], queries.xyz_world[0, 0])
    assert torch.allclose(fake.calls[1]["query_point"][0, 0, 1:], queries.xyz_world[1, 0])
    # second item went through the second call (tag 1 -> +100 on x), in its own batch slot
    assert torch.allclose(out.tracks_world[0, 0, 0], queries.xyz_world[0, 0])
    assert torch.allclose(
        out.tracks_world[1, 0, 0], queries.xyz_world[1, 0] + torch.tensor([100.0, 0.0, 0.0])
    )


def test_short_clips_are_padded_to_one_window_and_sliced_back():
    tracker, fake = _tracker(seq_len=4)
    images = _images(1, 3)
    out = tracker(images, _queries([[0.0, 0.0, 2.0]], [0]), geometry=_geometry(1, 3))

    call = fake.calls[0]
    assert call["rgb_obs"].shape[1] == 4  # upstream's window loop needs a full window
    assert torch.allclose(call["rgb_obs"][:, 3], call["rgb_obs"][:, 2])  # last frame repeated
    assert torch.allclose(call["extrinsics"][:, 3], call["extrinsics"][:, 2])
    assert out.tracks_world.shape == (1, 3, 1, 3)
    assert out.metadata["padded_frames"] == 1


def test_clips_at_or_above_the_window_length_are_not_padded():
    tracker, fake = _tracker(seq_len=4)
    out = tracker(_images(1, 5), _queries([[0.0, 0.0, 2.0]], [0]), geometry=_geometry(1, 5))
    assert fake.calls[0]["rgb_obs"].shape[1] == 5
    assert out.metadata["padded_frames"] == 0


# ---------------------------------------------------------------------------
# Support grid and depth ROI
# ---------------------------------------------------------------------------
def test_support_grid_queries_are_appended_and_sliced_off():
    tracker, fake = _tracker(support_grid_size=3)
    out = tracker(_images(1, 4), _queries([[0.0, 0.0, 2.0]], [0]), geometry=_geometry(1, 4))

    qp = fake.calls[0]["query_point"]
    assert qp.shape == (1, 1 + 9, 4)  # 3x3 grid on top of the real query
    assert torch.all(qp[0, 1:, 0] == 0.0)  # upstream anchors the support grid at frame 0
    # identity pose, depth 2: the grid must unproject to (x, y, 2) at the grid pixels
    grid = _points_on_a_grid(3, H, W, device=torch.device("cpu"), dtype=torch.float32)[0]
    expected_x = (grid[:, 0] - 5.5) / 6.0 * 2.0
    expected_y = (grid[:, 1] - 3.5) / 4.0 * 2.0
    assert torch.allclose(qp[0, 1:, 1], expected_x, atol=1e-5)
    assert torch.allclose(qp[0, 1:, 2], expected_y, atol=1e-5)
    assert torch.allclose(qp[0, 1:, 3], torch.full((9,), 2.0), atol=1e-5)
    assert out.tracks_world.shape == (1, 4, 1, 3)  # support tracks are not returned


def test_support_grid_skips_frames_without_depth():
    depth = torch.full((1, 4, 1, H, W), 2.0)
    depth[:, 0] = 0.0  # no valid depth to lift at the anchor frame
    tracker, fake = _tracker(support_grid_size=3)
    tracker(
        _images(1, 4),
        _queries([[0.0, 0.0, 2.0]], [0]),
        geometry=_geometry(1, 4, depth=depth, depth_valid=depth > 0),
    )
    assert fake.calls[0]["query_point"].shape == (1, 1, 4)


def test_points_on_a_grid_matches_cotracker():
    grid = _points_on_a_grid(2, 8, 12, device=torch.device("cpu"), dtype=torch.float32)
    margin = 12 / 64
    assert grid.shape == (1, 4, 2)
    assert torch.allclose(grid[0, 0], torch.tensor([margin, margin]))
    assert torch.allclose(grid[0, 3], torch.tensor([12 - margin, 8 - margin]))
    single = _points_on_a_grid(1, 8, 12, device=torch.device("cpu"), dtype=torch.float32)
    assert torch.allclose(single[0, 0], torch.tensor([6.0, 4.0]))


def test_depth_roi_is_the_upstream_iqr_window():
    depth = torch.arange(1.0, 101.0).reshape(1, 1, 10, 10)
    roi = _depth_roi(depth)
    q25 = torch.kthvalue(depth.reshape(-1), 25).values
    q75 = torch.kthvalue(depth.reshape(-1), 75).values
    assert roi.shape == (2,)
    assert roi[0].item() == pytest.approx(1e-7)
    assert roi[1].item() == pytest.approx((q75 + 1.5 * (q75 - q25)).item())


def test_depth_roi_ignores_invalid_depth_and_reaches_the_model():
    depth = torch.full((1, 4, 1, H, W), 2.0)
    depth[..., 0, :] = 0.0
    tracker, fake = _tracker(use_depth_roi=True)
    tracker(
        _images(1, 4),
        _queries([[0.0, 0.0, 2.0]], [0]),
        geometry=_geometry(1, 4, depth=depth, depth_valid=depth > 0),
    )
    roi = fake.calls[0]["extra"]["depth_roi"]
    assert roi.shape == (2,) and torch.allclose(roi, torch.tensor([1e-7, 2.0]))


def test_depth_roi_can_be_switched_off():
    tracker, fake = _tracker(use_depth_roi=False)
    tracker(_images(1, 4), _queries([[0.0, 0.0, 2.0]], [0]), geometry=_geometry(1, 4))
    assert "depth_roi" not in fake.calls[0]["extra"]


def test_a_clip_without_any_valid_depth_is_rejected():
    depth = torch.zeros(1, 4, 1, H, W)
    tracker, _ = _tracker(use_depth_roi=True)
    with pytest.raises(ValueError, match="depth"):
        tracker(
            _images(1, 4),
            _queries([[0.0, 0.0, 2.0]], [0]),
            geometry=_geometry(1, 4, depth=depth, depth_valid=depth > 0),
        )


# ---------------------------------------------------------------------------
# Upstream import guard and checkpoint loading
# ---------------------------------------------------------------------------
def test_module_collision_is_reported_not_overwritten(monkeypatch, tmp_path):
    foreign = types.ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", foreign)
    with pytest.raises(ImportError, match="models"):
        _check_module_collisions(tmp_path.resolve())
    assert sys.modules["models"] is foreign  # nothing was removed or replaced


def test_module_collision_detects_a_blocked_entry(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(ImportError, match="datasets"):
        _check_module_collisions(tmp_path.resolve())


def test_modules_from_the_checkout_itself_are_not_a_collision(monkeypatch, tmp_path):
    repo = tmp_path.resolve()
    for name in UPSTREAM_TOP_LEVEL:  # whatever this interpreter happens to have imported
        monkeypatch.delitem(sys.modules, name, raising=False)
    own = types.ModuleType("models")
    own.__file__ = str(repo / "models" / "__init__.py")
    monkeypatch.setitem(sys.modules, "models", own)
    _check_module_collisions(repo)


def test_checkpoint_must_carry_its_config(tmp_path):
    with pytest.raises(RuntimeError, match="not a TAPIP3D checkpoint"):
        _build_from_checkpoint(object(), {"state_dict": {}}, "ckpt.pth")


def test_checkpoint_builds_from_its_own_config():
    built: Dict[str, Any] = {}

    class _Models:
        @staticmethod
        def from_config(cfg, *, image_size):
            built["cfg"] = cfg
            built["image_size"] = image_size
            return nn.Linear(2, 3)

    reference = nn.Linear(2, 3)
    weights = {k: torch.full_like(v, 0.25) for k, v in reference.state_dict().items()}
    state = {
        "cfg": {"model": {"name": "point_tracker_3d"}, "train_dataset": {"resolution": [384, 512]}},
        "weight": weights,
    }
    model = _build_from_checkpoint(_Models(), state, "ckpt.pth")
    assert built["image_size"] == (384, 512)
    assert built["cfg"] == {"name": "point_tracker_3d"}
    assert torch.allclose(model.weight, torch.full_like(model.weight, 0.25))
    assert not model.training


def test_full_checkpoint_skips_encoder_download_without_mutating_training_config():
    class _Models:
        @staticmethod
        def from_config(cfg, *, image_size):
            assert cfg["encoder"]["pretrained"] is False
            return nn.Linear(2, 3)

    reference = nn.Linear(2, 3)
    encoder = {"name": "cotracker_cnn", "pretrained": True}
    state = {
        "cfg": {
            "model": {"name": "point_tracker_3d", "encoder": encoder},
            "train_dataset": {"resolution": [384, 512]},
        },
        "weight": reference.state_dict(),
    }
    model = _build_from_checkpoint(_Models(), state, "ckpt.pth")
    torch.testing.assert_close(model.weight, reference.weight)
    assert encoder["pretrained"] is True
    del state["weight"]["bias"]
    with pytest.raises(RuntimeError, match="missing: bias"):
        _build_from_checkpoint(_Models(), state, "incomplete.pth")


def test_weight_loading_reports_missing_unexpected_and_mismatched():
    model = nn.Linear(2, 3)
    good = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
    _load_weights(model, good, "ckpt.pth")
    assert torch.allclose(model.weight, torch.zeros_like(model.weight))

    missing = {k: v for k, v in good.items() if k != "bias"}
    with pytest.raises(RuntimeError, match=r"missing: bias"):
        _load_weights(model, missing, "ckpt.pth")

    unexpected = dict(good, extra_head=torch.zeros(1))
    with pytest.raises(RuntimeError, match=r"unexpected: extra_head"):
        _load_weights(model, unexpected, "ckpt.pth")

    mismatched = dict(good, bias=torch.zeros(7))
    with pytest.raises(RuntimeError, match=r"mismatched: bias"):
        _load_weights(model, mismatched, "ckpt.pth")


def test_weight_loading_rejects_a_non_state_dict():
    with pytest.raises(RuntimeError, match="not a state dict"):
        _load_weights(nn.Linear(2, 3), [1, 2, 3], "ckpt.pth")


def test_repo_path_must_look_like_a_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv(REPO_PATH_ENV, str(tmp_path))
    with pytest.raises(ImportError, match="point_tracker_3d.py"):
        TAPIP3DConfig().build()


def test_repo_path_config_beats_the_environment(tmp_path, monkeypatch):
    """Resolution only — building further would import the checkout into this process."""
    env_checkout = _fake_checkout(tmp_path / "from-env")
    configured = _fake_checkout(tmp_path / "from-config")
    monkeypatch.setenv(REPO_PATH_ENV, str(env_checkout))
    assert _resolve_repo_path(TAPIP3DConfig(repo_path=str(configured))) == configured
    assert _resolve_repo_path(TAPIP3DConfig()) == env_checkout


def _fake_checkout(root: Path) -> Path:
    (root / "models").mkdir(parents=True)
    (root / "models" / "point_tracker_3d.py").write_text("")
    return root.resolve()
