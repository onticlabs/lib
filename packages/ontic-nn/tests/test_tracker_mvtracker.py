"""Contract tests for the MVTracker adapter (no upstream package, no weights).

A fake stands in for ``EvaluationPredictor`` so the real conversion code — axis permutation,
camera inversion, pixel-centre intrinsics, depth masking and scaling, query encoding, scene
normalization and its inverse, batching — is what gets exercised and checked.
"""

from __future__ import annotations

import importlib
import sys

import pytest
import torch
import torch.nn as nn

from ontic_nn.trackers import TRACKERS, GeometrySequence, PointQueries
from ontic_nn.trackers.mvtracker import (
    MVTRACKER_REPO_URL,
    MVTracker,
    MVTrackerConfig,
)

HEIGHT, WIDTH = 32, 48
FRAMES, VIEWS = 8, 3
EYES = ((2.0, 0.0, 0.3), (0.0, 2.2, 0.3), (-1.8, -0.4, 0.5))


# --------------------------------------------------------------------- fakes
class FakePredictor(nn.Module):
    """Records the upstream call and echoes the queries back as constant trajectories."""

    def __init__(self, *, visibility: float = 0.25, thresholded_only: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.calls: list[dict] = []
        self.visibility = visibility
        self.thresholded_only = thresholded_only

    def forward(self, *, rgbs, depths, query_points_3d, intrs, extrs, **kwargs):
        self.calls.append(
            {
                "rgbs": rgbs.detach().clone(),
                "depths": depths.detach().clone(),
                "query_points_3d": query_points_3d.detach().clone(),
                "intrs": intrs.detach().clone(),
                "extrs": extrs.detach().clone(),
                "kwargs": kwargs,
            }
        )
        frames = rgbs.shape[2]
        xyz = query_points_3d[:, :, 1:]
        traj = xyz[:, None].expand(1, frames, xyz.shape[1], 3) + self.weight
        vis = torch.full(traj.shape[:3], self.visibility)
        if self.thresholded_only:
            return {"traj_e": traj, "vis_e": vis > 0.5}
        return {"traj_e": traj, "vis_e": vis > 0.5, "vis_e_as_prob": vis}


class ConstantPredictor(FakePredictor):
    """Returns the (zero-based) call index in every coordinate, to check batch assembly."""

    def forward(self, *, rgbs, depths, query_points_3d, intrs, extrs, **kwargs):
        index = len(self.calls)
        super().forward(
            rgbs=rgbs, depths=depths, query_points_3d=query_points_3d, intrs=intrs, extrs=extrs
        )
        shape = (1, rgbs.shape[2], query_points_3d.shape[1])
        return {
            "traj_e": torch.full((*shape, 3), float(index)),
            "vis_e": torch.ones(shape, dtype=torch.bool),
            "vis_e_as_prob": torch.full(shape, 0.75),
        }


# ------------------------------------------------------------------ fixtures
def look_at(eye) -> torch.Tensor:
    """Camera-to-world for a camera at ``eye`` looking at the origin (x right, y down, z fwd)."""
    eye = torch.tensor(eye, dtype=torch.float32)
    forward = -eye / eye.norm()
    up = torch.tensor([0.0, 0.0, 1.0])
    right = torch.linalg.cross(forward, up)
    right = right / right.norm()
    down = torch.linalg.cross(forward, right)
    c2w = torch.eye(4)
    c2w[:3, :3] = torch.stack([right, down, forward], dim=1)
    c2w[:3, 3] = eye
    return c2w


def make_geometry(batch: int = 1, *, invalid_box: bool = False) -> GeometrySequence:
    extrinsics = torch.stack([look_at(eye) for eye in EYES])  # (V,4,4)
    extrinsics = extrinsics[None, None].expand(batch, FRAMES, VIEWS, 4, 4).contiguous()

    focal = float(WIDTH)  # square pixels, principal point at the image centre
    intrinsics = torch.tensor(
        [[focal / WIDTH, 0.0, 0.5], [0.0, focal / HEIGHT, 0.5], [0.0, 0.0, 1.0]]
    )
    intrinsics = intrinsics[None, None, None].expand(batch, FRAMES, VIEWS, 3, 3).contiguous()

    view = torch.arange(VIEWS, dtype=torch.float32)[None, :, None, None]
    time = torch.arange(FRAMES, dtype=torch.float32)[:, None, None, None]
    depth = (2.0 + 0.1 * view + 0.01 * time).expand(FRAMES, VIEWS, HEIGHT, WIDTH)
    depth = depth[None].expand(batch, FRAMES, VIEWS, HEIGHT, WIDTH).contiguous()
    depth = depth + 0.05 * torch.arange(batch, dtype=torch.float32)[:, None, None, None, None]

    valid = None
    if invalid_box:
        valid = torch.ones_like(depth, dtype=torch.bool)
        valid[..., 4:9, 6:13] = False
    return GeometrySequence(
        depth=depth,
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        depth_valid=valid,
        frame_ids=("world",) * batch,
        units=("meters",) * batch,
        provenance={"depth": "synthetic"},
    )


def make_images(batch: int = 1) -> torch.Tensor:
    """``images[b,t,v]`` is the constant ``(t + 10v + 100b) / 1000``, so permutations show up."""
    view = torch.arange(VIEWS, dtype=torch.float32)[None, :]
    time = torch.arange(FRAMES, dtype=torch.float32)[:, None]
    offset = 100.0 * torch.arange(batch, dtype=torch.float32)[:, None, None]
    plane = (time + 10.0 * view)[None] + offset
    return (
        (plane / 1000.0)[:, :, :, None, None, None]
        .expand(batch, FRAMES, VIEWS, 3, HEIGHT, WIDTH)
        .contiguous()
    )


def unproject(uv_pixel: torch.Tensor, geometry: GeometrySequence, *, view: int, time: int):
    """A pixel (integer centre) at the geometry's depth → its world position."""
    normalized = torch.stack(
        [(uv_pixel[0] + 0.5) / WIDTH, (uv_pixel[1] + 0.5) / HEIGHT, torch.ones(())]
    )
    k = geometry.intrinsics[0, time, view]
    c2w = geometry.extrinsics[0, time, view]
    depth = geometry.depth[0, time, view, int(uv_pixel[1]), int(uv_pixel[0])]
    camera = torch.linalg.inv(k) @ normalized * depth
    return c2w[:3, :3] @ camera + c2w[:3, 3]


def make_queries(
    geometry: GeometrySequence, times: tuple[int, ...], pixels: tuple[tuple[int, int], ...]
) -> PointQueries:
    batch = geometry.depth.shape[0]
    xyz = torch.stack(
        [
            torch.stack(
                [
                    unproject(torch.tensor(pixel, dtype=torch.float32), geometry, view=0, time=t)
                    for t, pixel in zip(times, pixels)
                ]
            )
            for _ in range(batch)
        ]
    )
    count = len(times)
    return PointQueries(
        ids=torch.arange(count, dtype=torch.int64).expand(batch, count).contiguous(),
        time=torch.tensor(times, dtype=torch.int64).expand(batch, count).contiguous(),
        xyz_world=xyz,
    )


def build(**overrides) -> tuple[MVTracker, FakePredictor]:
    predictor = overrides.pop("predictor", None) or FakePredictor()
    overrides.setdefault("image_size", (HEIGHT, WIDTH))  # keep the resize an identity
    overrides.setdefault("bidirectional", False)  # these tests isolate a single native pass
    return MVTracker(MVTrackerConfig(**overrides), model=predictor), predictor


def camera_centers(extrs: torch.Tensor) -> torch.Tensor:
    """World camera centres of ``(V,3,4)`` world-to-camera matrices."""
    rotation, translation = extrs[:, :3, :3], extrs[:, :3, 3]
    return -(rotation.transpose(-1, -2) @ translation.unsqueeze(-1)).squeeze(-1)


# ------------------------------------------------------- registry and config
def test_registered_under_mvtracker_with_multiview_any_view_capabilities():
    assert TRACKERS["mvtracker"] is MVTrackerConfig
    capabilities = MVTrackerConfig.CAPABILITIES
    assert MVTracker.CAPABILITIES is capabilities
    assert capabilities.multiview and not capabilities.dense
    assert capabilities.arbitrary_query_times
    assert capabilities.execution_mode == "offline"
    assert capabilities.visibility_scope == "any_view"
    tracker, _ = build()
    assert tracker.capabilities is capabilities


def test_defaults_match_the_released_checkpoint_configuration():
    cfg = MVTrackerConfig()
    assert (cfg.model_dir, cfg.checkpoint_file) == (
        "ethz-vlg/mvtracker",
        "mvtracker_200000_june2025.pth",
    )
    # hubconf._build_model
    assert (cfg.sliding_window_len, cfg.stride, cfg.fmaps_dim) == (12, 4, 128)
    assert (cfg.num_heads, cfg.hidden_size) == (6, 256)
    assert (cfg.space_depth, cfg.time_depth, cfg.num_virtual_tracks) == (6, 6, 64)
    assert (cfg.corr_n_groups, cfg.corr_n_levels, cfg.corr_neighbors) == (1, 4, 16)
    assert cfg.corr_add_neighbor_offset and not cfg.corr_add_neighbor_xyz
    assert not cfg.corr_filter_invalid_depth
    # hubconf.mvtracker_predictor
    assert (cfg.grid_size, cfg.n_grids_per_view) == (4, 1)
    assert (cfg.local_grid_size, cfg.local_extent) == (18, 50)
    assert (cfg.visibility_threshold, cfg.n_iters) == (0.5, 6)
    assert not cfg.single_point
    assert cfg.freeze_tracker and cfg.allow_download


def test_flash_attention_setting_follows_installed_accelerators():
    assert MVTrackerConfig(use_flash_attention=True).resolved_use_flash_attention() is True
    assert MVTrackerConfig(use_flash_attention=False).resolved_use_flash_attention() is False
    auto = MVTrackerConfig().resolved_use_flash_attention()
    assert auto is torch.cuda.is_available()


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"scene_normalization": "manual"}, "positive scene_scale"),
        ({"scene_normalization": "sim3"}, "scene_normalization must be"),
        ({"image_size": (0, 8)}, "positive \\(height, width\\)"),
        ({"n_iters": 0}, "n_iters"),
        ({"grid_size": -1}, "grid_size"),
        ({"visibility_threshold": 1.5}, "visibility_threshold"),
        ({"rgb_scale": 0.0}, "rgb_scale"),
        ({"sliding_window_len": 1}, "sliding_window_len"),
        ({"scene_target_radius": 0.0}, "scene_target_radius"),
        ({"long_side": 0}, "long_side"),
    ],
)
def test_invalid_config_is_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        MVTracker(MVTrackerConfig(**overrides), model=FakePredictor())


def test_long_side_reproduces_the_released_interp_shape():
    tracker, _ = build(image_size=None, long_side=512)
    assert tracker._target_size(480, 640) == (384, 512)  # upstream interp_shape=(384, 512)
    assert tracker._target_size(1080, 1920) == (288, 512)
    pinned, _ = build(image_size=(240, 320))
    assert pinned._target_size(480, 640) == (240, 320)


# ----------------------------------------------------------- upstream import
def test_build_without_upstream_package_names_the_extra_and_repo(monkeypatch):
    monkeypatch.setitem(sys.modules, "mvtracker", None)
    with pytest.raises(ImportError, match=r"ontic-nn\[mvtracker\]") as excinfo:
        MVTrackerConfig().build()
    assert MVTRACKER_REPO_URL in str(excinfo.value)


def test_shadowed_top_level_mvtracker_is_rejected_not_overwritten(monkeypatch):
    adapter = sys.modules["ontic_nn.trackers.mvtracker"]
    monkeypatch.setitem(sys.modules, "mvtracker", adapter)
    with pytest.raises(ImportError, match="resolves to this adapter"):
        MVTrackerConfig().build()
    assert sys.modules["mvtracker"] is adapter  # the guard must not replace it


# -------------------------------------------------------- input conversions
def test_upstream_receives_bvt_layout_with_rgb_scaled_to_0_255():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0, 2), ((10, 7), (30, 20)))
    tracker, predictor = build(scene_normalization="none")
    tracker(images, queries, geometry=geometry)

    call = predictor.calls[0]
    assert call["rgbs"].shape == (1, VIEWS, FRAMES, 3, HEIGHT, WIDTH)
    assert call["depths"].shape == (1, VIEWS, FRAMES, 1, HEIGHT, WIDTH)
    assert call["intrs"].shape == (1, VIEWS, FRAMES, 3, 3)
    assert call["extrs"].shape == (1, VIEWS, FRAMES, 3, 4)
    assert call["query_points_3d"].shape == (1, 2, 4)

    for time in range(FRAMES):
        for view in range(VIEWS):
            expected = (time + 10.0 * view) / 1000.0 * 255.0
            assert torch.allclose(
                call["rgbs"][0, view, time], torch.full((3, HEIGHT, WIDTH), expected), atol=1e-3
            )
            assert torch.allclose(
                call["depths"][0, view, time, 0],
                torch.full((HEIGHT, WIDTH), 2.0 + 0.1 * view + 0.01 * time),
                atol=1e-5,
            )


def test_queries_are_encoded_as_t_x_y_z_at_the_query_time():
    geometry, images = make_geometry(), make_images()
    times, pixels = (0, 2, 5), ((10, 7), (30, 20), (1, 1))
    queries = make_queries(geometry, times, pixels)
    tracker, predictor = build(scene_normalization="none")
    tracker(images, queries, geometry=geometry)

    encoded = predictor.calls[0]["query_points_3d"][0]
    assert torch.equal(encoded[:, 0], torch.tensor(times, dtype=torch.float32))
    assert torch.allclose(encoded[:, 1:], queries.xyz_world[0], atol=1e-5)


def test_extrinsics_are_inverted_to_world_to_camera():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0,), ((10, 7),))
    tracker, predictor = build(scene_normalization="none")
    tracker(images, queries, geometry=geometry)

    extrs = predictor.calls[0]["extrs"][0, :, 0]  # (V,3,4) at t=0
    centers = geometry.extrinsics[0, 0, :, :3, 3]
    # the world camera centre must map to the camera-space origin
    mapped = (extrs[:, :3, :3] @ centers.unsqueeze(-1)).squeeze(-1) + extrs[:, :3, 3]
    assert torch.allclose(mapped, torch.zeros(VIEWS, 3), atol=1e-5)
    # and the recovered centres must be the originals, not the c2w translation column
    assert torch.allclose(camera_centers(extrs), centers, atol=1e-5)
    # the scene in front of each camera keeps positive camera z
    point = queries.xyz_world[0, 0]
    camera = (extrs[:, :3, :3] @ point).squeeze(-1) + extrs[:, :3, 3]
    assert (camera[:, 2] > 0).all()


def test_intrinsics_use_integer_pixel_centres():
    geometry, images = make_geometry(), make_images()
    pixel = (10, 7)
    queries = make_queries(geometry, (0,), (pixel,))
    tracker, predictor = build(scene_normalization="none")
    tracker(images, queries, geometry=geometry)

    call = predictor.calls[0]
    k = call["intrs"][0, 0, 0]
    assert torch.allclose(k[0, 0], torch.tensor(float(WIDTH)))  # fx_norm * W
    assert torch.allclose(k[1, 1], torch.tensor(float(WIDTH)))  # fy_norm * H, square pixels
    # a normalized principal point of 0.5 is the integer-centre midpoint (W-1)/2, (H-1)/2
    assert torch.allclose(k[0, 2], torch.tensor((WIDTH - 1) / 2))
    assert torch.allclose(k[1, 2], torch.tensor((HEIGHT - 1) / 2))

    # the query built from Ontic's (x+0.5)/W centre must land on the integer pixel upstream
    extrs = call["extrs"][0, 0, 0]
    point = call["query_points_3d"][0, 0, 1:]
    camera = extrs[:3, :3] @ point + extrs[:3, 3]
    projected = (k @ camera)[:2] / (k @ camera)[2]
    assert torch.allclose(projected, torch.tensor(pixel, dtype=torch.float32), atol=1e-3)


def test_invalid_depth_is_passed_as_the_zero_sentinel():
    geometry, images = make_geometry(invalid_box=True), make_images()
    queries = make_queries(geometry, (0,), ((30, 20),))
    tracker, predictor = build(scene_normalization="none")
    tracker(images, queries, geometry=geometry)

    depths = predictor.calls[0]["depths"][0, 1, 3, 0]
    assert torch.equal(depths[4:9, 6:13], torch.zeros(5, 7))
    assert (depths[:4] > 0).all() and (depths[9:] > 0).all()


def test_resize_is_applied_before_the_upstream_predictor():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0,), ((10, 7),))
    tracker, predictor = build(image_size=(16, 24), scene_normalization="none")
    out = tracker(images, queries, geometry=geometry)

    call = predictor.calls[0]
    assert call["rgbs"].shape[-2:] == (16, 24)
    assert call["depths"].shape[-2:] == (16, 24)
    # normalized intrinsics are resolution independent: pixel focals follow the new grid
    assert torch.allclose(call["intrs"][0, 0, 0, 0, 0], torch.tensor(24.0))
    assert torch.allclose(call["intrs"][0, 0, 0, 0, 2], torch.tensor(11.5))
    assert out.metadata["input_resolution"] == (16, 24)


# ------------------------------------------------------- scene normalization
def test_camera_radius_normalization_rescales_and_is_exactly_inverted():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0, 3), ((10, 7), (30, 20)))
    tracker, predictor = build()  # scene_normalization="camera_radius" is the default
    out = tracker(images, queries, geometry=geometry)

    call = predictor.calls[0]
    scale, translation = out.metadata["scene_normalization"]["per_batch_scale_translation"][0]
    assert abs(scale - 1.0) > 0.2, "the normalization must actually rescale this scene"

    # the model saw normalized queries ...
    sent = call["query_points_3d"][0, :, 1:]
    assert not torch.allclose(sent, queries.xyz_world[0], atol=1e-3)
    assert torch.allclose(sent, queries.xyz_world[0] * scale + torch.tensor(translation), atol=1e-4)
    # ... and the echoed trajectories come back in the original frame
    assert torch.allclose(out.tracks_world[0, 0], queries.xyz_world[0], atol=1e-3, rtol=1e-3)


def test_normalized_depth_and_cameras_stay_consistent_at_the_target_radius():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0,), ((10, 7),))
    target = 6.3
    tracker, predictor = build(scene_target_radius=target)
    tracker(images, queries, geometry=geometry)

    call = predictor.calls[0]
    depths = call["depths"][0, :, 0, 0]  # (V,H,W) at t=0
    k = call["intrs"][0, :, 0]
    extrs = call["extrs"][0, :, 0]

    ys, xs = torch.meshgrid(
        torch.arange(HEIGHT, dtype=torch.float32),
        torch.arange(WIDTH, dtype=torch.float32),
        indexing="ij",
    )
    pixels = torch.stack([xs, ys, torch.ones_like(xs)], dim=-1)
    camera = torch.einsum("vij,hwj->vhwi", torch.linalg.inv(k), pixels) * depths[..., None]
    rotation = extrs[:, :3, :3].transpose(-1, -2)
    world = torch.einsum("vij,vhwj->vhwi", rotation, camera - extrs[:, None, None, :3, 3])

    # depth, intrinsics and extrinsics describe one consistent normalized scene, centred on
    # the first-frame depth centroid, with the cameras at the model's working radius
    assert torch.allclose(world.reshape(-1, 3).mean(0), torch.zeros(3), atol=1e-3)
    distances = camera_centers(extrs).norm(dim=-1)
    assert pytest.approx(target, abs=1e-3) == float(distances.median())


def test_manual_and_none_scene_modes():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0,), ((10, 7),))

    tracker, predictor = build(
        scene_normalization="manual", scene_scale=2.0, scene_translation=(1.0, -2.0, 0.5)
    )
    out = tracker(images, queries, geometry=geometry)
    sent = predictor.calls[0]["query_points_3d"][0, :, 1:]
    assert torch.allclose(
        sent, queries.xyz_world[0] * 2.0 + torch.tensor([1.0, -2.0, 0.5]), atol=1e-5
    )
    assert torch.allclose(predictor.calls[0]["depths"][0, 0, 0, 0, 0, 0], torch.tensor(4.0))
    assert torch.allclose(out.tracks_world[0, 0], queries.xyz_world[0], atol=1e-4)

    plain, plain_predictor = build(scene_normalization="none")
    plain(images, queries, geometry=geometry)
    assert torch.allclose(
        plain_predictor.calls[0]["query_points_3d"][0, :, 1:], queries.xyz_world[0], atol=1e-5
    )


def test_camera_radius_rejects_geometry_it_cannot_normalize():
    geometry, images = make_geometry(), make_images()
    geometry.depth_valid = torch.zeros_like(geometry.depth, dtype=torch.bool)
    queries = make_queries(make_geometry(), (0,), ((10, 7),))
    tracker, _ = build()
    with pytest.raises(ValueError, match="no valid depth"):
        tracker(images, queries, geometry=geometry)


# --------------------------------------------------------------- output side
def test_valid_marks_only_frames_at_or_after_the_query_time():
    geometry, images = make_geometry(), make_images()
    times = (0, 3, 5)
    queries = make_queries(geometry, times, ((10, 7), (30, 20), (1, 1)))
    tracker, _ = build(scene_normalization="none")
    out = tracker(images, queries, geometry=geometry)

    frame = torch.arange(FRAMES)[:, None]
    assert torch.equal(out.valid[0], frame >= torch.tensor(times))
    assert out.metadata["valid_rule"] == "frame index >= query time"


def test_visibility_is_the_native_any_view_probability():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0,), ((10, 7),))
    tracker, _ = build(scene_normalization="none", predictor=FakePredictor(visibility=0.25))
    out = tracker(images, queries, geometry=geometry)

    assert out.visibility_scope == "any_view"
    assert out.visibility_per_view is None
    assert torch.allclose(out.visibility, torch.full((1, FRAMES, 1), 0.25))
    assert out.metadata["visibility"]["source"] == "vis_e_as_prob"
    assert out.metadata["visibility"]["upstream_threshold"] == 0.5


def test_thresholded_only_visibility_is_rejected():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0,), ((10, 7),))
    tracker, _ = build(scene_normalization="none", predictor=FakePredictor(thresholded_only=True))
    with pytest.raises(RuntimeError, match="vis_e_as_prob"):
        tracker(images, queries, geometry=geometry)


def test_ids_and_metadata_are_preserved():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0, 2), ((10, 7), (30, 20)))
    queries.ids = torch.tensor([[41, 17]], dtype=torch.int64)
    tracker, _ = build(scene_normalization="none")
    out = tracker(images, queries, geometry=geometry)

    assert torch.equal(out.ids, queries.ids)
    assert out.metadata["units"] == ("meters",)
    assert out.metadata["frame_ids"] == ("world",)
    assert out.metadata["geometry_provenance"] == {"depth": "synthetic"}
    assert out.metadata["upstream"]["api_revision"].startswith("ceea8ad")
    assert out.metadata["rgb_scale"] == 255.0


def test_empty_queries_short_circuit_without_calling_upstream():
    geometry, images = make_geometry(), make_images()
    queries = PointQueries(
        ids=torch.zeros(1, 0, dtype=torch.int64),
        time=torch.zeros(1, 0, dtype=torch.int64),
        xyz_world=torch.zeros(1, 0, 3),
    )
    tracker, predictor = build(scene_normalization="none")
    out = tracker(images, queries, geometry=geometry)
    assert out.tracks_world.shape == (1, FRAMES, 0, 3)
    assert out.visibility_scope == "any_view"
    assert predictor.calls == []


# -------------------------------------------------------------- native limits
def test_batches_are_looped_one_sequence_at_a_time():
    geometry, images = make_geometry(batch=2), make_images(batch=2)
    queries = make_queries(geometry, (0, 1), ((10, 7), (30, 20)))
    tracker, predictor = build(scene_normalization="none", predictor=ConstantPredictor())
    out = tracker(images, queries, geometry=geometry)

    assert len(predictor.calls) == 2, "upstream only supports batch size 1"
    for call in predictor.calls:
        assert call["rgbs"].shape[0] == 1 and call["query_points_3d"].shape[0] == 1
    # each sequence is its own call, in order
    assert torch.allclose(predictor.calls[0]["rgbs"][0, 0, 0, 0, 0, 0], torch.tensor(0.0))
    assert torch.allclose(predictor.calls[1]["rgbs"][0, 0, 0, 0, 0, 0], torch.tensor(25.5))
    assert torch.allclose(out.tracks_world[0], torch.zeros_like(out.tracks_world[0]))
    assert torch.allclose(out.tracks_world[1], torch.ones_like(out.tracks_world[1]))


def test_short_clips_are_rejected_instead_of_returning_zero_tracks():
    geometry, images = make_geometry(), make_images()
    short = GeometrySequence(
        depth=geometry.depth[:, :6],
        extrinsics=geometry.extrinsics[:, :6],
        intrinsics=geometry.intrinsics[:, :6],
    )
    queries = make_queries(short, (0,), ((10, 7),))
    tracker, predictor = build(scene_normalization="none")
    with pytest.raises(ValueError, match="sliding window"):
        tracker(images[:, :6], queries, geometry=short)
    assert predictor.calls == []


def test_late_queries_without_the_support_grid_are_rejected():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (5,), ((10, 7),))
    tracker, _ = build(scene_normalization="none", grid_size=0)
    with pytest.raises(ValueError, match="sliding window"):
        tracker(images, queries, geometry=geometry)
    # the default support grid contributes queries at t=0, so the same clip is fine
    with_support, _ = build(scene_normalization="none")
    with_support(images, queries, geometry=geometry)


def test_monocular_and_mismatched_inputs_are_rejected():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0,), ((10, 7),))
    tracker, _ = build(scene_normalization="none")
    with pytest.raises(ValueError, match="batch/time/view"):
        tracker(images[:, :, :1], queries, geometry=geometry)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        tracker(images + 2.0, queries, geometry=geometry)


# ------------------------------------------------------------------ freezing
def test_frozen_tracker_stays_in_eval_and_detached_under_train():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0,), ((10, 7),))

    tracker, predictor = build(scene_normalization="none")
    assert not predictor.weight.requires_grad and not predictor.training
    tracker.train(True)
    assert tracker.training and not tracker.model.training
    assert not tracker(images, queries, geometry=geometry).tracks_world.requires_grad


# ------------------------------------------------- build against upstream API
CORE_STUB = """
import torch.nn as nn


def _knn_torch(*args, **kwargs):
    raise NotImplementedError


knn = _knn_torch


class MVTracker(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.vis_predictor = nn.Linear(2, 1)
"""

PREDICTOR_STUB = """
import torch.nn as nn


class EvaluationPredictor(nn.Module):
    def __init__(self, multiview_model, **kwargs):
        super().__init__()
        self.model = multiview_model
        self.kwargs = kwargs
        self.model.eval()
"""


@pytest.fixture
def stub_upstream(tmp_path, monkeypatch):
    """Install a minimal on-disk `mvtracker` package exposing the upstream constructor API."""

    def install(predictor: str = PREDICTOR_STUB):
        sources = {
            "mvtracker/__init__.py": "",
            "mvtracker/models/__init__.py": "",
            "mvtracker/models/core/__init__.py": "",
            "mvtracker/models/core/mvtracker/__init__.py": "",
            "mvtracker/models/core/mvtracker/mvtracker.py": CORE_STUB,
            "mvtracker/models/evaluation_predictor_3dpt.py": predictor,
        }
        for name, source in sources.items():
            path = tmp_path / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        monkeypatch.syspath_prepend(str(tmp_path))
        importlib.invalidate_caches()

    yield install
    for name in [m for m in sys.modules if m == "mvtracker" or m.startswith("mvtracker.")]:
        del sys.modules[name]


def test_build_forwards_the_released_kwargs_to_the_upstream_constructors(stub_upstream):
    stub_upstream()
    tracker = MVTrackerConfig(model_dir="").build()  # no repo id → random init, no download

    core = tracker.model.model.kwargs
    assert core["sliding_window_len"] == 12 and core["stride"] == 4
    assert (core["fmaps_dim"], core["hidden_size"], core["num_heads"]) == (128, 256, 6)
    assert (core["space_depth"], core["time_depth"]) == (6, 6)
    assert core["num_virtual_tracks"] == 64 and core["add_space_attn"]
    assert (core["corr_n_groups"], core["corr_n_levels"], core["corr_neighbors"]) == (1, 4, 16)
    assert core["corr_add_neighbor_offset"] and not core["corr_add_neighbor_xyz"]
    assert not core["corr_filter_invalid_depth"]
    # upstream's in-forward normalization is broken at the pinned revision and never enabled
    assert core["normalize_scene_in_fwd_pass"] is False

    predictor = tracker.model.kwargs
    assert predictor["interp_shape"] is None  # the adapter owns the resize
    assert predictor["sift_size"] == 0 and predictor["num_uniformly_sampled_pts"] == 0
    assert (predictor["grid_size"], predictor["n_grids_per_view"]) == (4, 1)
    assert (predictor["local_grid_size"], predictor["local_extent"]) == (18, 50)
    assert predictor["n_iters"] == 6 and predictor["visibility_threshold"] == 0.5
    assert predictor["single_point"] is False

    assert tracker.checkpoint_info["path"] is None
    assert tracker.upstream_info["knn_backend"] == "_knn_torch"
    assert tracker.upstream_info["api_revision"].startswith("ceea8ad")
    assert not tracker.model.model.vis_predictor.weight.requires_grad  # freeze_tracker default


def test_checkpoint_is_unwrapped_and_unexpected_keys_are_rejected(stub_upstream, tmp_path):
    stub_upstream()
    weight = torch.full((1, 2), 0.25)
    bias = torch.tensor([0.5])
    path = tmp_path / "mvtracker_200000_june2025.pth"
    torch.save(
        {
            "model": {"model.vis_predictor.weight": weight, "model.vis_predictor.bias": bias},
            "optimizer": {},
            "total_steps": 200000,
        },
        path,
    )
    tracker = MVTrackerConfig(checkpoint_path=str(path)).build()
    assert tracker.checkpoint_info["missing_keys"] == 0
    assert torch.equal(tracker.model.model.vis_predictor.weight, weight)
    assert torch.equal(tracker.model.model.vis_predictor.bias, bias)

    ghost = tmp_path / "ghost.pth"
    torch.save({"vis_predictor.weight": weight, "vis_predictor.bias": bias, "ghost": bias}, ghost)
    with pytest.raises(RuntimeError, match="keys the model does not define"):
        MVTrackerConfig(checkpoint_path=str(ghost)).build()


def test_offline_checkpoint_miss_is_reported(stub_upstream, tmp_path):
    pytest.importorskip("huggingface_hub")
    stub_upstream()
    cfg = MVTrackerConfig(cache_dir=str(tmp_path / "hf"), allow_download=False)
    with pytest.raises(RuntimeError, match="allow_download"):
        cfg.build()


def test_missing_visualizer_dependency_names_the_real_culprit(stub_upstream, monkeypatch):
    monkeypatch.setitem(sys.modules, "moviepy.editor", None)
    stub_upstream("import moviepy.editor  # noqa: F401\n")
    with pytest.raises(ImportError, match="moviepy") as excinfo:
        MVTrackerConfig(model_dir="").build()
    assert "mvtracker` package is installed" in str(excinfo.value)


@pytest.mark.skipif(importlib.util.find_spec("mvtracker") is None, reason="mvtracker not installed")
def test_real_upstream_predictor_runs_a_forward():
    """Opt-in: exercises the released code path with random weights (no checkpoint)."""
    cfg = MVTrackerConfig(
        model_dir="",
        image_size=(HEIGHT, WIDTH),
        use_flash_attention=False,
        space_depth=1,
        time_depth=1,
        hidden_size=64,
        num_heads=2,
        num_virtual_tracks=4,
        corr_n_levels=2,  # keep enough context pixels for KNN on this tiny CPU grid
        corr_neighbors=4,
        n_iters=1,
        grid_size=2,
    )
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0, 2), ((10, 7), (30, 20)))
    out = cfg.build()(images, queries, geometry=geometry)
    assert out.tracks_world.shape == (1, FRAMES, 2, 3)
    assert torch.isfinite(out.tracks_world).all()
    assert out.visibility_scope == "any_view"


def test_unfrozen_tracker_trains_and_propagates_gradients():
    geometry, images = make_geometry(), make_images()
    queries = make_queries(geometry, (0,), ((10, 7),))

    tracker, predictor = build(scene_normalization="none", freeze_tracker=False)
    assert predictor.weight.requires_grad
    tracker.train(True)
    assert tracker.model.training
    out = tracker(images, queries, geometry=geometry)
    assert out.tracks_world.requires_grad
    out.tracks_world.sum().backward()
    assert predictor.weight.grad is not None
