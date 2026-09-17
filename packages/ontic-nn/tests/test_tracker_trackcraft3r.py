"""TrackCraft3R adapter tests against a stand-in that mirrors the released predictor.

The real ``WanSceneFlowPredictor`` needs a Wan2.1 base model and a CUDA GPU, so these tests
substitute a predictor that reproduces the parts of upstream's contract the adapter has to get
right: the mandatory ``depth_map`` / ``extrinsics_w2c`` assert, the ``np.arange`` unprojection
grid with no half-pixel offset, the frame-0 camera output frame, and the
``query_uv * W_out/orig_w`` → ``astype(int)`` sampling with its out-of-bounds drop.

Because the stand-in holds the reference frame's own point cloud static, the correct world-frame
answer is exactly ``queries.xyz_world`` at every timestep. That makes the round-trip tests fail
if the principal point is off by half a pixel, if the wrong frame's pose is used, or if anything
rescales the trajectories.
"""

from __future__ import annotations

import math
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from ontic_nn.trackers import TRACKERS, GeometrySequence, PointQueries
from ontic_nn.trackers import trackcraft3r as tc3r
from ontic_nn.trackers.trackcraft3r import TrackCraft3R, TrackCraft3RConfig

H_MODEL, W_MODEL = 32, 48  # multiples of 16, as the Wan2.1 VAE/DiT stack requires
T_CLIP = 3
FX_N, FY_N, CX_N, CY_N = 0.9, 1.1, 0.5, 0.5
QUERY_PIXELS = ((0, 0), (W_MODEL - 1, H_MODEL - 1), (5, 7), (20, 13))


# ---------------------------------------------------------------------------
# Scene / query construction
# ---------------------------------------------------------------------------
def _rot_y(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


def _c2w(index: int, *, camera: str, scale: float) -> torch.Tensor:
    m = torch.eye(4, dtype=torch.float64)
    if camera == "identity":
        return m.float()
    angle = 0.3 + (0.15 * index if camera == "moving" else 0.0)
    m[:3, :3] = _rot_y(angle)
    offset = torch.tensor([0.7, -0.4, 0.2], dtype=torch.float64)
    if camera == "moving":
        offset = offset + index * torch.tensor([0.05, -0.02, 0.03], dtype=torch.float64)
    m[:3, 3] = offset * scale
    return m.float()


def make_scene(
    *,
    b: int = 1,
    t: int = T_CLIP,
    v: int = 1,
    camera: str = "static",
    scale: float = 1.0,
    seed: int = 0,
    depth_hw: tuple[int, int] = (H_MODEL, W_MODEL),
    image_hw: tuple[int, int] = (H_MODEL, W_MODEL),
):
    """RGB plus a valid ``GeometrySequence``; depth and camera centres both carry ``scale``."""
    g = torch.Generator().manual_seed(seed)
    depth = (2.0 + torch.rand(b, t, v, *depth_hw, generator=g)) * scale
    images = torch.rand(b, t, v, 3, *image_hw, generator=g)
    k = torch.zeros(b, t, v, 3, 3)
    k[..., 0, 0], k[..., 1, 1] = FX_N, FY_N
    k[..., 0, 2], k[..., 1, 2], k[..., 2, 2] = CX_N, CY_N, 1.0
    extrinsics = torch.stack([_c2w(j, camera=camera, scale=scale) for j in range(t)])
    geometry = GeometrySequence(
        depth=depth,
        extrinsics=extrinsics[None, :, None].repeat(b, 1, v, 1, 1),
        intrinsics=k,
        frame_ids=("scene",) * b,
        units=("meters",) * b,
        provenance={"depth": "synthetic"},
    )
    geometry.validate()
    return images, geometry


def _normalized_uv(pixels=QUERY_PIXELS) -> torch.Tensor:
    """Ontic's edge-origin pixel centres ``(x + 0.5) / W`` on the model grid."""
    return torch.tensor(
        [[(x + 0.5) / W_MODEL, (y + 0.5) / H_MODEL] for x, y in pixels], dtype=torch.float32
    ).reshape(-1, 2)


def make_queries(geometry: GeometrySequence, *, pixels=QUERY_PIXELS, time_index: int = 0):
    b = geometry.depth.shape[0]
    uv = _normalized_uv(pixels).expand(b, -1, -1).contiguous()
    n = uv.shape[1]
    time = torch.full((b, n), time_index, dtype=torch.int64)
    view = torch.zeros((b, n), dtype=torch.int64)
    return PointQueries.from_pixels(geometry, time, view, uv)


def lift_manually(geometry: GeometrySequence, pixels=QUERY_PIXELS) -> PointQueries:
    """Queries lifted by hand, so they can also sit on pixels with masked-out depth."""
    b, _, _, hd, wd = geometry.depth.shape
    uv = _normalized_uv(pixels)
    xyz = torch.zeros(b, len(pixels), 3)
    for i in range(b):
        c2w = geometry.extrinsics[i, 0, 0]
        for j, row in enumerate(uv):
            u, v = float(row[0]), float(row[1])
            x = min(int(u * wd), wd - 1)
            y = min(int(v * hd), hd - 1)
            d = float(geometry.depth[i, 0, 0, y, x])
            cam = torch.tensor([(u - CX_N) / FX_N, (v - CY_N) / FY_N, 1.0]) * d
            xyz[i, j] = c2w[:3, :3] @ cam + c2w[:3, 3]
    n = len(pixels)
    return PointQueries(
        ids=torch.arange(n, dtype=torch.int64).expand(b, -1).contiguous(),
        time=torch.zeros(b, n, dtype=torch.int64),
        xyz_world=xyz,
        source_view=torch.zeros(b, n, dtype=torch.int64),
        source_uv=uv.expand(b, -1, -1).contiguous(),
    )


# ---------------------------------------------------------------------------
# Stand-in predictors
# ---------------------------------------------------------------------------
class _FakePipe:
    """Stands in for ``WanVideoPipeline`` so ``_PredictorModule`` has modules to freeze."""

    def __init__(self) -> None:
        self.dit = nn.Linear(2, 2)
        self.vae = nn.Linear(2, 2)


class StaticScenePredictor:
    """Reproduces ``WanSceneFlowPredictor.predict``'s signature, geometry and query sampling.

    The "prediction" is frame 0's own unprojected point cloud, held static over the clip (plus an
    optional camera-0-space ``motion`` offset). Upstream undoes its percentile/centroid
    normalisation inside ``predict``, so this returns metric camera-0 coordinates directly and
    the adapter must not rescale them.
    """

    def __init__(self, *, motion=None, vis_dense=None, drop_last: bool = False) -> None:
        self.motion = motion
        self.vis_dense = vis_dense
        self.drop_last = drop_last
        self.calls: list[dict] = []
        self.pipe = _FakePipe()
        self._last_vis_dense = None
        self._last_oob_mask = None

    def _dense(self, images_pil, intrinsics, depth_map):
        h, w = depth_map.shape[1:]
        fx, fy, cx, cy = (float(x) for x in intrinsics)
        vv, uu = np.meshgrid(
            np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij"
        )
        d0 = depth_map[0].astype(np.float64)
        # Upstream's grid: integer pixel indices, no +0.5 anywhere.
        pts = np.stack([(uu - cx) / fx * d0, (vv - cy) / fy * d0, d0], axis=-1)
        traj = np.repeat(pts[None], len(images_pil), axis=0)  # (T,H,W,3), static scene
        if self.motion is not None:
            traj = traj + np.asarray(self.motion, dtype=np.float64)[:, None, None, :]
        return traj.transpose(3, 0, 1, 2)  # (3,T,H_out,W_out)

    def predict(
        self, images_pil, query_uv, visibility, intrinsics, depth_map=None, extrinsics_w2c=None
    ):
        assert depth_map is not None and extrinsics_w2c is not None, (
            "TrackCraft3R requires depth_map and extrinsics_w2c"
        )
        self.calls.append(
            {
                "num_frames": len(images_pil),
                "image_size": (images_pil[0].height, images_pil[0].width),
                "query_uv": np.array(query_uv, copy=True),
                "visibility": np.array(visibility, copy=True),
                "intrinsics": np.array(intrinsics, copy=True),
                "depth_map": np.array(depth_map, copy=True),
                "extrinsics_w2c": np.array(extrinsics_w2c, copy=True),
            }
        )
        traj3d = self._dense(images_pil, intrinsics, depth_map)
        self._last_vis_dense = self.vis_dense

        h_out, w_out = traj3d.shape[2], traj3d.shape[3]
        orig_h, orig_w = images_pil[0].height, images_pil[0].width
        scaled = np.asarray(query_uv, dtype=np.float64) * np.array([w_out / orig_w, h_out / orig_h])
        oob = (
            (scaled[:, 0] >= 0)
            & (scaled[:, 0] < w_out)
            & (scaled[:, 1] >= 0)
            & (scaled[:, 1] < h_out)
        )
        self._last_oob_mask = oob
        kept = scaled[oob][: -1 if self.drop_last else None]
        u_q, v_q = kept[:, 0].astype(int), kept[:, 1].astype(int)
        return traj3d[:, :, v_q, u_q].transpose(1, 2, 0).astype(np.float32)


class PixelIndexPredictor(StaticScenePredictor):
    """Returns ``[u, v, 1]`` at every pixel, so a track reveals which pixel was sampled."""

    def _dense(self, images_pil, intrinsics, depth_map):
        h, w = depth_map.shape[1:]
        vv, uu = np.meshgrid(
            np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij"
        )
        pts = np.stack([uu, vv, np.ones_like(uu)], axis=-1)
        return np.repeat(pts[None], len(images_pil), axis=0).transpose(3, 0, 1, 2)


def make_config(**kw) -> TrackCraft3RConfig:
    base = dict(height=H_MODEL, width=W_MODEL, num_frames=T_CLIP, device="cpu")
    base.update(kw)
    return TrackCraft3RConfig(**base)


def build(predictor, **kw) -> TrackCraft3R:
    return TrackCraft3R(make_config(**kw), predictor=predictor)


# ---------------------------------------------------------------------------
# Registration and declared restrictions
# ---------------------------------------------------------------------------
def test_registered_under_its_name():
    assert TRACKERS["trackcraft3r"] is TrackCraft3RConfig


def test_capabilities_declare_the_native_restrictions():
    for holder in (TrackCraft3RConfig, TrackCraft3R):
        caps = holder.CAPABILITIES
        assert caps.multiview is False
        assert caps.dense is True
        assert caps.arbitrary_query_times is False
        assert caps.execution_mode == "offline"
        assert caps.visibility_scope == "query_view"


def test_defaults_match_the_released_configuration():
    cfg = TrackCraft3RConfig()
    assert (cfg.height, cfg.width) == (480, 832)
    assert cfg.num_frames == 12
    assert cfg.allow_variable_clip_length is False
    assert cfg.model_id == "Wan-AI/Wan2.1-T2V-1.3B"
    assert cfg.hf_repo_id == "trackcraft3r/checkpoint"
    assert cfg.checkpoint_filename == "model.safetensors"
    assert cfg.lora_rank == 1024
    assert cfg.lora_target_modules == "q,k,v,o,ffn.0,ffn.2"
    assert cfg.diag_max_depth == 80.0


# ---------------------------------------------------------------------------
# Geometry round-trips
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("camera", ["identity", "static", "moving"])
def test_reference_cloud_round_trips_into_the_world_frame(camera):
    """A static reference cloud must come back as exactly the query XYZ, at every timestep."""
    images, geometry = make_scene(camera=camera)
    queries = make_queries(geometry)
    predictor = StaticScenePredictor()
    out = build(predictor).forward(images, queries, geometry=geometry)

    assert out.tracks_world.shape == (1, T_CLIP, len(QUERY_PIXELS), 3)
    expected = queries.xyz_world[:, None].expand(-1, T_CLIP, -1, -1)
    torch.testing.assert_close(out.tracks_world, expected, atol=1e-4, rtol=1e-4)
    assert out.valid.all()
    assert torch.equal(out.ids, queries.ids)


def test_moving_camera_does_not_leak_into_the_trajectory():
    """With a moving camera the per-frame poses differ, so only frame 0's c2w gives a fixed point."""
    images, geometry = make_scene(camera="moving")
    per_frame = geometry.extrinsics[0, :, 0]
    assert not torch.allclose(per_frame[0], per_frame[1], atol=1e-3)

    out = build(StaticScenePredictor()).forward(images, make_queries(geometry), geometry=geometry)
    spread = (out.tracks_world - out.tracks_world[:, :1]).abs().max()
    assert spread < 1e-4


def test_predictor_receives_frame_zero_normalized_world_to_camera_poses():
    images, geometry = make_scene(camera="moving")
    predictor = StaticScenePredictor()
    build(predictor).forward(images, make_queries(geometry), geometry=geometry)

    passed = torch.from_numpy(predictor.calls[0]["extrinsics_w2c"])
    assert passed.shape == (T_CLIP, 4, 4)
    torch.testing.assert_close(passed[0], torch.eye(4), atol=1e-5, rtol=1e-5)

    # Each matrix must take a point from the frame-0 camera into frame j's camera, which is what
    # upstream's ``w2c_0 @ c2w[j]`` composition assumes.
    c2w = geometry.extrinsics[0, :, 0]
    point = torch.tensor([0.3, -0.2, 2.5, 1.0])
    in_camera_0 = torch.linalg.inv(c2w[0]) @ point
    for j in range(T_CLIP):
        expected = torch.linalg.inv(c2w[j]) @ point
        torch.testing.assert_close(passed[j] @ in_camera_0, expected, atol=1e-4, rtol=1e-4)


def test_trajectory_scale_follows_the_geometry_units():
    """Upstream returns depth-map units; the adapter must neither normalise nor rescale them."""
    factor = 7.5
    base_images, base_geometry = make_scene(camera="static", scale=1.0)
    big_images, big_geometry = make_scene(camera="static", scale=factor)

    base = build(StaticScenePredictor()).forward(
        base_images, make_queries(base_geometry), geometry=base_geometry
    )
    big = build(StaticScenePredictor()).forward(
        big_images, make_queries(big_geometry), geometry=big_geometry
    )
    torch.testing.assert_close(big.tracks_world, base.tracks_world * factor, atol=1e-3, rtol=1e-4)
    assert big.metadata["input_depth_above_diag_limit_fraction"] == (0.0,)


def test_query_uv_selects_the_pixel_its_centre_falls_in():
    """``(x+0.5)/W`` must land on column ``x`` after upstream's truncating ``astype(int)``."""
    images, geometry = make_scene(camera="identity")
    out = build(PixelIndexPredictor()).forward(images, make_queries(geometry), geometry=geometry)

    expected_x = torch.tensor([float(x) for x, _ in QUERY_PIXELS])
    expected_y = torch.tensor([float(y) for _, y in QUERY_PIXELS])
    torch.testing.assert_close(out.tracks_world[0, :, :, 0], expected_x.expand(T_CLIP, -1))
    torch.testing.assert_close(out.tracks_world[0, :, :, 1], expected_y.expand(T_CLIP, -1))


def test_intrinsics_are_converted_to_integer_centred_pixels():
    images, geometry = make_scene()
    predictor = StaticScenePredictor()
    build(predictor).forward(images, make_queries(geometry), geometry=geometry)

    fx, fy, cx, cy = predictor.calls[0]["intrinsics"]
    assert fx == pytest.approx(FX_N * W_MODEL)
    assert fy == pytest.approx(FY_N * H_MODEL)
    # Upstream's np.arange grid has no half-pixel offset, so the principal point loses 0.5.
    assert cx == pytest.approx(CX_N * W_MODEL - 0.5)
    assert cy == pytest.approx(CY_N * H_MODEL - 0.5)


def test_motion_is_carried_through_the_frame_zero_pose():
    """A rigid offset in camera-0 coordinates must appear rotated into the world frame."""
    motion = np.stack([np.zeros(3), np.array([0.25, 0.0, 0.0]), np.array([0.0, 0.5, 0.0])])
    images, geometry = make_scene(camera="static")
    queries = make_queries(geometry)
    out = build(StaticScenePredictor(motion=motion)).forward(images, queries, geometry=geometry)

    rotation = geometry.extrinsics[0, 0, 0, :3, :3]
    offsets = (rotation @ torch.from_numpy(motion).float().T).T
    expected = queries.xyz_world[:, None] + offsets[None, :, None]
    torch.testing.assert_close(out.tracks_world, expected, atol=1e-4, rtol=1e-4)


def test_inputs_are_resampled_onto_the_native_grid():
    """Coarse depth and oversized RGB both arrive at ``(height, width)``, and still round-trip."""
    images, geometry = make_scene(depth_hw=(16, 24), image_hw=(64, 96), camera="static")
    queries = make_queries(geometry)
    predictor = StaticScenePredictor()
    out = build(predictor).forward(images, queries, geometry=geometry)

    call = predictor.calls[0]
    assert call["image_size"] == (H_MODEL, W_MODEL)
    assert call["depth_map"].shape == (T_CLIP, H_MODEL, W_MODEL)
    assert call["num_frames"] == T_CLIP
    assert call["visibility"].shape == (T_CLIP, len(QUERY_PIXELS))
    expected = queries.xyz_world[:, None].expand(-1, T_CLIP, -1, -1)
    torch.testing.assert_close(out.tracks_world, expected, atol=1e-4, rtol=1e-4)


def test_each_batch_item_is_predicted_on_its_own_geometry():
    images, geometry = make_scene(b=2, camera="moving")
    queries = make_queries(geometry)
    predictor = StaticScenePredictor()
    out = build(predictor).forward(images, queries, geometry=geometry)

    assert len(predictor.calls) == 2  # upstream is batch-size 1
    assert not np.allclose(predictor.calls[0]["depth_map"], predictor.calls[1]["depth_map"])
    expected = queries.xyz_world[:, None].expand(-1, T_CLIP, -1, -1)
    torch.testing.assert_close(out.tracks_world, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Depth masking
# ---------------------------------------------------------------------------
def test_masked_depth_is_filled_and_reported_and_marks_its_queries_invalid():
    images, geometry = make_scene(camera="static")
    hole_x, hole_y = 5, 7
    valid = torch.ones_like(geometry.depth, dtype=torch.bool)
    valid[0, 0, 0, hole_y, hole_x] = False
    geometry.depth_valid = valid
    geometry.validate()

    queries = lift_manually(geometry)
    predictor = StaticScenePredictor()
    out = build(predictor, query_depth_tolerance=None).forward(images, queries, geometry=geometry)

    supplied = predictor.calls[0]["depth_map"]
    assert np.isfinite(supplied).all() and (supplied > 0).all()
    frame0 = geometry.depth[0, 0, 0]
    assert supplied[0, hole_y, hole_x] == pytest.approx(
        float(frame0[valid[0, 0, 0]].median()), abs=1e-6
    )
    assert out.metadata["depth_invalid_fraction"] == pytest.approx(
        (1.0 / (T_CLIP * H_MODEL * W_MODEL),)
    )

    hole_query = QUERY_PIXELS.index((hole_x, hole_y))
    assert not out.valid[0, :, hole_query].any()
    others = [i for i in range(len(QUERY_PIXELS)) if i != hole_query]
    assert out.valid[0][:, others].all()


def test_a_fully_masked_frame_is_rejected():
    images, geometry = make_scene(camera="static")
    valid = torch.ones_like(geometry.depth, dtype=torch.bool)
    valid[0, 1] = False  # queries stay on frame 0, so this reaches the depth packer
    geometry.depth_valid = valid
    geometry.validate()

    with pytest.raises(ValueError, match="no valid depth"):
        build(StaticScenePredictor()).forward(images, make_queries(geometry), geometry=geometry)


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------
def test_dense_visibility_is_sampled_when_it_carries_a_time_axis():
    images, geometry = make_scene(camera="static")
    rng = np.random.default_rng(3)
    dense = rng.random((T_CLIP, H_MODEL, W_MODEL)).astype(np.float32)
    out = build(StaticScenePredictor(vis_dense=dense)).forward(
        images, make_queries(geometry), geometry=geometry
    )

    assert out.visibility_scope == "query_view"
    assert out.metadata["visibility_source"].endswith("_last_vis_dense")
    expected = torch.from_numpy(np.stack([dense[:, y, x] for x, y in QUERY_PIXELS], axis=1))
    torch.testing.assert_close(out.visibility[0], expected)


@pytest.mark.parametrize("dense", [None, np.zeros((H_MODEL, W_MODEL), dtype=np.float32)])
def test_visibility_without_a_time_axis_is_reported_unavailable(dense):
    """A single 2-D map is not per-timestep visibility, so it is not broadcast into one."""
    images, geometry = make_scene(camera="static")
    out = build(StaticScenePredictor(vis_dense=dense)).forward(
        images, make_queries(geometry), geometry=geometry
    )
    assert out.visibility is None
    assert out.visibility_scope == "unavailable"
    assert out.metadata["visibility_source"] == "unavailable"


# ---------------------------------------------------------------------------
# Rejected inputs
# ---------------------------------------------------------------------------
def test_query_times_other_than_the_reference_frame_are_rejected():
    images, geometry = make_scene(camera="static")
    queries = make_queries(geometry, time_index=1)
    with pytest.raises(ValueError, match="arbitrary_query_times"):
        build(StaticScenePredictor()).forward(images, queries, geometry=geometry)


def test_multiple_views_are_rejected_rather_than_looped():
    images, geometry = make_scene(v=2, camera="static")
    queries = make_queries(geometry)
    with pytest.raises(ValueError, match="monocular"):
        build(StaticScenePredictor()).forward(images, queries, geometry=geometry)


def test_clip_length_is_restricted_to_the_evaluated_setting():
    images, geometry = make_scene(t=T_CLIP + 1, camera="static")
    queries = make_queries(geometry)
    with pytest.raises(ValueError, match="frame clips"):
        build(StaticScenePredictor()).forward(images, queries, geometry=geometry)

    tracker = build(StaticScenePredictor(), allow_variable_clip_length=True)
    out = tracker.forward(images, queries, geometry=geometry)
    assert out.tracks_world.shape[1] == T_CLIP + 1


@pytest.mark.parametrize("size", [{"height": 40}, {"width": 50}, {"height": 0}])
def test_grid_must_be_a_multiple_of_sixteen(size):
    with pytest.raises(ValueError, match="multiple of 16"):
        build(StaticScenePredictor(), **size)


def test_finetuning_is_refused_because_upstream_has_no_gradient_path():
    with pytest.raises(ValueError, match="freeze_tracker"):
        build(StaticScenePredictor(), freeze_tracker=False)


def test_time_varying_intrinsics_are_rejected():
    images, geometry = make_scene(camera="static")
    geometry.intrinsics[0, 1, 0, 0, 0] *= 1.05  # upstream takes one [fx,fy,cx,cy] per clip
    with pytest.raises(ValueError, match="vary"):
        build(StaticScenePredictor()).forward(images, make_queries(geometry), geometry=geometry)


def test_dropped_queries_are_reported_instead_of_silently_realigned():
    images, geometry = make_scene(camera="static")
    with pytest.raises(RuntimeError, match=tc3r.TRACKCRAFT3R_REVISION):
        build(StaticScenePredictor(drop_last=True)).forward(
            images, make_queries(geometry), geometry=geometry
        )


def test_empty_queries_short_circuit():
    images, geometry = make_scene(camera="static")
    queries = PointQueries(
        ids=torch.zeros(1, 0, dtype=torch.int64),
        time=torch.zeros(1, 0, dtype=torch.int64),
        xyz_world=torch.zeros(1, 0, 3),
        source_view=torch.zeros(1, 0, dtype=torch.int64),
        source_uv=torch.zeros(1, 0, 2),
    )
    predictor = StaticScenePredictor()
    out = build(predictor).forward(images, queries, geometry=geometry)

    assert out.tracks_world.shape == (1, T_CLIP, 0, 3)
    assert predictor.calls == []


# ---------------------------------------------------------------------------
# Module plumbing and upstream loading
# ---------------------------------------------------------------------------
def test_pipeline_modules_are_frozen_and_kept_in_eval():
    tracker = build(StaticScenePredictor())
    assert list(tracker.parameters())  # the fake pipeline's modules were registered
    assert not any(p.requires_grad for p in tracker.parameters())
    assert not tracker.model.training
    tracker.train()
    assert not tracker.model.training


def test_repo_path_must_point_at_a_checkout(tmp_path):
    cfg = make_config(repo_path=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="not a TrackCraft3R checkout"):
        tc3r.load_predictor(cfg)


def test_foreign_top_level_package_is_rejected_not_overwritten(monkeypatch, tmp_path):
    foreign = types.ModuleType("evaluation")
    foreign.__file__ = str(tmp_path / "somewhere" / "__init__.py")
    monkeypatch.setitem(sys.modules, "evaluation", foreign)

    with pytest.raises(ImportError, match="already imported from"):
        tc3r.guard_upstream_packages(None)
    assert sys.modules["evaluation"] is foreign  # nothing was replaced


def test_matching_checkout_passes_the_import_guard(monkeypatch, tmp_path):
    repo = (tmp_path / "repo").resolve()
    package = repo / "evaluation"
    package.mkdir(parents=True)
    (package / "wan_scene_flow_predictor.py").write_text("")

    module = types.ModuleType("evaluation")
    module.__path__ = [str(package)]
    monkeypatch.setitem(sys.modules, "evaluation", module)

    tc3r.guard_upstream_packages(repo)
    tc3r.guard_upstream_packages(None)
    with pytest.raises(ImportError, match="different TrackCraft3R checkout"):
        tc3r.guard_upstream_packages((tmp_path / "elsewhere").resolve())
