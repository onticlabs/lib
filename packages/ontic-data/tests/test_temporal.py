"""TemporalSceneDataset on a tiny in-memory subclass: indexing, split, example schema."""

import torch

from ontic_data.config import DatasetCfg
from ontic_data.temporal import SceneView, TemporalSceneDataset, loader_horizon, split_records
from ontic_data.view_sampler import ViewSamplerCfg

N_SCENES, N_FRAMES, N_CAMS, H, W = 2, 6, 4, 12, 12


def _pixels(scene, frame, cam):
    g = torch.Generator().manual_seed(scene * 1000 + frame * 10 + cam)
    return torch.rand(3, H, W, generator=g)


class ToyView(SceneView):
    def __init__(self, ds, rec):
        self.ds, self.rec = ds, rec
        self.cam_names = [f"cam_{i}" for i in range(N_CAMS)]
        K = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]])
        self.intrinsics = K.expand(N_CAMS, 3, 3).clone()
        self._extr = torch.eye(4).expand(N_CAMS, 4, 4).clone()
        self._extr[:, 0, 3] = torch.arange(N_CAMS).float()

    def extrinsics(self, frame_idx):
        return self._extr

    def load_views(self, cam_ixs, frame_idx, out_hw, *, depth=False, side="target"):
        out = {"image": torch.stack([_pixels(self.rec, frame_idx, int(c)) for c in cam_ixs])}
        if side == "target":
            out["state_mask"] = out["image"][:, :1] > 0.5
        return out

    def load_actions(self, frame_indices):
        if not getattr(self.ds.cfg, "actions", False):
            return None
        T = len(frame_indices)
        return {"hand": torch.cat([torch.rand(T, 1, 5, 3), torch.ones(T, 1, 5, 1)], -1)}


class ToyDataset(TemporalSceneDataset):
    def _discover(self):
        return list(range(N_SCENES))

    def _n_frames_of(self, rec):
        return N_FRAMES

    def _scene_name(self, rec, t_start):
        return f"scene{rec}_{t_start}"

    def _open_scene(self, rec):
        return ToyView(self, rec)


def _cfg(**kw):
    base = dict(
        image_shape=[H, W],
        n_step_state=1,
        n_step_predict=2,
        val_n_step_predict=2,
        view_sampler=ViewSamplerCfg(num_context_views=2, num_target_views=1),
    )
    base.update(kw)
    return DatasetCfg(**base)


def _build(stage="train", seed=0, **kw):
    cfg = _cfg(**kw)
    return ToyDataset(
        cfg, stage, cfg.build_view_sampler(stage, torch.Generator().manual_seed(seed)), **{}
    )


def test_snippet_indexing_and_speedup():
    ds = _build()
    assert len(ds) == N_SCENES * (N_FRAMES - 3 + 1)
    assert ds._index[:4] == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert len(_build(speedup=2)) == N_SCENES * 1


def test_example_schema_and_shapes():
    ds = _build(workspace_min=[-1, -1, -1], workspace_max=[1, 1, 1])
    ds.cfg.actions = True
    ex = ds[1]
    assert set(ex) == {"scene", "context", "target", "workspace_min", "workspace_max", "actions"}
    assert ex["scene"] == "scene0_1"
    ctx, tgt = ex["context"], ex["target"]
    for k in ("image", "extrinsics", "intrinsics", "near", "far", "depth_is_metric", "index"):
        assert k in ctx and k in tgt
    assert ctx["image"].shape == (1, 2, 3, H, W)
    assert tgt["image"].shape == (3, 1, 3, H, W)
    assert tgt["state_mask"].shape == (3, 1, 1, H, W) and "state_mask" not in ctx
    assert ctx["near"].shape == (1, 2) and tgt["depth_is_metric"].shape == (3, 1)
    assert ex["actions"]["hand"].shape == (3, 1, 5, 4)
    assert ex["workspace_min"].tolist() == [-1, -1, -1]
    # pixels come from the right (scene, frame, camera)
    t, cam = 1, int(ctx["index"][0, 0])
    assert torch.equal(ctx["image"][0, 0], _pixels(0, t, cam))
    assert torch.equal(tgt["image"][2, 0], _pixels(0, t + 2, int(tgt["index"][2, 0])))


def test_validation_keeps_target_views_fixed():
    ds = _build("val")
    ex = ds[0]
    idx = ex["target"]["index"]
    assert torch.equal(idx[1], idx[0]) and torch.equal(idx[2], idx[0])


def test_consistent_cameras_reuse_state_views():
    ds = _build(consistent_cameras=True)
    ex = ds[0]
    ctx = ex["context"]["index"][0]
    # target_includes_context is False -> target == context at every step
    assert all(torch.equal(t, ctx) for t in ex["target"]["index"])


def test_horizon_aware_loading_truncates_but_keeps_rng_stream():
    full = _build(seed=5)[2]
    ds = _build(seed=5, horizon_aware_loading=True)
    ds.step_fn = lambda: 0
    ds.horizon_fn = lambda step, n_pred_full: 1
    short = ds[2]
    assert short["target"]["image"].shape[0] == 2
    assert torch.equal(short["target"]["image"], full["target"]["image"][:2])
    assert torch.equal(short["target"]["index"], full["target"]["index"][:2])


def test_loader_horizon_gating():
    cfg = _cfg(horizon_aware_loading=True)
    assert loader_horizon(cfg, "train", False, None, None, 3) == 3
    assert loader_horizon(cfg, "val", True, lambda: 0, lambda s, f: 0, 3) == 3
    assert loader_horizon(cfg, "train", False, lambda: 0, lambda s, f: 1, 3) == 2
    assert loader_horizon(cfg, "train", False, lambda: 0, lambda s, f: 99, 3) == 3


def test_split_records_deterministic_disjoint():
    records = list(range(10))
    train = split_records(records, "train", 0.15)
    val = split_records(records, "val", 0.15)
    assert len(val) == 2 and len(train) == 8
    assert not set(train) & set(val) and sorted(train + val) == records
    assert split_records(records, "test", 0.15) == val
    assert split_records(records[:4], "val", 0.15) == records[:4]


def test_load_sequence_views():
    out = _build().load_sequence_views(1, 3, with_depth=False)
    assert out["image"].shape == (N_CAMS, 3, H, W)
    assert out["index"] == [f"cam_{i}" for i in range(N_CAMS)]
    assert out["scene"] == "scene1_0" and "actions" not in out
