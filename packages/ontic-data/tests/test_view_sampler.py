"""View sampler: determinism under a seeded generator, contract of each mode."""

import pytest
import torch

from ontic_data.view_sampler import ViewSampler, ViewSamplerCfg, build_camera_pairs

RIG = ["1-1", "1-2", "2-1", "2-2", "3-1", "3-2", "wrist_camera_r", "wrist_camera_l", "scene_camera"]


def _cams(n=6):
    return torch.eye(4).expand(n, 4, 4), torch.eye(3).expand(n, 3, 3)


def _draw(cfg, seed, n_draws=5, **kw):
    sampler = ViewSampler(cfg, "train", torch.Generator().manual_seed(seed))
    extr, intr = _cams()
    return [sampler.sample("s", extr, intr, **kw) for _ in range(n_draws)]


def test_seeded_generator_is_deterministic_and_seed_sensitive():
    cfg = ViewSamplerCfg(num_context_views=3, num_target_views=2)
    a, b = _draw(cfg, 0), _draw(cfg, 0)
    assert all(torch.equal(x[0], y[0]) and torch.equal(x[1], y[1]) for x, y in zip(a, b))
    c = _draw(cfg, 1)
    assert any(not torch.equal(x[0], y[0]) for x, y in zip(a, c))


def test_generator_does_not_touch_global_rng():
    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    _draw(ViewSamplerCfg(), 7)
    assert torch.equal(torch.rand(3), before)


def test_context_and_target_disjoint_sorted_and_sized():
    for ctx, tgt in _draw(ViewSamplerCfg(num_context_views=2, num_target_views=3), 3):
        assert ctx.tolist() == sorted(ctx.tolist()) and tgt.tolist() == sorted(tgt.tolist())
        assert len(ctx) == 2 and len(tgt) == 3
        assert not set(ctx.tolist()) & set(tgt.tolist())


def test_target_includes_context_and_all_remaining():
    cfg = ViewSamplerCfg(num_context_views=2, num_target_views=-1, target_includes_context=True)
    for ctx, tgt in _draw(cfg, 0):
        assert tgt.tolist() == list(range(6))
        assert set(ctx.tolist()) <= set(tgt.tolist())


def test_all_views_both_minus_one():
    (ctx, tgt), *_ = _draw(ViewSamplerCfg(num_context_views=-1, num_target_views=-1), 0, 1)
    assert ctx.tolist() == list(range(6)) and tgt.tolist() == list(range(6))


def test_fixed_context_groups_follow_camera_group_ix():
    cfg = ViewSamplerCfg(num_context_views=2, num_target_views=1, context_views=[[0, 1], [4, 5]])
    sampler = ViewSampler(cfg, "train", torch.Generator().manual_seed(0))
    extr, intr = _cams()
    assert sampler.sample("s", extr, intr, camera_group_ix=0)[0].tolist() == [0, 1]
    assert sampler.sample("s", extr, intr, camera_group_ix=1)[0].tolist() == [4, 5]
    assert sampler.sample("s", extr, intr, camera_group_ix=2)[0].tolist() == [0, 1]


def test_update_camera_indices_remaps_once():
    cfg = ViewSamplerCfg(num_context_views=2, num_target_views=1, context_views=[[2, 4], [4, 6]])
    sampler = ViewSampler(cfg, "train")
    sampler.update_camera_indices([2, 4, 6])
    sampler.update_camera_indices([9, 9, 9])  # ignored: already remapped
    assert sampler.context_views == [[0, 1], [1, 2]]


def test_too_many_views_requested():
    with pytest.raises(ValueError, match="need 7 views"):
        _draw(ViewSamplerCfg(num_context_views=4, num_target_views=3), 0, 1)


def test_build_camera_pairs_numbered_then_rings():
    assert build_camera_pairs(tuple(RIG)) == ((0, 1), (2, 3), (4, 5))
    rings = ("ring_00", "ring_01", "ring_02", "ring_03", "ring_04", "birds_eye")
    assert build_camera_pairs(rings) == ((0, 1), (2, 3))


def test_paired_mode_pairs_context_with_target_and_draws_extras():
    cfg = ViewSamplerCfg(num_context_views=3, num_target_views=3, paired_views=True)
    sampler = ViewSampler(cfg, "train", torch.Generator().manual_seed(0))
    extr, intr = _cams(len(RIG))
    pairs = dict(build_camera_pairs(tuple(RIG)))
    pool = {RIG.index(n) for n in cfg.pair_extra_pool}
    for _ in range(10):
        ctx, tgt = sampler.sample("s", extr, intr, cam_names=RIG)
        ctx, tgt = ctx.tolist(), tgt.tolist()
        assert len(ctx) == 3 and len(tgt) == 3 and not set(ctx) & set(tgt)
        ctx_pairs = [c for c in ctx if c in pairs]
        assert len(ctx_pairs) == 2 and all(pairs[c] in tgt for c in ctx_pairs)
        extras = set(ctx + tgt) - set(ctx_pairs) - {pairs[c] for c in ctx_pairs}
        assert extras <= pool and len(extras) == 2


def test_paired_mode_needs_cam_names_and_consistent_counts():
    extr, intr = _cams(len(RIG))
    with pytest.raises(ValueError, match="cam_names"):
        ViewSampler(ViewSamplerCfg(paired_views=True), "train").sample("s", extr, intr)
    bad = ViewSamplerCfg(num_context_views=3, num_target_views=2, paired_views=True)
    with pytest.raises(ValueError, match="num_target_views"):
        ViewSampler(bad, "train").sample("s", extr, intr, cam_names=RIG)
