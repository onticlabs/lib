"""Standalone tests for ontic_lib.pointops.alignment."""

import pytest
import torch

from ontic_lib.pointops import alignment as A
from ontic_lib.transforms.rotations import rotation_6d_to_matrix


def _random_rotation(generator):
    return rotation_6d_to_matrix(torch.randn(6, generator=generator))


def _random_poses(batch, views, generator):
    poses = torch.eye(4).repeat(batch, views, 1, 1).clone()
    for b in range(batch):
        for v in range(views):
            poses[b, v, :3, :3] = _random_rotation(generator)
            poses[b, v, :3, 3] = torch.randn(3, generator=generator)
    return poses


def test_sim3_recovers_known_transform():
    g = torch.Generator().manual_seed(1)
    source = _random_poses(2, 6, g)
    rotation = _random_rotation(g)
    translation = torch.randn(3, generator=g)
    scale = torch.tensor(2.3)
    target = A.align_cameras_sim3(source, rotation, translation, scale)

    recovered_rotation, recovered_translation, recovered_scale = A.align_camera_poses_sim3(
        source, target
    )
    assert torch.allclose(recovered_rotation[0], rotation, atol=1e-5)
    assert torch.allclose(recovered_translation[0], translation, atol=1e-5)
    assert torch.allclose(recovered_scale, torch.full_like(recovered_scale, 2.3), atol=1e-5)


def test_se3_recovers_known_rigid_transform():
    g = torch.Generator().manual_seed(2)
    source = _random_poses(2, 5, g)
    rotation = _random_rotation(g)
    translation = torch.randn(3, generator=g)
    target = A.align_cameras_sim3(source, rotation, translation, torch.tensor(1.0))

    recovered_rotation, recovered_translation = A.align_camera_poses_se3(source, target)
    assert torch.allclose(recovered_rotation[0], rotation, atol=1e-5)
    assert torch.allclose(recovered_translation[0], translation, atol=1e-5)


def test_align_points_sim3_matches_manual():
    g = torch.Generator().manual_seed(3)
    points = torch.randn(4, 7, 3, generator=g)
    rotation = _random_rotation(g).expand(4, 3, 3)
    translation = torch.randn(4, 3, generator=g)
    scale = torch.rand(4, generator=g) + 0.5

    out = A.align_points_sim3(points, rotation, translation, scale)
    manual = (
        scale[:, None, None] * torch.matmul(points, rotation.transpose(-1, -2))
        + (translation[:, None])
    )
    assert torch.allclose(out, manual, atol=1e-5)


def test_clamp_scale_bounds():
    scale = torch.tensor([1e-9, 1.0, 1e9, float("nan"), float("inf"), float("-inf")])
    clamped = A.clamp_scale(scale, minimum=0.01, maximum=100.0)
    assert torch.all(clamped >= 0.01)
    assert torch.all(clamped <= 100.0)
    assert torch.isfinite(clamped).all()
    # nan maps to 1.0 then stays within bounds.
    assert clamped[3].item() == pytest.approx(1.0)
    # +inf clamps to the maximum, -inf to the minimum.
    assert clamped[4].item() == pytest.approx(100.0)
    assert clamped[5].item() == pytest.approx(0.01)


def test_anchor_transform_maps_source_to_target():
    g = torch.Generator().manual_seed(4)
    source = torch.eye(4)
    source[:3, :3] = _random_rotation(g)
    source[:3, 3] = torch.randn(3, generator=g)
    target = torch.eye(4)
    target[:3, :3] = _random_rotation(g)
    target[:3, 3] = torch.randn(3, generator=g)

    transform = A.anchor_transform(source, target)
    assert torch.allclose(transform @ source, target, atol=1e-5)


# --------------------------------------------------------------------------- #
# Metric alignment: compute_alignment / align / apply_metric_scale + conf masks
# --------------------------------------------------------------------------- #


def _has(name):
    import importlib

    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


def _spread_poses(views, scale=1.0, offset=(0.0, 0.0, 0.0)):
    """Identity-rotation cameras spread along x; centres scaled + shifted."""
    poses = torch.eye(4).expand(views, 4, 4).clone()
    centers = torch.zeros(views, 3)
    centers[:, 0] = torch.arange(views).float()
    poses[:, :3, 3] = scale * centers + torch.tensor(offset)
    return poses.unsqueeze(0)  # (1, V, 4, 4)


def _identity_intrinsics(views):
    k = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
    return k.expand(views, 3, 3).clone().unsqueeze(0)


def test_percentile_conf_threshold_keep_all_and_drop_fraction():
    assert A.percentile_conf_threshold(None, 50.0) == float("-inf")
    conf = torch.arange(1, 101).float().reshape(10, 10)
    assert A.percentile_conf_threshold(conf, 0) == float("-inf")
    threshold = A.percentile_conf_threshold(conf, 80)
    kept = (conf > threshold).float().mean().item()
    assert abs(kept - 0.2) < 0.05


def test_conf_drop_mask_is_per_sample():
    conf = torch.stack([torch.arange(10).float(), 100 + torch.arange(10).float()])
    mask = A.conf_drop_mask(conf, 50.0)
    assert mask.shape == conf.shape and mask.dtype == torch.bool
    # each sample keeps its own top half, regardless of the other sample's scale.
    assert mask[0].sum() == mask[1].sum()
    assert A.conf_drop_mask(conf, 0.0) is None
    assert A.conf_drop_mask(None, 30.0) is None


def test_compute_alignment_none_is_identity_and_falls_back_to_target_cameras():
    pred, gt = _spread_poses(3, scale=0.5), _spread_poses(3)
    intr = _identity_intrinsics(3)
    scale, cams, k = A.compute_alignment(pred, intr, gt, intr, "none")
    assert torch.equal(scale, torch.ones(1)) and cams is pred and k is intr
    scale, cams, k = A.compute_alignment(None, None, gt, intr, "none")
    assert cams is gt and k is intr


def test_compute_alignment_prescale_gt_returns_umeyama_scale_and_target_cameras():
    pred, gt = _spread_poses(3, scale=0.5), _spread_poses(3)
    intr = _identity_intrinsics(3)
    scale, cams, _ = A.compute_alignment(pred, intr, gt, intr, "prescale_gt")
    assert scale.item() == pytest.approx(2.0, abs=1e-5)
    assert cams is gt


def test_compute_alignment_sim3_points_maps_predicted_cameras_onto_target():
    pred, gt = _spread_poses(3, scale=0.5, offset=(1.0, -2.0, 0.5)), _spread_poses(3)
    intr = _identity_intrinsics(3)
    scale, cams, _ = A.compute_alignment(pred, intr, gt, intr, "sim3_points")
    assert scale.item() == pytest.approx(2.0, abs=1e-5)
    assert torch.allclose(cams, gt, atol=1e-5)


def test_compute_alignment_metric_mono_requires_scale_and_rigidly_fits():
    pred, gt = _spread_poses(3, scale=0.5), _spread_poses(3)
    intr = _identity_intrinsics(3)
    with pytest.raises(ValueError):
        A.compute_alignment(pred, intr, gt, intr, "metric_mono")
    scale, cams, _ = A.compute_alignment(pred, intr, gt, intr, "metric_mono", metric_scale=2.0)
    assert scale.shape == (1,) and scale.item() == 2.0
    assert torch.allclose(cams, gt, atol=1e-5)  # true scale -> rigid fit recovers GT
    assert torch.equal(pred[..., :3, 3], _spread_poses(3, scale=0.5)[..., :3, 3])  # untouched


def test_compute_alignment_metric_mono_without_target_centres_on_cloud():
    pred = _spread_poses(2, offset=(0.0, 0.0, 0.0))
    intr = _identity_intrinsics(2)
    depth = torch.full((1, 2, 4, 4), 2.0)
    scale, cams, _ = A.compute_alignment(
        pred, intr, None, None, "metric_mono", metric_scale=1.0, depth=depth
    )
    from ontic_lib.depth.lifting import depth_to_world_points

    points = depth_to_world_points(depth[0].unsqueeze(-1), cams[0], intr[0], (4, 4))
    assert torch.allclose(points.reshape(-1, 3).mean(0), torch.zeros(3), atol=1e-5)


def test_compute_alignment_rejects_unknown_mode():
    pred, gt = _spread_poses(2), _spread_poses(2)
    with pytest.raises(ValueError):
        A.compute_alignment(pred, None, gt, None, "bogus")


def test_align_applies_scale_to_depth():
    pred, gt = _spread_poses(3, scale=0.5), _spread_poses(3)
    intr = _identity_intrinsics(3)
    depth = torch.rand(1, 3, 4, 4) + 0.5
    scaled, cams, _ = A.align(depth, pred, intr, gt, intr, "prescale_gt")
    assert torch.allclose(scaled, 2.0 * depth, atol=1e-5)
    assert torch.allclose(A.apply_metric_scale(depth, torch.tensor([3.0])), 3.0 * depth)


@pytest.mark.skipif(not _has("fwomo_3d.utils.camera_utils"), reason="needs fwomo_3d")
class TestFrontierParity:
    """Bitwise parity with ``fwomo_3d.utils.camera_utils`` (frontier venv only)."""

    def _random_c2w(self, batch, views, seed):
        g = torch.Generator().manual_seed(seed)
        poses = torch.eye(4).repeat(batch, views, 1, 1)
        for b in range(batch):
            for v in range(views):
                poses[b, v, :3, :3] = _random_rotation(g)
                poses[b, v, :3, 3] = torch.randn(3, generator=g) * 2
        return poses

    @pytest.mark.parametrize("mode", ["none", "prescale_gt", "sim3_points", "metric_mono"])
    def test_align_matches_frontier(self, mode):
        from fwomo_3d.utils import camera_utils as F

        g = torch.Generator().manual_seed(0)
        pred, gt = self._random_c2w(2, 4, 1), self._random_c2w(2, 4, 2)
        intr = _identity_intrinsics(4).repeat(2, 1, 1, 1)
        depth = torch.rand(2, 4, 6, 8, generator=g) + 0.3
        kwargs = {"metric_scale": torch.tensor([1.7, 0.4])} if mode == "metric_mono" else {}
        ours = A.align(depth, pred, intr, gt, intr, mode, **kwargs)
        theirs = F.align(depth, pred, intr, gt, intr, mode, **kwargs)
        for a, b in zip(ours, theirs):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_metric_mono_no_target_matches_frontier(self):
        from fwomo_3d.utils import camera_utils as F

        g = torch.Generator().manual_seed(3)
        pred = self._random_c2w(2, 3, 4)
        intr = _identity_intrinsics(3).repeat(2, 1, 1, 1)
        depth = torch.rand(2, 3, 5, 7, generator=g) + 0.3
        ours = A.compute_alignment(
            pred, intr, None, None, "metric_mono", metric_scale=2.0, depth=depth
        )
        theirs = F.compute_alignment(
            pred, intr, None, None, "metric_mono", metric_scale=2.0, depth=depth
        )
        for a, b in zip(ours, theirs):
            torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

    def test_conf_helpers_match_frontier(self):
        from fwomo_3d.utils import camera_utils as F

        conf = torch.rand(2, 3, 5, 5, generator=torch.Generator().manual_seed(5)) * 40 + 1
        for pct in (0.0, 25.0, 80.0, 100.0):
            assert A.percentile_conf_threshold(conf, pct) == F.percentile_conf_threshold(conf, pct)
            ours, theirs = A.conf_drop_mask(conf, pct), F.conf_drop_mask(conf, pct)
            assert (ours is None and theirs is None) or torch.equal(ours, theirs)
