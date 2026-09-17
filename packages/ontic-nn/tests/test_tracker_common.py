"""Geometry invariants at the tracker boundary, without pretrained dependencies."""

import pytest
import torch
import subprocess
import sys

from ontic_nn.trackers import GeometrySequence, PointQueries
from ontic_nn.trackers.common import (
    make_tracker_output,
    pixel_intrinsics,
    project_queries,
    resize_tracker_inputs,
    validate_tracker_inputs,
)
from ontic_nn.wrappers.common import BackboneOutput


def geometry(b=2, t=3, v=2, h=4, w=6):
    c2w = torch.eye(4).repeat(b, t, v, 1, 1)
    c2w[..., 0, 3] = torch.arange(t)[None, :, None] * 0.25
    c2w[:, :, 1:, 1, 3] = 0.7
    k = torch.tensor([[0.8, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]])
    return GeometrySequence(
        torch.full((b, t, v, h, w), 2.0),
        c2w,
        k.repeat(b, t, v, 1, 1),
        frame_ids=tuple(f"world-{i}" for i in range(b)),
        units=("meters",) * b,
    )


def queries(g):
    b, t, v, h, w = g.depth.shape
    times = torch.tensor([0, t - 1]).expand(b, -1)
    views = torch.tensor([0, v - 1]).expand(b, -1)
    uv = torch.tensor([[0.5 / w, 0.5 / h], [(w - 0.5) / w, (h - 0.5) / h]])
    return PointQueries.from_pixels(g, times, views, uv.expand(b, -1, -1))


def test_lift_and_project_query_times_views_and_moving_cameras():
    g = geometry()
    q = queries(g)
    # Camera 0 at t0: top-left pixel ray scaled to z=2.
    expected0 = torch.tensor([(0.5 / 6 - 0.5) * 2 / 0.8, (0.5 / 4 - 0.5) * 2 / 1.2, 2])
    # Camera 1 at t2 translates x=.5, y=.7; same z-depth in both cameras.
    expected1 = -expected0 + torch.tensor([0.5, 0.7, 4.0])
    torch.testing.assert_close(q.xyz_world[0], torch.stack([expected0, expected1]))
    torch.testing.assert_close(project_queries(q, g), q.source_uv)
    assert torch.equal(q.ids, torch.tensor([[0, 1], [0, 1]]))


def test_integer_center_intrinsics_preserve_pixel_rays_and_input():
    g = geometry(b=1, t=1, v=1)
    q = queries(g)
    original = g.intrinsics.clone()
    k = pixel_intrinsics(g.intrinsics[0, 0, 0], 4, 6)
    projection = (k @ q.xyz_world[0].T).T
    xy = projection[:, :2] / projection[:, 2:]
    torch.testing.assert_close(xy, torch.tensor([[0.0, 0.0], [5.0, 3.0]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(g.intrinsics, original)
    edge_k = pixel_intrinsics(g.intrinsics, 4, 6, integer_centers=False)
    torch.testing.assert_close(edge_k[..., 0, 2], torch.full((1, 1, 1), 3.0))


def test_depth_resizing_masks_nan_and_preserves_units():
    g = geometry(b=1, t=1, v=1, h=2, w=2)
    g.depth[0, 0, 0, 0, 0] = float("nan")
    g.depth_valid = torch.ones_like(g.depth, dtype=torch.bool)
    g.depth_valid[0, 0, 0, 1, 1] = False
    rgb = torch.ones(1, 1, 1, 3, 2, 2)
    images, depth, valid = resize_tracker_inputs(rgb, g, size=(4, 4))
    assert images.shape == (1, 1, 1, 3, 4, 4)
    assert torch.isfinite(depth).all()
    assert (depth[valid] == 2).all()
    assert (depth[~valid] == 0).all()
    assert valid.sum() == 8
    assert torch.isnan(g.depth[0, 0, 0, 0, 0])


def test_invalid_depth_cannot_seed_a_query():
    g = geometry(b=1, t=1, v=1)
    g.depth.zero_()
    with pytest.raises(ValueError, match="valid finite positive depth"):
        queries(g)


def test_queries_behind_visible_surface_are_rejected():
    g = geometry(b=1, t=1, v=1)
    q = queries(g)
    q.xyz_world *= 2
    with pytest.raises(ValueError, match="visible source depth"):
        project_queries(q, g)


def test_visibility_does_not_invalidate_hidden_trajectory():
    g = geometry(b=1)
    q = queries(g)
    tracks = q.xyz_world[:, None].expand(-1, 3, -1, -1).clone()
    tracks[0, 2, 1] = float("nan")
    out = make_tracker_output(q, g, tracks, torch.zeros(1, 3, 2), visibility_scope="any_view")
    assert out.valid[0, 0].all()
    assert not out.valid[0, 2, 1]
    assert out.visibility_per_view is None
    assert out.visibility_scope == "any_view"
    assert out.metadata["units"] == ("meters",)
    with pytest.raises(ValueError, match="probabilities"):
        make_tracker_output(q, g, tracks, torch.full((1, 3, 2), 2.0))


def test_explicit_backbone_camera_source_and_shared_frame():
    g = geometry(b=1, t=2, v=1)
    outputs = [
        BackboneOutput(
            data={
                "depth": g.depth[:, t],
                "extrinsics": g.extrinsics[:, t],
                "intrinsics": g.intrinsics[:, t],
                "extrinsics_pred": g.extrinsics[:, t].clone(),
                "intrinsics_pred": g.intrinsics[:, t],
            }
        )
        for t in range(2)
    ]
    for out in outputs:
        out.data["extrinsics_pred"][..., 0, 3] += 10
    selected = GeometrySequence.from_backbone_outputs(
        outputs,
        camera_source="predicted",
        shared_world_frame=True,
    )
    torch.testing.assert_close(selected.extrinsics[..., 0, 3], g.extrinsics[..., 0, 3] + 10)
    with pytest.raises(ValueError, match="shared world frame"):
        GeometrySequence.from_backbone_outputs(
            outputs, camera_source="provided", shared_world_frame=False
        )
    del outputs[0].data["extrinsics_pred"]
    with pytest.raises(ValueError, match="lacks selected geometry"):
        GeometrySequence.from_backbone_outputs(
            outputs, camera_source="predicted", shared_world_frame=True
        )


def test_queries_keep_persistent_ids_and_reject_duplicates():
    g = geometry(b=1)
    q = queries(g)
    q.ids[:] = 9
    with pytest.raises(ValueError, match="unique"):
        validate_tracker_inputs(torch.ones(1, 3, 2, 3, 4, 6), q, g)


def test_monocular_wrapper_cannot_silently_flatten_views():
    g = geometry(b=1)
    q = queries(g)
    with pytest.raises(ValueError, match="V=1"):
        validate_tracker_inputs(torch.ones(1, 3, 2, 3, 4, 6), q, g, multiview=False)


def test_empty_query_batch_has_well_defined_output():
    g = geometry(b=1, t=1, v=1)
    q = PointQueries.from_pixels(
        g,
        torch.empty(1, 0, dtype=torch.long),
        torch.empty(1, 0, dtype=torch.long),
        torch.empty(1, 0, 2),
    )
    assert q.xyz_world.shape == (1, 0, 3)
    out = make_tracker_output(q, g, torch.empty(1, 1, 0, 3))
    assert out.valid.shape == (1, 1, 0)
    assert out.visibility_scope == "unavailable"


def test_similarity_scale_is_not_hidden_in_pose_rotation():
    g = geometry(b=1)
    g.extrinsics[..., :3, :3] *= 2
    with pytest.raises(ValueError, match="rigid rotations"):
        g.validate()


def test_registry_import_never_loads_optional_research_packages():
    # A fresh process catches accidental eager imports even when another test
    # has already imported a wrapper or substituted an upstream package.
    code = """
import importlib.abc
import sys
blocked = {'mvtracker', 'models', 'diffsynth', 'evaluation', 'huggingface_hub',
           'transformers', 'peft', 'modelscope', 'pointops2_cuda', 'torch_scatter'}
class NoResearchImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise AssertionError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, NoResearchImports())
from ontic_nn.trackers import TRACKERS
assert set(TRACKERS) == {'mvtracker', 'tapip3d', 'trackcraft3r'}
assert TRACKERS['mvtracker'].CAPABILITIES.multiview
assert not TRACKERS['tapip3d'].CAPABILITIES.multiview
assert not TRACKERS['trackcraft3r'].CAPABILITIES.arbitrary_query_times
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
