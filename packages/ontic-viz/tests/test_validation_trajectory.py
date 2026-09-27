"""ontic_viz.validation.trajectory: intrinsics lerp, focus-point extrinsics interpolation, wobble."""

from __future__ import annotations

import torch

from ontic_viz.validation import generate_wobble, interpolate_extrinsics, interpolate_intrinsics


def _look_at(origin, target, up=(0.0, -1.0, 0.0)):
    origin = torch.tensor(origin, dtype=torch.float64)
    z = torch.tensor(target, dtype=torch.float64) - origin
    z = z / z.norm()
    up = torch.tensor(up, dtype=torch.float64)
    x = torch.cross(up, z, dim=-1)
    x = x / x.norm()
    y = torch.cross(z, x, dim=-1)
    e = torch.eye(4, dtype=torch.float64)
    e[:3, 0], e[:3, 1], e[:3, 2], e[:3, 3] = x, y, z, origin
    return e.float()


def test_interpolate_intrinsics_is_linear_with_time_axis_inserted():
    k0 = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]])
    k1 = torch.tensor([[3.0, 0, 0.5], [0, 2.0, 0.5], [0, 0, 1]])
    t = torch.tensor([0.0, 0.5, 1.0])
    out = interpolate_intrinsics(k0, k1, t)
    assert out.shape == (3, 3, 3)
    assert torch.equal(out[0], k0) and torch.equal(out[2], k1)
    assert out[1, 0, 0] == 2.0 and out[1, 1, 1] == 1.5 and out[1, 0, 2] == 0.5
    batched = interpolate_intrinsics(torch.stack([k0, k1]), torch.stack([k1, k0]), t)
    assert batched.shape == (2, 3, 3, 3) and torch.equal(batched[1, 0], k1)


def test_interpolate_extrinsics_hits_endpoints_and_orbits_focus_point():
    e0 = _look_at((0.0, 0.0, -2.0), (0.0, 0.0, 0.0))
    e1 = _look_at((2.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    t = torch.tensor([0.0, 0.5, 1.0])
    out = interpolate_extrinsics(e0, e1, t)
    assert out.shape == (3, 4, 4) and out.dtype == torch.float32
    assert torch.allclose(out[0], e0, atol=1e-5) and torch.allclose(out[2], e1, atol=1e-5)
    mid = out[1]
    # the focus point is the origin: the midway camera keeps distance 2 and looks at it
    assert torch.allclose(mid[:3, 3].norm(), torch.tensor(2.0), atol=1e-5)
    look_to_origin = -mid[:3, 3] / mid[:3, 3].norm()
    assert torch.allclose(mid[:3, 2], look_to_origin, atol=1e-5)
    rot = mid[:3, :3]
    assert torch.allclose(rot @ rot.T, torch.eye(3), atol=1e-5)


def test_interpolate_extrinsics_parallel_looks_lerps_about_midpoint():
    e0 = _look_at((0.0, 0.0, -2.0), (0.0, 0.0, 0.0))
    e1 = e0.clone()
    e1[:3, 3] += torch.tensor([1.0, 0.0, 0.0])
    out = interpolate_extrinsics(e0, e1, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(out[1, :3, 3], torch.tensor([0.5, 0.0, -2.0]), atol=1e-5)
    assert torch.allclose(out[1, :3, :3], e0[:3, :3], atol=1e-5)


def test_generate_wobble_translates_in_image_plane_scaled_by_t():
    e = _look_at((0.0, 0.0, -2.0), (0.0, 0.0, 0.0))
    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    out = generate_wobble(e[None], torch.tensor([0.4]), t)
    assert out.shape == (1, 4, 4, 4)
    assert torch.allclose(out[0, 0], e)  # t = 0: radius * t = 0
    # camera-space offset (sin, -cos) * radius * t, expressed in world via the rotation
    for i, ti in enumerate(t.tolist()):
        r = 0.4 * ti
        cam_offset = torch.tensor(
            [
                torch.sin(torch.tensor(2 * torch.pi * ti)) * r,
                -torch.cos(torch.tensor(2 * torch.pi * ti)) * r,
                0.0,
            ]
        )
        expected = e[:3, 3] + e[:3, :3] @ cam_offset
        assert torch.allclose(out[0, i, :3, 3], expected, atol=1e-6)
        assert torch.allclose(out[0, i, :3, :3], e[:3, :3])
    unbatched = generate_wobble(e, torch.tensor(0.4), t)
    assert unbatched.shape == (4, 4, 4) and torch.equal(unbatched, out[0])
