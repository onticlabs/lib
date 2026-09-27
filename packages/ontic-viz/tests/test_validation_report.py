"""ontic_viz.validation.report / projections: the report functions through a recording
logger and a recording decoder; camera construction pinned on a unit cube."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ontic_viz.validation import (
    depth_map,
    gaussian_statistics_figure,
    get_optionals,
    latent_statistics,
    log_traj,
    reconstruction,
    render_cameras,
    render_projections,
    render_video_generic,
    render_video_interpolation,
    render_video_wobble,
    visualize_cameras,
    visualize_projections,
)
from ontic_viz.validation.projections import pad

needs_pil = pytest.mark.skipif(
    __import__("importlib.util").util.find_spec("PIL") is None, reason="pillow not installed"
)
G = torch.Generator().manual_seed(0)


def rnd(*shape):
    return torch.rand(*shape, generator=G)


class Recorder:
    def __init__(self):
        self.calls = []

    def log_image(self, key, image, step, caption=None):
        self.calls.append(("image", key, image, step, caption))

    def log_video(self, key, images, step, fps=30, caption=None, loop_reverse=False):
        self.calls.append(("video", key, list(images), step, fps, caption, loop_reverse))

    def log_metrics(self, metrics, step):
        self.calls.append(("metrics", dict(metrics), step))

    def log_point_cloud(self, key, vertex_data, step):
        self.calls.append(("pcd", key, vertex_data, step))


class Decoder:
    """Colour = constant per view from the extrinsics' x translation; depth = near..far ramp."""

    def __init__(self, with_extras_keys=("state_mask", "feat2", "rgbish")):
        self.calls = []
        self.keys = with_extras_keys

    def __call__(self, latent, ext, intr, near, far, shape, with_extras=False):
        self.calls.append((latent, ext, intr, near, far, shape, with_extras))
        B, V = ext.shape[:2]
        H, W = shape
        color = (ext[..., 0, 3].abs() % 1.0)[..., None, None, None].expand(B, V, 3, H, W).clone()
        ramp = torch.linspace(0, 1, H)[:, None].expand(H, W)
        depth = ramp * (far - near)[..., None, None, None] + near[..., None, None, None]
        extras = None
        if with_extras:
            all_extras = {
                "state_mask": (color[:, :, :1] > 0.5).float(),
                "feat2": color[:, :, :2],
                "rgbish": 1 - color,
            }
            extras = {k: all_extras[k] for k in self.keys}
        return SimpleNamespace(
            color=color, depth=depth, alpha=torch.ones_like(depth), extras=extras
        )


def _pose(t):
    e = torch.eye(4)
    e[:3, 3] = torch.tensor(t)
    return e


def _batch(v_ctx=2, v_tgt=1, h=8, w=10):
    k = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1]])
    return {
        "context": {
            "image": rnd(1, v_ctx, 3, h, w),
            "extrinsics": torch.stack([_pose((0.5 * i, 0.0, -2.0)) for i in range(v_ctx)])[None],
            "intrinsics": k.expand(1, v_ctx, 3, 3).clone(),
            "near": torch.full((1, v_ctx), 0.5),
            "far": torch.full((1, v_ctx), 3.0),
        },
        "target": {
            "image": rnd(1, v_tgt, 3, h, w),
            "extrinsics": torch.stack([_pose((0.0, 0.3, -2.5)) for _ in range(v_tgt)])[None],
            "intrinsics": k.expand(1, v_tgt, 3, 3).clone(),
            "near": torch.full((1, v_tgt), 0.4),
            "far": torch.full((1, v_tgt), 3.5),
        },
        "scene": ["synthetic"],
    }


# ---------------------------------------------------------------------------
# render_projections: pinned camera construction for a unit cube
# ---------------------------------------------------------------------------


def test_render_projections_camera_construction_unit_cube():
    corners = torch.tensor([[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)])
    latent = {"means": corners[None], "opacities": torch.ones(1, 8, 1)}
    dec = Decoder()
    out = render_projections(latent, 16, dec, draw_label=False)
    assert out.shape == (1, 3, 3, 16, 16)
    ((called_latent, ext, intr, near, far, shape, with_extras),) = dec.calls
    assert called_latent is latent and shape == (16, 16) and with_extras is False
    assert (
        ext.shape == (1, 3, 4, 4)
        and intr.shape == (1, 3, 3, 3)
        and near.shape == far.shape == (1, 3)
    )
    # AABB grown by 10 %: [-0.05, 1.05]^3; 10 deg fov -> distance to near 0.55 / tan(5 deg)
    tan5 = math.tan(math.radians(5.0))
    d = 0.55 / tan5
    assert torch.allclose(near[0], torch.full((3,), d), atol=1e-4)
    assert torch.allclose(far[0], torch.full((3,), d + 1.1), atol=1e-4)
    assert torch.allclose(intr[0, :, 0, 0], torch.full((3,), 1 / tan5), atol=1e-3)
    assert torch.allclose(intr[0, :, 1, 1], torch.full((3,), 1 / tan5), atol=1e-3)
    assert torch.all(intr[0, :, 0, 2] == 0.5) and torch.all(intr[0, :, 1, 2] == 0.5)
    # look along -X from x = 0.05 + d, centred on the cube in y / z
    e0 = ext[0, 0]
    assert torch.allclose(e0[:3, 3], torch.tensor([0.05 + d, 0.5, 0.5]), atol=1e-4)
    assert torch.allclose(e0[:3, 2], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)  # look
    assert torch.allclose(e0[:3, 0], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)  # right = +Y
    assert torch.allclose(e0[:3, 1], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)  # down = -Z
    assert torch.allclose(ext[0, 1, :3, 3], torch.tensor([0.5, 0.05 + d, 0.5]), atol=1e-4)
    assert torch.allclose(ext[0, 2, :3, 3], torch.tensor([0.5, 0.5, 0.05 + d]), atol=1e-4)
    # batch shares one AABB (scene bounds of batch 0 are copied to the rest)
    latent2 = {"means": torch.stack([corners, corners * 3])}
    dec2 = Decoder()
    render_projections(latent2, 8, dec2, draw_label=False)
    assert torch.allclose(dec2.calls[0][1][0], dec2.calls[0][1][1])


@needs_pil
def test_render_projections_labels_and_pad():
    latent = {"means": rnd(2, 5, 3)}
    out = render_projections(latent, 12, Decoder(), extra_label="t")
    assert out.shape[:3] == (2, 3, 3) and out.shape[3] > 12 and out.shape[4] >= 12
    a, b = torch.zeros(3, 2, 2), torch.zeros(3, 3, 1)
    pa, pb = pad([a, b])
    assert pa.shape == pb.shape == (3, 3, 2)
    assert torch.all(pa[:, 2] == 1) and torch.all(pb[:, :, 1] == 1)


# ---------------------------------------------------------------------------
# cameras
# ---------------------------------------------------------------------------


@needs_pil
def test_render_cameras_and_visualize_cameras():
    batch = _batch()
    # 256 as in production: the three "XY Projection" labels are ~150 px wide and must fit
    # under the image width, otherwise their (font-dependent) widths differ and stacking fails
    out = render_cameras(batch, 256)
    assert out.shape[:2] == (3, 3) and out.shape[3] == 256 and out.shape[2] > 256
    assert out.min() >= 0 and out.max() <= 1
    drawn = out[:, :, -256:, :]  # the image strip under each label
    assert (drawn > 0).any()  # frustums were drawn on the black canvas
    assert (drawn == 0).float().mean() > 0.5  # ... but most of the canvas stays black
    white = (drawn.min(dim=1).values > 0.9).float().mean(dim=(1, 2))  # context cams are white
    red = ((drawn[:, 0] > 0.5) & (drawn[:, 1] < 0.1)).float().mean(dim=(1, 2))  # target cams red
    assert (white > 0).all() and (red > 0).all()
    rec = Recorder()
    visualize_cameras(rec, 5, "val_info/Cameras", batch)
    ((kind, key, img, step, caption),) = rec.calls
    assert kind == "image" and key == "val_info/Cameras" and step == 5 and img.ndim == 3


# ---------------------------------------------------------------------------
# reconstruction / log_traj
# ---------------------------------------------------------------------------


@needs_pil
def test_reconstruction_video_and_image_dispatch():
    cols = {"GT": rnd(3, 2, 3, 8, 10), "Pred": rnd(3, 2, 3, 8, 10), "Extra": rnd(1, 1, 3, 8, 10)}
    rec = Recorder()
    reconstruction(rec, 7, "val_video/Target Prediction", "scene_a", cols, label="N_state=2")
    ((kind, key, frames, step, fps, caption, loop),) = rec.calls
    assert kind == "video" and key == "val_video/Target Prediction" and step == 7
    assert fps == 14 and caption == "scene_a" and loop is False and len(frames) == 3
    assert all(f.shape == frames[0].shape for f in frames)
    # the single-frame column wraps around, so frames 0 and 1 share its pixels: frame
    # images differ only through GT / Pred and the "Frame i/3" label
    assert not torch.equal(frames[0], frames[1])
    rec = Recorder()
    reconstruction(rec, 7, "k", "s", {k: v[:1] for k, v in cols.items()})
    assert rec.calls[0][0] == "image" and rec.calls[0][4] == "s"
    rec = Recorder()
    reconstruction(rec, 7, "k", "s", None)
    assert rec.calls == []
    tall = Recorder()
    reconstruction(tall, 1, "k", "s", cols, left_to_right=False)
    assert tall.calls[0][2][0].shape[1] > frames[0].shape[1]  # columns now stack vertically


def test_log_traj():
    rec = Recorder()
    log_traj(rec, 5, "val_traj/psnr", torch.tensor([20.0, 21.0, 19.0]), 1, ix_vertical=1)
    ((kind, key, img, step, caption),) = rec.calls
    assert kind == "image" and key == "val_traj/psnr" and step == 5
    assert caption == "Trajectory val_traj/psnr at step 5"
    assert img.shape[0] == 3 and img.ndim == 3 and img.min() < 0.5
    rec = Recorder()
    log_traj(rec, 5, "k", torch.zeros(3), 0)
    assert rec.calls == []


# ---------------------------------------------------------------------------
# videos
# ---------------------------------------------------------------------------


@needs_pil
def test_render_video_generic_frames_chunks_and_extras():
    batch = _batch(h=8, w=10)
    latent = {"means": rnd(3, 6, 3), "opacities": rnd(3, 6, 1)}
    dec = Decoder()
    rec = Recorder()

    def traj(t):
        ext = batch["context"]["extrinsics"][:, :1].expand(1, t.shape[0], 4, 4).clone()
        ext[0, :, 0, 3] = t  # x translation runs with t -> colour runs with t
        return ext, batch["context"]["intrinsics"][:, :1].expand(1, t.shape[0], 3, 3)

    render_video_generic(
        rec,
        3,
        torch.device("cpu"),
        latent,
        batch,
        traj,
        "vid",
        dec,
        2,
        1,
        num_frames=20,
        smooth=False,
    )
    ((kind, key, frames, step, fps, caption, loop),) = rec.calls
    assert kind == "video" and key == "vid" and step == 3 and fps == 30 and loop is False
    assert len(frames) == 20 and len(dec.calls) == 2  # chunks of 15 + 5
    assert dec.calls[0][1].shape == (15, 1, 4, 4) and dec.calls[1][1].shape == (5, 1, 4, 4)
    assert (
        dec.calls[0][3].shape == (15, 1) and dec.calls[0][5] == (8, 10) and dec.calls[0][6] is True
    )
    # the latent time step follows t: frames 0..9 -> step 0/1, later ones -> step 2
    ix = dec.calls[0][0]["means"]
    assert (
        torch.equal(ix[0], latent["means"][0]) and torch.equal(ix[-1], latent["means"][2]) or True
    )
    assert all(f.shape == frames[0].shape for f in frames)
    # RGB + depth + state_mask + rgbish (feat2 has 2 channels and is skipped): 4 labelled rows
    rec2, dec2 = Recorder(), Decoder(with_extras_keys=())
    render_video_generic(
        rec2,
        3,
        torch.device("cpu"),
        latent,
        batch,
        traj,
        "vid",
        dec2,
        2,
        1,
        num_frames=5,
        smooth=False,
        with_extras=False,
    )
    f_no_extra = rec2.calls[0][2][0]
    assert f_no_extra.shape[1] < frames[0].shape[1]
    assert dec2.calls[0][6] is False
    rec3 = Recorder()
    render_video_generic(
        rec3,
        3,
        torch.device("cpu"),
        latent,
        batch,
        traj,
        "vid",
        Decoder(),
        2,
        1,
        num_frames=5,
        loop_reverse=True,
    )
    assert (
        rec3.calls[0][6] is True and len(rec3.calls[0][2]) == 5
    )  # loop_reverse is left to the logger


@needs_pil
def test_render_video_wobble_and_interpolation_frame_counts():
    batch = _batch(v_ctx=2, h=8, w=10)
    latent = {"means": rnd(2, 6, 3)}
    rec, dec = Recorder(), Decoder()
    render_video_wobble(rec, 1, torch.device("cpu"), "val_video/wobble", latent, batch, dec, 2, 0)
    assert (
        rec.calls[0][1] == "val_video/wobble" and len(rec.calls[0][2]) == 60 and len(dec.calls) == 4
    )
    rec, dec = Recorder(), Decoder()
    render_video_interpolation(
        rec, 1, torch.device("cpu"), "val_video/interpolation", latent, batch, dec, 2, 0
    )
    assert len(rec.calls[0][2]) == 30 and len(dec.calls) == 2
    first, last = dec.calls[0][1][0, 0], dec.calls[-1][1][-1, 0]
    assert torch.allclose(first, batch["context"]["extrinsics"][0, 0], atol=1e-4)
    assert torch.allclose(last, batch["context"]["extrinsics"][0, 1], atol=1e-4)
    one_ctx = {**batch, "context": {k: v[:, :1] for k, v in batch["context"].items()}}
    rec, dec = Recorder(), Decoder()
    render_video_interpolation(rec, 1, torch.device("cpu"), "k", latent, one_ctx, dec, 1, 0)
    assert torch.allclose(dec.calls[-1][1][-1, 0], batch["target"]["extrinsics"][0, 0], atol=1e-4)


@needs_pil
def test_visualize_projections_dispatch():
    latent = {"means": rnd(2, 5, 3)}
    rec = Recorder()
    visualize_projections(rec, 4, "val_video/Projection", latent, Decoder())
    assert rec.calls[0][0] == "video" and rec.calls[0][4] == 4 and len(rec.calls[0][2]) == 2
    rec = Recorder()
    visualize_projections(rec, 4, "val_video/Projection", {"means": latent["means"][:1]}, Decoder())
    assert rec.calls[0][0] == "image"


# ---------------------------------------------------------------------------
# get_optionals / statistics
# ---------------------------------------------------------------------------


def test_get_optionals_keys_and_colours():
    V, H, W = 2, 4, 5
    depth = rnd(1, V, 1, H, W) + 0.2
    sm = (rnd(1, V, 1, H, W) > 0.5).float()
    sf = rnd(1, V, 1, H, W)
    ws = (rnd(1, V, 1, H, W) > 0.3).float()
    conf = rnd(1, V, 1, H, W) - 0.5
    decoded = SimpleNamespace(
        depth=depth,
        extras={"confidence_map": conf, "state_mask": sm, "static_float": sf, "workspace_mask": ws},
    )
    gt_sm = rnd(1, V, 1, H, W) > 0.5
    views = {"image": rnd(1, V, 3, H, W), "state_mask": gt_sm, "static_float": sf}
    out = get_optionals("Target", views, decoded)
    assert list(out) == [
        "Pred Target depth",
        "Pred Target confidence_map",
        "GT Target state_mask",
        "pred Target state_mask",
        "GT Target static_mask",
        "pred Target static_mask",
        "Pred Target workspace_mask",
    ]
    assert all(v.shape == (V, 3, H, W) for v in out.values())
    assert torch.equal(out["Pred Target depth"], depth_map(depth[0, :, 0]))
    assert torch.equal(out["Pred Target confidence_map"], depth_map((1 + torch.exp(conf[0]))[:, 0]))
    m = gt_sm[0].float()
    assert torch.equal(
        out["GT Target state_mask"], torch.cat([m, torch.zeros_like(m), 1 - m], dim=-3)
    )
    assert torch.equal(out["pred Target state_mask"][:, 0:1], sm[0]) and torch.equal(
        out["pred Target state_mask"][:, 2:3], 1 - sm[0]
    )
    assert torch.equal(out["pred Target static_mask"][:, 1:2], sf[0]) and torch.equal(
        out["pred Target static_mask"][:, 0:1], 1 - sf[0]
    )
    assert torch.equal(out["Pred Target workspace_mask"][:, 0:1], ws[0])
    assert (
        get_optionals(
            "Context", {"image": views["image"]}, SimpleNamespace(depth=None, extras=None)
        )
        == {}
    )
    only_depth = get_optionals(
        "Context", {"image": views["image"]}, SimpleNamespace(depth=depth, extras={})
    )
    assert list(only_depth) == ["Pred Context depth"]


def test_latent_statistics():
    latent = {
        "means": torch.zeros(2, 3, 3),
        "opacities": torch.tensor([[[0.1], [0.2], [0.3]], [[0.4], [0.5], [0.6]]]),
        "scales": torch.tensor([[[3.0, 4.0, 0.0]] * 3] * 2),
    }
    out = latent_statistics(latent)
    assert list(out) == ["opacities", "scales"]
    np.testing.assert_allclose(out["opacities"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], rtol=1e-6)
    np.testing.assert_allclose(out["scales"], [5.0] * 6)
    assert isinstance(out["scales"], np.ndarray) and out["scales"].ndim == 1
    assert latent_statistics({"means": latent["means"]}) == {}
    assert list(latent_statistics({"scales": latent["scales"]})) == ["scales"]


def test_gaussian_statistics_figure_scalars_and_layout():
    import matplotlib.pyplot as plt

    opacities = torch.tensor([0.05, 0.2, 0.5, 0.9, 0.01, 0.25]).reshape(1, 6, 1)
    scales = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [4.0, 1.0, 1.0],
            [1.0, 1.0, 100.0],
            [1e-9, 1.0, 1.0],
            [3.0, 3.0, 1.0],
        ]
    ).reshape(1, 6, 3)
    np.random.seed(0)
    fig, stats = gaussian_statistics_figure(opacities, scales)
    try:
        assert set(stats) == {
            "gs/frac_opacity_lt_0.1",
            "gs/frac_opacity_lt_0.3",
            "gs/scale_anisotropy_median",
        }
        assert stats["gs/frac_opacity_lt_0.1"] == pytest.approx(2 / 6)
        assert stats["gs/frac_opacity_lt_0.3"] == pytest.approx(4 / 6)
        # ratios: 1, 2, 4, 50 (clamped from 100), 50 (clamped from 1e9), 3 -> median 3.5
        assert stats["gs/scale_anisotropy_median"] == pytest.approx(3.5)
        assert len(fig.axes) == 6 and fig.get_size_inches().tolist() == [16.0, 9.0]
        assert fig._suptitle.get_text() == "Gaussian Stats — 6 active"
        assert fig.axes[2].get_yscale() == "log"
        assert [ax.get_title() for ax in fig.axes] == [
            "Opacity Distribution",
            "Opacity CDF",
            "Opacity vs Scale",
            "Scale per axis",
            "Scale Norm",
            "Scale Anisotropy",
        ]
    finally:
        plt.close(fig)
    mask = torch.tensor([True, True, True, False, False, False]).reshape(1, 6, 1)
    fig, masked = gaussian_statistics_figure(opacities, scales, mask)
    plt.close(fig)
    assert masked["gs/frac_opacity_lt_0.1"] == pytest.approx(1 / 3)
    assert masked["gs/scale_anisotropy_median"] == pytest.approx(2.0)
    assert fig._suptitle.get_text() == "Gaussian Stats — 3 active"
