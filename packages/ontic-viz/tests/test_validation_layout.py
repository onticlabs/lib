"""ontic_viz.validation leaf pieces: layout, labels, depth colouring, image conversion,
extras -> RGB, SH DC -> RGB. Pinned on small tensors."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ontic_viz.validation import (
    add_border,
    add_label,
    depth_map,
    extras_rgb,
    fig_to_image,
    hcat,
    prep_image,
    sh_dc_to_rgb,
    vcat,
)
from ontic_viz.validation.annotation import EXPECTED_CHARACTERS


def _img(c, h, w, value):
    return torch.full((c, h, w), float(value))


def test_hcat_pads_cross_axis_with_gap_color_and_aligns_start():
    a, b = _img(3, 2, 2, 0.0), _img(3, 4, 1, 0.5)
    out = hcat(a, b, gap=1, gap_color=1)
    assert out.shape == (3, 4, 4)  # 2 + gap 1 + 1 wide, 4 tall
    assert torch.equal(out[:, :2, :2], a)
    assert torch.all(out[:, 2:, :2] == 1)  # padding below a is gap colour
    assert torch.all(out[:, :, 2] == 1)  # separator column
    assert torch.equal(out[:, :, 3:], b)


def test_hcat_alignment_aliases():
    a, b = _img(3, 2, 2, 0.0), _img(3, 4, 1, 0.5)
    assert torch.equal(hcat(a, b, align="top"), hcat(a, b, align="start"))
    assert torch.equal(hcat(a, b, align="bottom"), hcat(a, b, align="end"))
    bottom = hcat(a, b, align="bottom", gap=0)
    assert torch.all(bottom[:, :2, :2] == 1) and torch.equal(bottom[:, 2:, :2], a)
    center = hcat(a, b, align="center", gap=0)
    assert torch.equal(center[:, 1:3, :2], a)


def test_vcat_and_gap_color_per_channel():
    a, b = _img(3, 1, 2, 0.0), _img(3, 1, 3, 0.5)
    out = vcat(a, b, gap=2, gap_color=[0.1, 0.2, 0.3])
    assert out.shape == (3, 4, 3)
    expected_gap = torch.tensor([0.1, 0.2, 0.3])[:, None, None].expand(3, 2, 3)
    assert torch.equal(out[:, 1:3, :], expected_gap)
    assert torch.equal(out[:, 0, 2], torch.tensor([0.1, 0.2, 0.3]))  # right padding of a
    assert torch.equal(vcat(a, b, align="left"), vcat(a, b, align="start"))
    assert torch.equal(vcat(a, b, align="right"), vcat(a, b, align="end"))


def test_add_border_default_and_custom():
    a = _img(3, 2, 3, 0.25)
    out = add_border(a)
    assert (
        out.shape == (3, 18, 19)
        and torch.all(out[:, :8] == 1)
        and torch.equal(out[:, 8:10, 8:11], a)
    )
    red = add_border(a, border=1, color=[1, 0, 0])
    assert red.shape == (3, 4, 5)
    assert torch.equal(red[:, 0, 0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(red[:, 1:3, 1:4], a)
    assert torch.all(add_border(a, 2, 0)[:, 0] == 0)


def test_add_label_puts_text_strip_above_image():
    pytest.importorskip("PIL")
    from PIL import ImageFont

    img = torch.rand(3, 10, 12, generator=torch.Generator().manual_seed(0))
    out = add_label(img, "RGB")
    font = ImageFont.load_default(24)
    _, top, _, bottom = font.getbbox(EXPECTED_CHARACTERS)
    label_h = bottom - top
    assert out.shape[0] == 3 and out.shape[1] == label_h + 4 + 10
    assert out.shape[2] >= 12
    assert torch.equal(out[:, label_h + 4 :, :12], img)  # image sits under the 4 px gap
    assert torch.all(out[:, label_h : label_h + 4] == 1)  # gap is white
    strip = out[:, :label_h]
    assert strip.min() < 0.5 < strip.max()  # black ink on white
    assert torch.equal(add_label(img, "RGB"), out)  # deterministic


def test_depth_map_pins_turbo_lut_entries_and_handles_bad_values():
    import matplotlib

    d = torch.tensor([[1.0, 2.0, 4.0, 8.0, 16.0]])
    out = depth_map(d)
    assert out.shape == (3, 1, 5) and out.dtype == torch.float32
    # near = log(quantile_0.01) = log(1.04), far = log(quantile_0.99) = log(15.68); the
    # normalised values 1.015, 0.759, 0.504, 0.248, -0.007 clip to LUT rows 255/194/128/63/0
    lut = matplotlib.colormaps["turbo"](np.array([255, 194, 128, 63, 0]))[:, :3]
    expected = torch.tensor(lut, dtype=torch.float32).T[:, None, :]
    assert torch.equal(out, expected)
    # constant input: normalisation divides by 0 -> NaN -> colormap 'bad' colour (black)
    const = depth_map(torch.full((2, 2), 3.0))
    assert const.shape == (3, 2, 2) and torch.all(const == 0)
    # zeros: no positive values -> min().clip(0).log() = -inf, all NaN -> black
    assert torch.all(depth_map(torch.zeros(2, 2)) == 0)
    # torch.quantile propagates NaN / inf into `far`, so one bad value blanks the whole map
    # (pre-existing behaviour, kept)
    for bad in (float("nan"), float("inf")):
        mixed = d.clone()
        mixed[0, 2] = bad
        assert torch.all(depth_map(mixed) == 0)
    # negatives: excluded from `near`, their log is NaN -> black; the rest is unaffected
    neg = depth_map(torch.tensor([[-1.0, 1.0, 2.0, 4.0]]))
    assert torch.all(neg[:, 0, 0] == 0)
    assert torch.equal(neg[:, 0, 1], expected[:, 0, 0]) and torch.equal(
        neg[:, 0, 3], expected[:, 0, 4]
    )
    # batched: leading dims preserved
    assert depth_map(torch.rand(2, 3, 4) + 0.1).shape == (2, 3, 3, 4)


def test_prep_image_layouts():
    gray = torch.tensor([[0.0, 0.5], [1.0, 2.0]])
    out = prep_image(gray)
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8
    assert (
        out[0, 1, 0] == 127 and out[1, 1, 0] == 255 and out[0, 0, 0] == 0
    )  # 0.5*255 = 127.5 -> 127
    rgb = torch.zeros(3, 2, 2)
    rgb[0] = 1
    assert (prep_image(rgb)[..., 0] == 255).all() and (prep_image(rgb)[..., 1:] == 0).all()
    batched = torch.stack([rgb, 1 - rgb])
    side = prep_image(batched)
    assert (
        side.shape == (2, 4, 3)
        and side[0, 0, 0] == 255
        and side[0, 2, 0] == 0
        and side[0, 2, 1] == 255
    )
    assert prep_image(torch.ones(4, 2, 2)).shape == (2, 2, 4)


def test_fig_to_image_matches_figure_pixel_size():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot([0, 1], [0, 1])
    img = fig_to_image(fig, dpi=100)
    plt.close(fig)
    assert img.shape == (3, 200, 300) and img.dtype == torch.float32
    assert img.min() >= 0 and img.max() <= 1
    assert torch.all(img[:, 0, 0] == 1)  # white figure background corner
    assert img.min() < 0.9  # something was drawn


def test_extras_rgb_rule():
    one = torch.rand(2, 1, 1, 3, 4)
    three = torch.rand(2, 1, 3, 3, 4)
    two = torch.rand(2, 1, 2, 3, 4)
    out = extras_rgb({"mask": one, "rgb": three, "feat": two})
    assert list(out) == ["mask", "rgb"]
    assert out["rgb"] is three
    assert out["mask"].shape == (2, 1, 3, 3, 4)
    assert torch.equal(out["mask"][:, :, 0:1], one) and torch.all(out["mask"][:, :, 1:] == 0)
    assert extras_rgb({}) == {} and extras_rgb(None) == {}


def test_sh_dc_to_rgb_uses_gsplat_constant_and_clamps():
    c0 = 0.28209479177387814
    harmonics = torch.zeros(3, 3, 4)
    harmonics[0, :, 0] = 0.0  # grey 0.5
    harmonics[1, :, 0] = 10.0  # clamps to 1
    harmonics[2, 0, 0] = 1.0 / c0  # red channel: 0.5 + 1 -> clamp 1; others 0.5
    harmonics[2, 1, 0] = -0.5 / c0  # green -> 0
    out = sh_dc_to_rgb(harmonics)
    assert isinstance(out, np.ndarray) and out.shape == (3, 3)
    np.testing.assert_allclose(out[0], [127.5, 127.5, 127.5])
    np.testing.assert_allclose(out[1], [255.0, 255.0, 255.0])
    np.testing.assert_allclose(out[2], [255.0, 0.0, 127.5], atol=1e-4)
    assert sh_dc_to_rgb(torch.zeros(2, 5, 3, 1)).shape == (2, 5, 3)
