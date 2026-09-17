"""Hue-log depth codec: exact code round trip, validity classes, torch == numpy."""

import numpy as np
import torch

from ontic_data import depth_codec as dc

LO, HI = 0.05, 5.0


def test_every_code_round_trips_exactly():
    # The top code decodes to exactly max_m, which float rounding can push past the far
    # plane (-> SKY), so the exact round trip covers every code below it.
    codes = np.arange(dc.GUARD, dc.GUARD + dc.LEVELS - 1)
    depth = dc.code_to_depth(codes, LO, HI)
    frame = dc.encode_depth_to_rgb(depth, LO, HI)
    assert np.array_equal(dc.decode_code(frame), codes)
    back, kind = dc.decode_rgb_to_depth(frame, LO, HI)
    assert np.all(kind == dc.VALID)
    assert np.allclose(back, depth, rtol=1e-6)


def test_quantisation_within_relative_step():
    rng = np.random.default_rng(0)
    depth = np.exp(rng.uniform(np.log(LO), np.log(HI), size=(32, 48)))
    back, kind = dc.decode_rgb_to_depth(dc.encode_depth_to_rgb(depth, LO, HI), LO, HI)
    assert np.all(kind == dc.VALID)
    assert np.all(np.abs(back / depth - 1) <= dc.relative_step(LO, HI))


def test_sky_and_hole_are_separable_and_read_zero():
    depth = np.array([[LO, HI, 800.0, 0.01, np.nan, np.inf]])
    frame = dc.encode_depth_to_rgb(depth, LO, HI)
    back, kind = dc.decode_rgb_to_depth(frame, LO, HI)
    assert kind.tolist() == [[dc.VALID, dc.VALID, dc.SKY, dc.HOLE, dc.HOLE, dc.SKY]]
    assert back[0, 2:].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert np.all(frame[0, 2] == dc.SKY_GREY) and np.all(frame[0, 3] == dc.HOLE_GREY)


def test_torch_decoder_matches_numpy():
    rng = np.random.default_rng(1)
    depth = np.exp(rng.uniform(np.log(LO) - 1, np.log(HI) + 1, size=(2, 16, 16)))
    frame = dc.encode_depth_to_rgb(depth, LO, HI)
    unit_np, kind_np = dc.decode_rgb_to_unit_log(frame)
    unit_t, kind_t = dc.decode_unit_log_torch(torch.from_numpy(frame))
    assert np.array_equal(kind_t.numpy(), kind_np)
    assert np.allclose(unit_t.numpy(), unit_np)
