# Vendored verbatim from molmospaces_harness/datagen/depth_codec.py
# (github.com/onticlabs/robotics, packages/molmospaces-harness) -- the module that WROTE
# the SynthRobot depth videos, so it is the authoritative decoder. Re-sync by re-copying
# upstream; do not edit or reimplement.
"""Put depth into an 8-bit RGB video without the low byte turning into noise.

Upstream packs 16-bit linear depth into the R and G channels: R is the high
byte, G the low byte, B unused. That is a reasonable-looking scheme and it is
badly wrong once the frames go through a lossy codec, because the low byte is a
sawtooth that wraps 256 times across the range -- pixel-scale noise by
construction. x264 cannot preserve it and spends most of the bitrate trying.
Measured on recorded ``astribot_pick_review`` episodes at the shipping settings
(``libx264rgb``, CRF 23, 0.05-5 m):

===============  ==========  ===============  =============
camera           file        median error     p95 relative
===============  ==========  ===============  =============
head             2.58 MB     18.9 mm          3.7%
wrist            1.05 MB     --               7.0% (13% under 0.5 m)
===============  ==========  ===============  =============

18.9 mm is exactly one high-byte step (4.95 m / 256). The low byte arrives as
garbage, so the advertised 76 um precision is fiction, and the depth videos come
out 8.8x larger than the *colour* videos of the same scene.

What this module writes instead: **log depth on the hue wheel.**

- **Log, not linear or disparity.** Training wants relative accuracy, and a log
  ramp spends precision evenly in relative terms -- measured p95 stays within
  0.37-0.56% from 0.24 m out to 4 m, on head, wrist and ring cameras alike.
  Linear is worst exactly where it matters (13% under 0.5 m on a wrist camera).
  Disparity over-serves the near field: it was 5x worse than log on a head
  camera, because it crowds most of its codes into 0.05-0.3 m, which for a room
  camera is empty.
- **Hue, not byte-packing.** Walking the code around the colour wheel keeps
  every channel a slow trapezoid instead of a sawtooth: at any depth exactly one
  channel is ramping and the other two are pinned at 0 or 255. Adjacent depths
  are adjacent colours, so a compression error is a *small* depth error rather
  than a jump. This is Intel's colorization scheme (*Depth image compression by
  colorization for Intel RealSense Depth Cameras*, 2019), which ships in
  librealsense.

Two things a textbook hue implementation gets wrong, both measured here:

- **The wheel wraps, and both ends are red.** Code 0 is ``(255,0,0)`` and code
  1529 is ``(255,0,1)``; they are told apart by whether green exceeds blue, so a
  one-step compression error turns 5 m into 5 cm. On a wrist camera that showed
  up as a 9810% error. :data:`GUARD` codes are left unused at each end, which
  drops gross (>25%) errors from 0.28% of pixels to 0.0007%. Widening the guard
  past 32 buys nothing further and costs precision, since the remaining codes
  have to cover the same range.
- **Invalid needs to be told from valid by desaturation, not by darkness.** The
  obvious test -- "is it nearly black?" -- misreads a valid pixel whose one
  saturated channel gets rung down next to a hole; that alone accounted for
  0.0466% of pixels. Every valid colour instead has ``max == 255`` and
  ``min == 0``, so :func:`decode_validity` scores ``min + (255 - max)``, which is
  0 on the hue curve and 255 at any grey. That takes flips to 0.0001%. The score
  counts dimness as well as desaturation, so a valid colour uniformly darkened
  past ~50% crosses the cut -- deliberate, since by then it is as close to
  :data:`HOLE_GREY` as to the curve, and far outside what ringing produces.

The valid colours are a 1-D curve of 1530 points through the RGB cube, so 99.99%
of the cube is free, and one reserved grey would be a waste of it. Two are used,
which recovers a distinction the old encoding threw away: a ray that hits nothing
comes back from the renderer at ~826 m, and that is an open window, not a failed
measurement. :data:`SKY_GREY` and :data:`HOLE_GREY` separate them, and they stay
separable -- 0.00000% confusion at CRF 15.

What is *not* spent on more depth levels: the wheel quantises to 0.34% a step
while the codec delivers ~0.5%, so the codec is the limit and extra levels would
buy nothing. The spare colour space is better spent on margin.

Sizes land about where the old scheme did (head 1.12 MB against 2.58, wrist 1.28
against 1.05) while relative error improves by roughly an order of magnitude.
Everything above was measured against frames decoded from already-compressed
recordings, so real sim depth -- which is smoother -- should compress better than
these numbers suggest, not worse.
"""

from __future__ import annotations

import numpy as np

#: Codes on the full hue wheel: six segments of 255.
HUE_CODES = 1530

#: Codes left unused at each end of the wheel, so the red-to-red wrap can never
#: be crossed by compression noise. 32 is where the gross-error rate flattens.
GUARD = 32

#: Codes actually carrying depth, ``GUARD`` through ``GUARD + LEVELS - 1``.
LEVELS = HUE_CODES - 2 * GUARD

#: Reserved greys. Every valid colour is fully saturated, so any grey is
#: maximally far from all of them; these two are far from each other as well.
SKY_GREY = 192  #: Ray hit nothing -- open window, past the far plane.
HOLE_GREY = 64  #: No usable measurement -- nearer than the floor, or not finite.

#: ``min + (255 - max)`` above this reads as invalid. 0 on the hue curve, 255 on
#: any grey, so the cut sits halfway with margin either side.
DESATURATION_CUT = 128

#: Luma above this picks :data:`SKY_GREY` over :data:`HOLE_GREY`.
GREY_CUT = (SKY_GREY + HOLE_GREY) // 2

VALID, SKY, HOLE = 0, 1, 2


#: ``(argmax, argmin) -> (segment, does the middle channel count down)``. Which
#: channel is brightest and which darkest is what places a colour on the wheel,
#: and both survive ringing far better than comparing values at a segment join.
SEGMENTS = [
    ((0, 2), 0, False),  # red     -> yellow
    ((1, 2), 1, True),  # yellow  -> green
    ((1, 0), 2, False),  # green   -> cyan
    ((2, 0), 3, True),  # cyan    -> blue
    ((2, 1), 4, False),  # blue    -> magenta
    ((0, 1), 5, True),  # magenta -> red
]


def _build_wheel() -> np.ndarray:
    """``(1530, 3)`` uint8: the hue curve, one row per code."""
    wheel = np.zeros((HUE_CODES, 3), dtype=np.uint8)
    ramp = np.arange(255)
    full = np.full(255, 255)
    zero = np.zeros(255, dtype=int)
    segments = [
        (full, ramp, zero),  # red     -> yellow
        (255 - ramp, full, zero),  # yellow  -> green
        (zero, full, ramp),  # green   -> cyan
        (zero, 255 - ramp, full),  # cyan    -> blue
        (ramp, zero, full),  # blue    -> magenta
        (full, zero, 255 - ramp),  # magenta -> red
    ]
    for index, (red, green, blue) in enumerate(segments):
        wheel[index * 255 : (index + 1) * 255] = np.stack([red, green, blue], axis=-1)
    return wheel


WHEEL = _build_wheel()


def depth_to_code(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """Log-spaced code in ``[GUARD, GUARD + LEVELS - 1]``, for in-range depth only."""
    span = np.log(max_m / min_m)
    # Non-finite depth carries no code -- classify() sends it to a reserved grey
    # and the value here is never read. It still has to not be NaN, or the cast
    # to an index warns and lands somewhere arbitrary.
    finite = np.nan_to_num(
        np.asarray(depth_m, dtype=np.float64), nan=min_m, posinf=max_m, neginf=min_m
    )
    unit = np.log(np.clip(finite, min_m, max_m) / min_m) / span
    return (np.round(unit * (LEVELS - 1)).astype(np.int64) + GUARD).astype(np.int64)


def code_to_depth(code: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """Metres for a code, the inverse of :func:`depth_to_code`."""
    unit = np.clip((np.asarray(code, dtype=np.float64) - GUARD) / (LEVELS - 1), 0.0, 1.0)
    return min_m * np.exp(unit * np.log(max_m / min_m))


def classify(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """Split depth into :data:`VALID` / :data:`SKY` / :data:`HOLE`.

    Past the far plane is sky: the renderer returns ~826 m for a ray that hit
    nothing, and an open window is a real observation -- ``+inf`` means the same
    thing and is counted with it. Nearer than the floor, or NaN, is a hole: no
    usable measurement, which is what 0 has always meant here.
    """
    depth = np.asarray(depth_m, dtype=np.float64)
    kind = np.full(depth.shape, VALID, dtype=np.uint8)
    with np.errstate(invalid="ignore"):
        kind[depth > max_m] = SKY
        kind[depth < min_m] = HOLE
    kind[np.isnan(depth)] = HOLE
    return kind


def encode_depth_to_rgb(depth_m: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
    """Metric depth ``(..., H, W)`` -> ``(..., H, W, 3)`` uint8 ready for the codec."""
    if not max_m > min_m > 0:
        raise ValueError(f"need 0 < min < max, got {min_m}..{max_m}")
    depth = np.asarray(depth_m, dtype=np.float64)
    frame = WHEEL[np.clip(depth_to_code(depth, min_m, max_m), 0, HUE_CODES - 1)]
    kind = classify(depth, min_m, max_m)
    frame[kind == SKY] = SKY_GREY
    frame[kind == HOLE] = HOLE_GREY
    return frame


def decode_validity(frame: np.ndarray) -> np.ndarray:
    """``(..., H, W)`` of :data:`VALID` / :data:`SKY` / :data:`HOLE`.

    Distance from the saturated hue curve, not darkness -- see the module
    docstring for why the obvious test misreads pixels beside a hole.
    """
    rgb = np.asarray(frame)
    low = rgb.min(axis=-1).astype(np.int32)
    high = rgb.max(axis=-1).astype(np.int32)
    invalid = (low + (255 - high)) > DESATURATION_CUT
    luma = rgb.mean(axis=-1)
    return np.where(invalid, np.where(luma > GREY_CUT, SKY, HOLE), VALID).astype(np.uint8)


def decode_code(frame: np.ndarray) -> np.ndarray:
    """Recover the hue code, clamped into the guarded band.

    Which segment a colour is on follows from *which* channel is the maximum and
    which the minimum -- both stable under the ringing a codec adds, unlike the
    value comparisons a naive inverse makes at the segment joins.
    """
    rgb = np.asarray(frame).astype(np.int32)
    high = rgb.argmax(axis=-1)
    low = rgb.argmin(axis=-1)
    total = rgb.sum(axis=-1)
    mid = total - rgb.max(axis=-1) - rgb.min(axis=-1)

    code = np.zeros(rgb.shape[:-1], dtype=np.int64)
    for (top, bottom), segment, descending in SEGMENTS:
        on = (high == top) & (low == bottom)
        step = (255 - mid) if descending else mid
        code = np.where(on, segment * 255 + step, code)
    return np.clip(code, GUARD, GUARD + LEVELS - 1)


def decode_rgb_to_depth(
    frame: np.ndarray, min_m: float, max_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """``(..., H, W, 3)`` uint8 -> ``(depth_m, kind)``.

    Invalid pixels read 0.0 metres, as they always have; ``kind`` says whether
    that 0 is sky or a hole.
    """
    kind = decode_validity(frame)
    depth = code_to_depth(decode_code(frame), min_m, max_m)
    return np.where(kind == VALID, depth, 0.0).astype(np.float32), kind


def decode_rgb_to_unit_log(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(..., H, W, 3)`` uint8 -> ``(unit_log_depth, kind)``, no metres involved.

    The hue code *is* log depth, so a network training on log depth wants this
    and never needs the exponential. ``unit`` runs 0 at ``min_m`` to 1 at
    ``max_m``; recover metres with :func:`code_to_depth` only where geometry is
    actually needed.
    """
    kind = decode_validity(frame)
    unit = (decode_code(frame).astype(np.float32) - GUARD) / (LEVELS - 1)
    return np.where(kind == VALID, unit, 0.0).astype(np.float32), kind


def relative_step(min_m: float, max_m: float) -> float:
    """Quantisation step as a fraction of depth.

    Constant across the range, that being the point of the log ramp. Well under
    what the codec itself contributes (~0.5% at CRF 15), so this is not the term
    that limits accuracy.
    """
    return float(np.expm1(np.log(max_m / min_m) / (LEVELS - 1)))


def decode_unit_log_torch(frames):
    """The training-side decoder: uint8 frames straight off decord -> unit log depth.

    Takes ``(..., H, W, 3)`` uint8 on any device and returns ``(unit, kind)``
    without leaving it, which is the point -- decord's torch bridge hands over a
    tensor, and going via numpy to decode would drag it back off the GPU.

    ``unit`` is normalised log depth, 0 at ``min_m`` and 1 at ``max_m``. That is
    what the hue code already is, so a network training on log depth consumes
    this directly and the exponential never runs. Invalid pixels read 0.0; take
    them from ``kind``, not from the value, since 0.0 is also a legitimate reading
    at ``min_m``.

    Kept deliberately equivalent to :func:`decode_rgb_to_unit_log` --
    ``tests/test_depth_codec.py`` holds the two to the same answers.
    """
    import torch

    rgb = frames.to(torch.int32)
    high_value, high = rgb.max(dim=-1)
    low_value, low = rgb.min(dim=-1)
    middle = rgb.sum(dim=-1) - high_value - low_value

    invalid = (low_value + (255 - high_value)) > DESATURATION_CUT
    luma = rgb.to(torch.float32).mean(dim=-1)
    kind = torch.where(
        invalid,
        torch.where(luma > GREY_CUT, torch.tensor(SKY), torch.tensor(HOLE)).to(rgb.device),
        torch.tensor(VALID).to(rgb.device),
    ).to(torch.uint8)

    code = torch.zeros_like(high_value)
    for (top, bottom), segment, descending in SEGMENTS:
        on = (high == top) & (low == bottom)
        step = (255 - middle) if descending else middle
        code = torch.where(on, segment * 255 + step, code)
    code = code.clamp(GUARD, GUARD + LEVELS - 1)

    unit = (code - GUARD).to(torch.float32) / (LEVELS - 1)
    return torch.where(kind == VALID, unit, torch.zeros_like(unit)), kind
