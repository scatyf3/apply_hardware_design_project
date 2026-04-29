"""Tests for the rectify golden model.

Checks:
 1. rectify() and rectify_vectorized() agree bit-for-bit on the same math.
 2. Identity maps: output is a pixel-exact copy of the source.
 3. Output shape correctness for non-square maps.
 4. Out-of-bounds coordinates are filled with the border value.
 5. Numerical sanity against a hand-computed bilinear example.
 6. (Optional) close agreement with cv2.remap when cv2 is available.
    Skipped if cv2 isn't installed — the golden model does not depend on it.
"""

import numpy as np
import pytest

from rectify import make_identity_maps, rectify, rectify_vectorized


def _rand_img(h, w, c=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (h, w) if c is None else (h, w, c)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


# ---------------------------------------------------------------------------
# 1. loop vs vectorised
# ---------------------------------------------------------------------------

def test_loop_vs_vectorized_gray():
    img = _rand_img(24, 32, seed=10)
    map_x, map_y = make_identity_maps(24, 32)
    # Add small sub-pixel offsets to exercise the bilinear path
    rng = np.random.default_rng(99)
    map_x = map_x + rng.uniform(-0.4, 0.4, map_x.shape).astype(np.float32)
    map_y = map_y + rng.uniform(-0.4, 0.4, map_y.shape).astype(np.float32)

    a = rectify(img, map_x, map_y)
    b = rectify_vectorized(img, map_x, map_y)
    assert a.shape == b.shape
    assert np.array_equal(a, b), "loop and vectorized variants disagree (gray)"


def test_loop_vs_vectorized_rgb():
    img = _rand_img(20, 30, c=3, seed=11)
    map_x, map_y = make_identity_maps(16, 24)

    # Fractional maps pointing into the source image
    rng = np.random.default_rng(42)
    map_x = rng.uniform(0, 29, (16, 24)).astype(np.float32)
    map_y = rng.uniform(0, 19, (16, 24)).astype(np.float32)

    a = rectify(img, map_x, map_y)
    b = rectify_vectorized(img, map_x, map_y)
    assert a.shape == b.shape == (16, 24, 3)
    assert np.array_equal(a, b), "loop and vectorized variants disagree (RGB)"


# ---------------------------------------------------------------------------
# 2. Identity maps
# ---------------------------------------------------------------------------

def test_identity_gray():
    img = _rand_img(32, 48, seed=2)
    map_x, map_y = make_identity_maps(32, 48)
    out = rectify_vectorized(img, map_x, map_y)
    assert np.array_equal(out, img), "identity map should reproduce source exactly"


def test_identity_rgb():
    img = _rand_img(32, 48, c=3, seed=3)
    map_x, map_y = make_identity_maps(32, 48)
    out = rectify_vectorized(img, map_x, map_y)
    assert np.array_equal(out, img), "identity map (RGB) should reproduce source exactly"


# ---------------------------------------------------------------------------
# 3. Output shape
# ---------------------------------------------------------------------------

def test_output_shape():
    img = _rand_img(40, 60, c=3, seed=4)
    H_out, W_out = 25, 35
    map_x = np.zeros((H_out, W_out), dtype=np.float32)
    map_y = np.zeros((H_out, W_out), dtype=np.float32)
    out = rectify_vectorized(img, map_x, map_y)
    assert out.shape == (H_out, W_out, 3)


# ---------------------------------------------------------------------------
# 4. Out-of-bounds border fill
# ---------------------------------------------------------------------------

def test_oob_border_value():
    img = _rand_img(10, 10, seed=5)
    H_out, W_out = 4, 4
    # All coordinates point outside the source
    map_x = np.full((H_out, W_out), -5.0, dtype=np.float32)
    map_y = np.full((H_out, W_out), -5.0, dtype=np.float32)
    border = 42
    out_loop = rectify(img, map_x, map_y, border_value=border)
    out_vec = rectify_vectorized(img, map_x, map_y, border_value=border)
    assert np.all(out_loop == border)
    assert np.all(out_vec == border)


# ---------------------------------------------------------------------------
# 5. Hand-computed bilinear sanity
# ---------------------------------------------------------------------------

def test_hand_computed():
    # 2×2 source; sample at (0.5, 0.5) → average of all four corners = 25
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    map_x = np.array([[0.5]], dtype=np.float32)
    map_y = np.array([[0.5]], dtype=np.float32)
    val_loop = rectify(img, map_x, map_y)
    val_vec = rectify_vectorized(img, map_x, map_y)
    assert abs(float(val_loop[0, 0]) - 25.0) < 1.0   # allow ±1 for uint8 rounding
    assert abs(float(val_vec[0, 0]) - 25.0) < 1.0


# ---------------------------------------------------------------------------
# 6. Optional OpenCV comparison
# ---------------------------------------------------------------------------

def test_against_opencv_if_available():
    try:
        import cv2
    except ImportError:
        pytest.skip("cv2 not installed — skipping OpenCV comparison")

    img = _rand_img(64, 96, c=3, seed=6)
    H_out, W_out = 50, 70
    rng = np.random.default_rng(7)
    map_x = rng.uniform(0, 95, (H_out, W_out)).astype(np.float32)
    map_y = rng.uniform(0, 63, (H_out, W_out)).astype(np.float32)

    ours = rectify_vectorized(img, map_x, map_y).astype(np.int16)
    ref = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(np.int16)

    diff = np.abs(ours - ref)
    # Allow up to 1 LSB for rounding differences between implementations
    assert diff.max() <= 1, f"max diff vs cv2.remap = {diff.max()}"
    assert (diff == 0).mean() > 0.95, f"only {(diff == 0).mean():.2%} exact matches"
