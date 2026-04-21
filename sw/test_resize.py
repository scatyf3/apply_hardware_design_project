"""Tests for the resize golden model.

Checks:
 1. resize() and resize_vectorized() agree bit-for-bit on the same math.
 2. Identity case: resizing to the same shape returns the source unchanged.
 3. Downscale/upscale shape correctness.
 4. Numerical sanity against a hand-computed bilinear example.
 5. (Optional) close agreement with OpenCV's INTER_LINEAR when cv2 is available.
    Skipped if cv2 isn't installed — the golden model does not depend on it.
"""

from __future__ import annotations

import numpy as np

from resize import bilinear_sample, resize, resize_vectorized


def _rand_img(h: int, w: int, c: int | None = None, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (h, w) if c is None else (h, w, c)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)


def test_loop_vs_vectorized():
    img = _rand_img(17, 23, 3, seed=1)
    a = resize(img, 11, 29)
    b = resize_vectorized(img, 11, 29)
    assert a.shape == b.shape == (11, 29, 3)
    assert np.array_equal(a, b), "loop and vectorized variants disagree"


def test_identity_shape():
    img = _rand_img(32, 48, 3, seed=2)
    out = resize_vectorized(img, 32, 48)
    # u = x, v = y with no fractional part — exact copy.
    assert np.array_equal(out, img)


def test_output_shapes():
    img = _rand_img(40, 60, seed=3)
    assert resize_vectorized(img, 20, 30).shape == (20, 30)
    assert resize_vectorized(img, 80, 120).shape == (80, 120)


def test_hand_computed_bilinear():
    # 2x2 image; sample at (0.5, 0.5) — equal weights on all four corners.
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    val = bilinear_sample(img, 0.5, 0.5)
    assert abs(float(val) - 25.0) < 1e-6


def test_against_opencv_if_available():
    try:
        import cv2
    except ImportError:
        print("cv2 not installed — skipping OpenCV comparison")
        return

    img = _rand_img(64, 96, 3, seed=4)
    out_h, out_w = 40, 50
    ours = resize_vectorized(img, out_h, out_w).astype(np.int16)

    # OpenCV's INTER_LINEAR uses half-pixel centering by default, which does
    # not match u = s_x * x. Use WARP_INVERSE_MAP with an explicit affine so
    # the two share the same coordinate formulation.
    sx = img.shape[1] / out_w
    sy = img.shape[0] / out_h
    M = np.array([[sx, 0, 0], [0, sy, 0]], dtype=np.float32)
    ref = cv2.warpAffine(
        img, M, (out_w, out_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    ).astype(np.int16)

    diff = np.abs(ours - ref)
    # Rounding differs (we clip-to-uint8 vs OpenCV's saturating round); allow
    # up to 1 LSB and require most pixels to match exactly.
    assert diff.max() <= 1, f"max diff vs OpenCV = {diff.max()}"
    assert (diff == 0).mean() > 0.95, f"only {(diff == 0).mean():.2%} exact matches"


if __name__ == "__main__":
    tests = [
        test_loop_vs_vectorized,
        test_identity_shape,
        test_output_shapes,
        test_hand_computed_bilinear,
        test_against_opencv_if_available,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print("\nAll tests passed.")
