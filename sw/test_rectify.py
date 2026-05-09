from __future__ import annotations

import numpy as np

from rectify import identity_maps, rectify, shifted_maps


def _rand_img(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w), dtype=np.uint8)


def test_identity_maps_copy_input():
    img = _rand_img(24, 32, seed=1)
    mx, my = identity_maps(24, 32)
    out = rectify(img, mx, my)
    assert np.array_equal(out, img)


def test_fractional_shift_shape_and_dtype():
    img = _rand_img(24, 32, seed=2)
    mx, my = shifted_maps(24, 32, dx=0.35, dy=0.65)
    out = rectify(img, mx, my)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_border_clamp_is_replicate():
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    mx = np.array([[-100.0, 100.0], [-100.0, 100.0]], dtype=np.float32)
    my = np.array([[-100.0, -100.0], [100.0, 100.0]], dtype=np.float32)
    out = rectify(img, mx, my)
    expected = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    assert np.array_equal(out, expected)


def test_known_half_pixel_average():
    img = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    mx = np.array([[0.5]], dtype=np.float32)
    my = np.array([[0.5]], dtype=np.float32)
    out = rectify(img, mx, my)
    assert int(out[0, 0]) == 25


if __name__ == "__main__":
    tests = [
        test_identity_maps_copy_input,
        test_fractional_shift_shape_and_dtype,
        test_border_clamp_is_replicate,
        test_known_half_pixel_average,
    ]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print("\nAll rectify Python tests passed.")

