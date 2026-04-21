"""Software golden model for the resize stage.

Maps each output pixel (x, y) to a source coordinate (u, v) in the input image
using u = s_x * x, v = s_y * y, then performs bilinear interpolation from the
four nearest source pixels. Matches the formulation in plan.md.
"""

from __future__ import annotations

import numpy as np


def bilinear_sample(img: np.ndarray, u: float, v: float) -> np.ndarray:
    H, W = img.shape[:2]

    u = np.clip(u, 0.0, W - 1.0)
    v = np.clip(v, 0.0, H - 1.0)

    u0 = int(np.floor(u))
    v0 = int(np.floor(v))
    u1 = min(u0 + 1, W - 1)
    v1 = min(v0 + 1, H - 1)

    du = np.float32(u - u0)
    dv = np.float32(v - v0)

    w00 = (np.float32(1.0) - du) * (np.float32(1.0) - dv)
    w10 = du * (np.float32(1.0) - dv)
    w01 = (np.float32(1.0) - du) * dv
    w11 = du * dv

    p00 = img[v0, u0].astype(np.float32)
    p10 = img[v0, u1].astype(np.float32)
    p01 = img[v1, u0].astype(np.float32)
    p11 = img[v1, u1].astype(np.float32)

    return w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11


def resize(img_in: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize img_in to (out_h, out_w) with bilinear interpolation.

    Follows the u = s_x * x, v = s_y * y formulation from plan.md (no
    half-pixel centering) so the hardware coordinate generator can use the
    same arithmetic.
    """
    in_h, in_w = img_in.shape[:2]
    scale_x = np.float32(in_w / out_w)
    scale_y = np.float32(in_h / out_h)

    out_shape = (out_h, out_w) + img_in.shape[2:]
    out = np.zeros(out_shape, dtype=np.float32)

    for y in range(out_h):
        v = scale_y * np.float32(y)
        for x in range(out_w):
            u = scale_x * np.float32(x)
            out[y, x] = bilinear_sample(img_in, u, v)

    return np.clip(out, 0, 255).astype(img_in.dtype)


def resize_vectorized(img_in: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Vectorized equivalent of resize() — same math, faster for large images."""
    in_h, in_w = img_in.shape[:2]
    scale_x = np.float32(in_w / out_w)
    scale_y = np.float32(in_h / out_h)

    xs = np.clip(np.arange(out_w, dtype=np.float32) * scale_x, 0.0, np.float32(in_w - 1))
    ys = np.clip(np.arange(out_h, dtype=np.float32) * scale_y, 0.0, np.float32(in_h - 1))

    u0 = np.floor(xs).astype(np.int32)
    v0 = np.floor(ys).astype(np.int32)
    u1 = np.minimum(u0 + 1, in_w - 1)
    v1 = np.minimum(v0 + 1, in_h - 1)

    du = (xs - u0.astype(np.float32)).reshape(1, -1)
    dv = (ys - v0.astype(np.float32)).reshape(-1, 1)

    img_f = img_in.astype(np.float32)
    p00 = img_f[np.ix_(v0, u0)]
    p10 = img_f[np.ix_(v0, u1)]
    p01 = img_f[np.ix_(v1, u0)]
    p11 = img_f[np.ix_(v1, u1)]

    if img_in.ndim == 3:
        du = du[..., None]
        dv = dv[..., None]

    one = np.float32(1.0)
    w00 = (one - du) * (one - dv)
    w10 = du * (one - dv)
    w01 = (one - du) * dv
    w11 = du * dv

    out = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11
    return np.clip(out, 0, 255).astype(img_in.dtype)
