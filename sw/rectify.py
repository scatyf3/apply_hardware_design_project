"""Software golden model for the rectify / remap stage.

For every output pixel (x, y), map_x[y, x] and map_y[y, x] specify the source
coordinate (u, v) in the input image. The output value is sampled with bilinear
interpolation and border-replicate clamping. This matches hls/rectify.cpp.
"""

from __future__ import annotations

import numpy as np

from resize import bilinear_sample


def rectify(img_in: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    if map_x.shape != map_y.shape:
        raise ValueError(f"map_x shape {map_x.shape} != map_y shape {map_y.shape}")
    if map_x.ndim != 2:
        raise ValueError("map_x/map_y must be HxW arrays")

    out_h, out_w = map_x.shape
    out_shape = (out_h, out_w) + img_in.shape[2:]
    out = np.zeros(out_shape, dtype=np.float32)

    for y in range(out_h):
        for x in range(out_w):
            u = np.float32(map_x[y, x])
            v = np.float32(map_y[y, x])
            out[y, x] = bilinear_sample(img_in, u, v)

    return np.clip(out, 0, 255).astype(img_in.dtype)


def rectify_vectorized(img_in: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of rectify() — same math, faster for large maps.

    Mirrors the float32 op-order of resize_vectorized so the result is bit-exact
    against both the loop rectify() above and the HLS rectify_kernel.
    """
    if map_x.shape != map_y.shape:
        raise ValueError(f"map_x shape {map_x.shape} != map_y shape {map_y.shape}")
    if map_x.ndim != 2:
        raise ValueError("map_x/map_y must be HxW arrays")

    in_h, in_w = img_in.shape[:2]

    u = np.clip(map_x.astype(np.float32), np.float32(0.0), np.float32(in_w - 1))
    v = np.clip(map_y.astype(np.float32), np.float32(0.0), np.float32(in_h - 1))

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.minimum(u0 + 1, in_w - 1)
    v1 = np.minimum(v0 + 1, in_h - 1)

    du = u - u0.astype(np.float32)
    dv = v - v0.astype(np.float32)

    img_f = img_in.astype(np.float32)
    p00 = img_f[v0, u0]
    p10 = img_f[v0, u1]
    p01 = img_f[v1, u0]
    p11 = img_f[v1, u1]

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


def identity_maps(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    ys = np.tile(np.arange(h, dtype=np.float32).reshape(h, 1), (1, w))
    return xs, ys


def shifted_maps(h: int, w: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    mx, my = identity_maps(h, w)
    return mx + np.float32(dx), my + np.float32(dy)
