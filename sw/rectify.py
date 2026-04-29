"""Software golden model for the rectify (remap) stage.

Applies geometric remapping using two pre-computed lookup tables map_x and
map_y that specify, for each output pixel (x, y), the fractional source
coordinate (u, v) to sample from the resized image:

    u = map_x[y, x],   v = map_y[y, x]
    I_out[y, x] = bilinear_sample(I_resize, u, v)

map_x / map_y are typically generated offline from camera calibration
parameters (e.g. via cv2.initUndistortRectifyMap) and reused at runtime.

Matches the formulation in plan.md so the hardware coordinate lookup can use
the same arithmetic.
"""

import numpy as np

from resize import bilinear_sample


def rectify(
    img_resize,  # type: np.ndarray
    map_x,       # type: np.ndarray
    map_y,       # type: np.ndarray
    border_value=0,
):
    """Remap img_resize using pre-computed fractional coordinate maps.

    Parameters
    ----------
    img_resize:
        Source image, typically the output of the resize stage.
        Shape (H_resize, W_resize) or (H_resize, W_resize, C).
    map_x:
        Float32 array of shape (H_out, W_out).  Each entry gives the
        fractional *column* (u) to sample from img_resize.
    map_y:
        Float32 array of shape (H_out, W_out).  Each entry gives the
        fractional *row* (v) to sample from img_resize.
    border_value:
        Pixel value used for out-of-bounds source coordinates.

    Returns
    -------
    np.ndarray of the same dtype as img_resize, shape (H_out, W_out[, C]).
    """
    assert map_x.shape == map_y.shape, "map_x and map_y must have the same shape"
    H_out, W_out = map_x.shape
    out_shape = (H_out, W_out) + img_resize.shape[2:]
    out = np.zeros(out_shape, dtype=np.float32)

    in_h, in_w = img_resize.shape[:2]

    for y in range(H_out):
        for x in range(W_out):
            u = float(map_x[y, x])
            v = float(map_y[y, x])
            # Clamp to valid source range; out-of-bounds → border_value
            if u < 0.0 or u > in_w - 1 or v < 0.0 or v > in_h - 1:
                out[y, x] = border_value
            else:
                out[y, x] = bilinear_sample(img_resize, u, v)

    return np.clip(out, 0, 255).astype(img_resize.dtype)


def rectify_vectorized(
    img_resize,  # type: np.ndarray
    map_x,       # type: np.ndarray
    map_y,       # type: np.ndarray
    border_value=0,
):
    """Vectorized equivalent of rectify() — same math, faster for large images."""
    assert map_x.shape == map_y.shape, "map_x and map_y must have the same shape"
    H_out, W_out = map_x.shape
    in_h, in_w = img_resize.shape[:2]

    u = map_x.astype(np.float32)  # (H_out, W_out)
    v = map_y.astype(np.float32)  # (H_out, W_out)

    # Identify out-of-bounds pixels before clamping
    oob = (u < 0.0) | (u > np.float32(in_w - 1)) | \
          (v < 0.0) | (v > np.float32(in_h - 1))

    # Clamp coordinates for safe indexing
    u_c = np.clip(u, 0.0, np.float32(in_w - 1))
    v_c = np.clip(v, 0.0, np.float32(in_h - 1))

    u0 = np.floor(u_c).astype(np.int32)
    v0 = np.floor(v_c).astype(np.int32)
    u1 = np.minimum(u0 + 1, in_w - 1)
    v1 = np.minimum(v0 + 1, in_h - 1)

    du = (u_c - u0.astype(np.float32))  # (H_out, W_out)
    dv = (v_c - v0.astype(np.float32))  # (H_out, W_out)

    img_f = img_resize.astype(np.float32)

    # Gather the four neighbours for every output pixel
    p00 = img_f[v0, u0]
    p10 = img_f[v0, u1]
    p01 = img_f[v1, u0]
    p11 = img_f[v1, u1]

    if img_resize.ndim == 3:
        du = du[..., None]
        dv = dv[..., None]
        oob = oob[..., None]

    one = np.float32(1.0)
    w00 = (one - du) * (one - dv)
    w10 = du * (one - dv)
    w01 = (one - du) * dv
    w11 = du * dv

    out = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11

    # Apply border value to out-of-bounds pixels
    out = np.where(oob, np.float32(border_value), out)

    return np.clip(out, 0, 255).astype(img_resize.dtype)


def make_identity_maps(h, w):
    """Return (map_x, map_y) that reproduce the source image unchanged.

    Useful as a sanity-check baseline: rectify(img, *make_identity_maps(...))
    should return a pixel-exact copy of img.
    """
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    map_x, map_y = np.meshgrid(xs, ys)
    return map_x, map_y


def make_maps_from_calibration(
    K,
    dist_coeffs,
    R,
    P,
    image_size,
):
    """Generate rectification maps from camera calibration parameters.

    Wraps cv2.initUndistortRectifyMap so the rest of the pipeline does not
    depend on OpenCV being installed.  Raises ImportError if cv2 is absent.

    Parameters
    ----------
    K            : 3×3 camera intrinsic matrix.
    dist_coeffs  : distortion coefficients (k1,k2,p1,p2[,k3[,…]]).
    R            : 3×3 rectification rotation matrix (identity if none).
    P            : 3×4 new projection matrix (use K with [0,0,0] column if none).
    image_size   : (width, height) of the *output* rectified image.

    Returns
    -------
    (map_x, map_y) as float32 arrays of shape (height, width).
    """
    import cv2  # optional dependency

    map_x, map_y = cv2.initUndistortRectifyMap(
        K, dist_coeffs, R, P, image_size, cv2.CV_32FC1
    )
    return map_x, map_y
