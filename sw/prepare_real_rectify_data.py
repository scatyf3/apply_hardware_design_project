"""Prepare real-image test data for the HLS rectify testbench.

Loads NYU PNGs from archive/data/nyu_real, converts to 8-bit grayscale, builds
several rectify maps (identity / fractional shift / crop / center-zoom / barrel
distortion), runs the Python golden (sw/rectify.py:rectify_vectorized) and
writes paired raw-uint8 + raw-float32 .bin files that hls/rectify_tb.cpp can
consume with plain fread().

Also dumps side-by-side input/output PNGs under sw/vis/ for qualitative
inspection of each remap.

Run:
    python sw/prepare_real_rectify_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from rectify import rectify, rectify_vectorized

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
NYU_DIR = PROJECT_DIR / "archive" / "data" / "nyu_real" / "train" / "images"
TESTDATA_DIR = PROJECT_DIR / "hls" / "testdata"
VIS_DIR = SCRIPT_DIR / "vis"

# (case_name, source_png_index, out_h, out_w, map_kind)
# Source images are diverse NYU scenes; map kinds cover the rectify math paths
# we want to exercise on real texture.
CASES = [
    ("nyu_rectify_identity",       0,   288, 384, "identity"),
    ("nyu_rectify_shift_frac",     42,  288, 384, "shift_frac"),
    ("nyu_rectify_crop_192x144",   137, 144, 192, "crop"),
    ("nyu_rectify_zoom_2x",        256, 288, 384, "zoom_2x"),
    ("nyu_rectify_barrel_mild",    500, 288, 384, "barrel"),
]


def build_maps(kind: str, in_h: int, in_w: int, out_h: int, out_w: int):
    """Return (map_x, map_y) as float32 H×W arrays for the named transform."""
    xs = np.tile(np.arange(out_w, dtype=np.float32), (out_h, 1))
    ys = np.tile(np.arange(out_h, dtype=np.float32).reshape(out_h, 1), (1, out_w))

    if kind == "identity":
        return xs.copy(), ys.copy()

    if kind == "shift_frac":
        return xs + np.float32(0.25), ys + np.float32(0.50)

    if kind == "crop":
        # Output covers a sub-window of the input starting at (96, 72).
        return xs + np.float32(96.0), ys + np.float32(72.0)

    if kind == "zoom_2x":
        # Sample a 2x-zoomed view of the input centre; out (cx_out, cy_out)
        # maps to in (cx_in, cy_in), with half-pixel-per-output-pixel steps.
        cx_in = np.float32((in_w - 1) / 2.0)
        cy_in = np.float32((in_h - 1) / 2.0)
        cx_out = np.float32((out_w - 1) / 2.0)
        cy_out = np.float32((out_h - 1) / 2.0)
        map_x = (xs - cx_out) * np.float32(0.5) + cx_in
        map_y = (ys - cy_out) * np.float32(0.5) + cy_in
        return map_x.astype(np.float32), map_y.astype(np.float32)

    if kind == "barrel":
        # Mild forward barrel distortion: pixels are pushed radially outward
        # from the centre by (1 + k*r^2). Corners read past the edges and
        # exercise BORDER_REPLICATE clamp on real texture.
        cx = np.float32((in_w - 1) / 2.0)
        cy = np.float32((in_h - 1) / 2.0)
        norm = np.float32(max(float(cx), float(cy)))
        nx = (xs - cx) / norm
        ny = (ys - cy) / norm
        r2 = nx * nx + ny * ny
        k = np.float32(0.10)
        factor = (np.float32(1.0) + k * r2).astype(np.float32)
        map_x = (cx + factor * (xs - cx)).astype(np.float32)
        map_y = (cy + factor * (ys - cy)).astype(np.float32)
        return map_x, map_y

    raise ValueError(f"unknown map kind: {kind}")


def load_gray(png_path: Path) -> np.ndarray:
    img = Image.open(png_path).convert("L")
    return np.array(img, dtype=np.uint8)


def make_pair_png(in_img: np.ndarray, out_img: np.ndarray, path: Path) -> None:
    """Save input | output side-by-side, padded to the taller image's height."""
    h = max(in_img.shape[0], out_img.shape[0])
    w = in_img.shape[1] + out_img.shape[1] + 4
    canvas = np.full((h, w), 128, dtype=np.uint8)
    canvas[: in_img.shape[0], : in_img.shape[1]] = in_img
    x_off = in_img.shape[1] + 4
    canvas[: out_img.shape[0], x_off : x_off + out_img.shape[1]] = out_img
    Image.fromarray(canvas, mode="L").save(path)


def overlay_grid(img: np.ndarray, spacing: int = 24, value: int = 255) -> np.ndarray:
    """Paint a thin regular grid onto a uint8 grayscale image (non-destructive)."""
    out = img.copy()
    for x in range(0, out.shape[1], spacing):
        out[:, x] = value
    out[:, out.shape[1] - 1] = value
    for y in range(0, out.shape[0], spacing):
        out[y, :] = value
    out[out.shape[0] - 1, :] = value
    return out


def verify_loop_equals_vectorized(in_img, map_x, map_y, out_vec, name):
    """Sanity: rectify() loop and rectify_vectorized() agree byte-for-byte.

    Only checks the first 32×32 patch — full 384×288 in pure Python would take
    minutes, and a corner sample already exercises the same float32 op-order.
    """
    sample_h = min(32, map_x.shape[0])
    sample_w = min(32, map_x.shape[1])
    out_loop = rectify(in_img, map_x[:sample_h, :sample_w], map_y[:sample_h, :sample_w])
    if not np.array_equal(out_loop, out_vec[:sample_h, :sample_w]):
        diff = np.abs(out_loop.astype(int) - out_vec[:sample_h, :sample_w].astype(int))
        raise SystemExit(
            f"case {name}: loop != vectorized on first {sample_h}x{sample_w} patch, "
            f"max|diff|={diff.max()}"
        )


def main() -> None:
    if not NYU_DIR.is_dir():
        raise SystemExit(f"source dataset not found: {NYU_DIR}")

    TESTDATA_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    png_files = sorted(NYU_DIR.glob("*.png"))
    if not png_files:
        raise SystemExit(f"no PNGs under {NYU_DIR}")

    manifest_lines = []
    for name, src_idx, out_h, out_w, kind in CASES:
        if src_idx >= len(png_files):
            raise SystemExit(
                f"case {name}: source index {src_idx} out of range "
                f"({len(png_files)} PNGs available)"
            )

        src_png = png_files[src_idx]
        in_img = load_gray(src_png)
        in_h, in_w = in_img.shape

        map_x, map_y = build_maps(kind, in_h, in_w, out_h, out_w)
        assert map_x.dtype == np.float32 and map_y.dtype == np.float32
        assert map_x.shape == (out_h, out_w), map_x.shape
        assert map_y.shape == (out_h, out_w), map_y.shape

        out_img = rectify_vectorized(in_img, map_x, map_y)
        assert out_img.shape == (out_h, out_w), out_img.shape
        assert out_img.dtype == np.uint8

        verify_loop_equals_vectorized(in_img, map_x, map_y, out_img, name)

        in_bin = TESTDATA_DIR / f"{name}_in.bin"
        mx_bin = TESTDATA_DIR / f"{name}_map_x.bin"
        my_bin = TESTDATA_DIR / f"{name}_map_y.bin"
        gold_bin = TESTDATA_DIR / f"{name}_gold.bin"
        in_img.tofile(in_bin)
        map_x.tofile(mx_bin)
        map_y.tofile(my_bin)
        out_img.tofile(gold_bin)

        pair_png = VIS_DIR / f"{name}_pair.png"
        if kind == "barrel":
            # Overlay a regular grid on the input and re-run the same map so the
            # output's bent grid lines make the radial distortion + edge clamp
            # immediately obvious. .bin fixtures above stay un-gridded.
            in_viz = overlay_grid(in_img, spacing=24, value=255)
            out_viz = rectify_vectorized(in_viz, map_x, map_y)
            make_pair_png(in_viz, out_viz, pair_png)
        else:
            make_pair_png(in_img, out_img, pair_png)

        manifest_lines.append(f"{name} {in_h} {in_w} {out_h} {out_w}")
        print(
            f"  {name:<32s}  {src_png.name}  {in_h}x{in_w} -> {out_h}x{out_w}  "
            f"map={kind:<12s} -> 4 .bin + {pair_png.name}"
        )

    manifest = TESTDATA_DIR / "rectify_cases.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nWrote {len(manifest_lines)} rectify case(s) to {manifest}")
    print(f"Visualizations under {VIS_DIR}")


if __name__ == "__main__":
    main()
