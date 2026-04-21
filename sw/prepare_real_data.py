"""Prepare real-image test data for the HLS resize testbench.

Loads NYU PNGs from archive/data/nyu_real, converts to 8-bit grayscale, runs
the Python golden (sw/resize.py) at several output sizes, and writes paired
raw-uint8 .bin files that hls/resize_tb.cpp can consume with plain fread().

Also dumps side-by-side PNGs under sw/vis/ for qualitative inspection.

Run:
    /home/yf3005/micromamba/envs/segenv/bin/python sw/prepare_real_data.py

All paths are resolved relative to the script location, so it works regardless
of the caller's cwd.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from resize import resize_vectorized

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
NYU_DIR = PROJECT_DIR / "archive" / "data" / "nyu_real" / "train" / "images"
TESTDATA_DIR = PROJECT_DIR / "hls" / "testdata"
VIS_DIR = SCRIPT_DIR / "vis"

# (case_name, source_png_index, out_h, out_w)
# Source images are diverse NYU scenes to avoid over-fitting one texture.
CASES = [
    ("nyu_downscale_320x240", 0,   240, 320),  # mild non-integer downscale
    ("nyu_half",              42,  144, 192),  # clean 2x downscale
    ("nyu_upscale_640x480",   137, 480, 640),  # non-integer upscale
    ("nyu_identity",          256, 288, 384),  # smoke test: identity
    ("nyu_square_96",         500, 96,  96),   # aspect-breaking both axes
]


def load_gray(png_path: Path) -> np.ndarray:
    """Load a PNG as H×W uint8 grayscale."""
    img = Image.open(png_path).convert("L")
    return np.array(img, dtype=np.uint8)


def make_pair_png(in_img: np.ndarray, out_img: np.ndarray, path: Path) -> None:
    """Save a side-by-side visualization: input on the left, output on the right.

    Both panels are padded to the taller image's height so PIL can paste them
    side by side without stretching either.
    """
    h = max(in_img.shape[0], out_img.shape[0])
    w = in_img.shape[1] + out_img.shape[1] + 4  # 4 px gutter
    canvas = np.full((h, w), 128, dtype=np.uint8)
    canvas[: in_img.shape[0], : in_img.shape[1]] = in_img
    x_off = in_img.shape[1] + 4
    canvas[: out_img.shape[0], x_off : x_off + out_img.shape[1]] = out_img
    Image.fromarray(canvas, mode="L").save(path)


def main() -> None:
    if not NYU_DIR.is_dir():
        raise SystemExit(f"source dataset not found: {NYU_DIR}")

    TESTDATA_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    png_files = sorted(NYU_DIR.glob("*.png"))
    if not png_files:
        raise SystemExit(f"no PNGs under {NYU_DIR}")

    manifest_lines = []
    for name, src_idx, out_h, out_w in CASES:
        if src_idx >= len(png_files):
            raise SystemExit(f"case {name}: source index {src_idx} out of range "
                             f"({len(png_files)} PNGs available)")

        src_png = png_files[src_idx]
        in_img = load_gray(src_png)
        in_h, in_w = in_img.shape

        out_img = resize_vectorized(in_img, out_h, out_w)
        assert out_img.shape == (out_h, out_w), out_img.shape
        assert out_img.dtype == np.uint8

        in_bin = TESTDATA_DIR / f"{name}_in.bin"
        gold_bin = TESTDATA_DIR / f"{name}_gold.bin"
        in_img.tofile(in_bin)
        out_img.tofile(gold_bin)

        pair_png = VIS_DIR / f"{name}_pair.png"
        make_pair_png(in_img, out_img, pair_png)

        manifest_lines.append(f"{name} {in_h} {in_w} {out_h} {out_w}")
        print(f"  {name:<24s}  {src_png.name}  {in_h}x{in_w} -> {out_h}x{out_w}  "
              f"-> {in_bin.name}, {gold_bin.name}, {pair_png.name}")

    manifest = TESTDATA_DIR / "cases.txt"
    manifest.write_text("\n".join(manifest_lines) + "\n")
    print(f"\nWrote {len(manifest_lines)} case(s) to {manifest}")
    print(f"Visualizations under {VIS_DIR}")


if __name__ == "__main__":
    main()
