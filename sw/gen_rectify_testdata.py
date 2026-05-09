"""Generate binary test vectors for the HLS rectify testbench."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from rectify import rectify


def write_bin(path: Path, arr: np.ndarray) -> None:
    arr = np.ascontiguousarray(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    print(f"wrote {path} shape={arr.shape} dtype={arr.dtype} bytes={arr.nbytes}")


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out_dir = repo / "hls" / "testdata"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_h, in_w = 8, 8
    out_h, out_w = 8, 8

    img = np.arange(in_h * in_w, dtype=np.uint8).reshape(in_h, in_w)

    xs = np.tile(np.arange(out_w, dtype=np.float32), (out_h, 1))
    ys = np.tile(np.arange(out_h, dtype=np.float32).reshape(out_h, 1), (1, out_w))

    map_x = xs + np.float32(0.25)
    map_y = ys + np.float32(0.50)

    expected = rectify(img, map_x, map_y)

    write_bin(out_dir / "rectify_input.bin", img.astype(np.uint8))
    write_bin(out_dir / "rectify_map_x.bin", map_x.astype(np.float32))
    write_bin(out_dir / "rectify_map_y.bin", map_y.astype(np.float32))
    write_bin(out_dir / "rectify_expected.bin", expected.astype(np.uint8))

    manifest = out_dir / "rectify_case.txt"
    manifest.write_text(
        f"in_h {in_h}\n"
        f"in_w {in_w}\n"
        f"out_h {out_h}\n"
        f"out_w {out_w}\n"
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
