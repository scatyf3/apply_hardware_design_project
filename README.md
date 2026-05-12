# Real-Time ROS 2 Camera Preprocessing Accelerator

Two Vitis HLS image-processing kernels for the AMD ROS 2 Perception Node pipeline: bilinear `resize_kernel` and map-based `rectify_kernel`. Architecture and motivation are in [plan.md](apply_hardware_design_project/plan.md); this README is the build / test / verification reference for both stages.

## verification at a glance

All results from Vitis HLS 2023.2, target part `xczu7ev-ffvc1156-2-e` (Zynq UltraScale+ ZCU104-class) at a 5 ns clock target.

| stage | python tests | hls csim | csynth | hls cosim | resources (BRAM / DSP / FF / LUT) |
|---|---|---|---|---|---|
| `resize_kernel`  | 5/5 PASS | 5/5 PASS | done; slack `−0.00 ns` (marginal) | 10/10 PASS (5 synthetic + 5 NYU real-data) | 2 / 37 / 5,206 / 6,406 |
| `rectify_kernel` | 4/4 PASS | 5/5 PASS | done; slack `0.00 ns` (clean)    | 5/5 PASS                                   | 12 / 13 / 7,542 / 8,903 |

Per-case verification tables, schedule analysis, and resource percentages of the target part are in the respective sections below. Raw HLS reports live under `hls/{resize,rectify}_hls/sol1/syn/report/` and `…/sim/report/`; the build trees are intentionally excluded from version control via `.gitignore` (`hls/*_hls/`) — the tables in this README are the canonical summary.

The resize top-module slack of `−0.00 ns` is HLS flagging a sub-rounding timing violation on the `EMIT_PIXEL` pipeline at the 5 ns target. The kernel synthesizes and co-simulates correctly; closing timing in Vivado place-and-route may require a slightly relaxed clock or LUT-area trade. The rectify kernel meets the same 5 ns target with no warning.

## python environment

The Python side (golden model + real-data fixture generation) only needs `numpy` and `Pillow`, pinned in [apply_hardware_design_project/requirements.txt](apply_hardware_design_project/requirements.txt). Any Python ≥ 3.9 works; tested on 3.10.

**Option A — venv + pip** (no conda needed):

```
python3 -m venv .venv
source .venv/bin/activate            # bash/zsh
# source .venv/bin/activate.csh      # tcsh
pip install -r apply_hardware_design_project/requirements.txt
```

**Option B — conda / micromamba**:

```
micromamba create -n resize python=3.10 numpy pillow
micromamba activate resize
```

After activation, all `python …` commands in this README use that interpreter.

OpenCV is optional — [sw/test_resize.py](apply_hardware_design_project/sw/test_resize.py) skips the cv2 cross-check when it's not importable. Enable it by uncommenting the `opencv-python` line in `requirements.txt` (or `pip install opencv-python`).

## dataset

Real-image tests reuse the NYU indoor-scene PNGs already on disk under [apply_hardware_design_project/archive/data/nyu_real/](apply_hardware_design_project/archive/data/nyu_real/) — no download needed:

- `train/images/` — 795 PNGs, **384×288 RGB** (primary source for the HLS resize testbench)
- `test/images/` — 654 PNGs, 384×288 RGB
- `train/masks_3class/`, `test/masks_3class/` — segmentation masks (unused by the resize work)

For the resize stage we convert RGB → 8-bit grayscale with `PIL.Image.convert("L")` and feed the raw uint8 bytes into the HLS C-testbench via headerless `.bin` files. Five fixed cases are generated and checked bit-exactly against the Python golden:

| case | input | output | coverage |
|---|---|---|---|
| `nyu_downscale_320x240` | 384×288 | 320×240 | non-integer down |
| `nyu_half`              | 384×288 | 192×144 | clean 2× down |
| `nyu_upscale_640x480`   | 384×288 | 640×480 | non-integer up |
| `nyu_identity`          | 384×288 | 384×288 | no-op smoke test |
| `nyu_square_96`         | 384×288 | 96×96   | both axes stretched |

Paired side-by-side visualizations land under [apply_hardware_design_project/sw/vis/](apply_hardware_design_project/sw/vis/) for eyeball checks (border artifacts, obvious aliasing).


## python golden model

[apply_hardware_design_project/sw/resize.py](apply_hardware_design_project/sw/resize.py) is the reference resize with bilinear interpolation. It follows the `u = sₓ·x, v = sᵧ·y` formulation in [apply_hardware_design_project/plan.md](apply_hardware_design_project/plan.md) (no half-pixel centering), and every arithmetic step runs in **float32** so the model is bit-exact against the HLS kernel and its C-sim testbench.

Three entry points:

- `bilinear_sample(img, u, v)` — single-sample helper, border-clamped to `[0, W−1] × [0, H−1]`.
- `resize(img, out_h, out_w)` — plain double loop; maps 1:1 to the HLS schedule and is the easiest path for line-by-line comparison against an HLS C-sim trace.
- `resize_vectorized(img, out_h, out_w)` — same math, numpy-vectorized; used to generate real-data goldens.

Run the unit tests (loop vs vectorized bit-exact, identity, shapes, hand-computed 2×2, OpenCV if installed):

```
python apply_hardware_design_project/sw/test_resize.py
```

Generate the real-image test fixtures for the HLS testbench (writes `apply_hardware_design_project/hls/testdata/*.bin` + `cases.txt` and the visualization PNGs under `apply_hardware_design_project/sw/vis/`):

```
python apply_hardware_design_project/sw/prepare_real_data.py
```


## hls hardware implementation

Vitis HLS C++ kernel for the resize stage lives under [apply_hardware_design_project/hls/](apply_hardware_design_project/hls/). It implements the same `u = sₓ·x, v = sᵧ·y` math as the Python golden in float32, so the C-testbench can `memcmp` the outputs bit-exactly — no tolerance windows.

### files

- [hls/resize.h](apply_hardware_design_project/hls/resize.h) — top-level declarations, `pixel_t = ap_uint<8>`, `dim_t = ap_uint<13>` (≤ 8191), compile-time bounds `RESIZE_MAX_IN_W/H`, `RESIZE_MAX_OUT_W/H` (default 1920×1080).
- [hls/resize.cpp](apply_hardware_design_project/hls/resize.cpp) — `resize_kernel(...)`, the synthesizable top.
- [hls/resize_tb.cpp](apply_hardware_design_project/hls/resize_tb.cpp) — self-contained C-testbench: 5 synthetic cases + every case listed in `testdata/cases.txt`.
- [hls/run_hls.tcl](apply_hardware_design_project/hls/run_hls.tcl) — Vitis HLS build/flow script (csim / csynth / cosim).
- [hls/testdata/](apply_hardware_design_project/hls/testdata/) — real-image `.bin` fixtures produced by `sw/prepare_real_data.py`.

### interface

```
void resize_kernel(
    hls::stream<pixel_t> &in_stream,   // AXI-Stream, row-major uint8 pixels
    hls::stream<pixel_t> &out_stream,  // AXI-Stream, row-major uint8 pixels
    dim_t in_w, dim_t in_h,            // AXI4-Lite (bundle=ctrl)
    dim_t out_w, dim_t out_h,
    float scale_x, float scale_y);     // pre-computed by PS: in_*/out_*
```

- **Data plane**: two `axis` ports — one input stream, one output stream, one 8-bit pixel per beat. Fits directly between a DMA source and a DMA sink in a Zynq streaming pipeline.
- **Control plane**: all dims and scale factors are `s_axilite` in a single `ctrl` bundle, plus the standard `return` register (ap_start / ap_done / ap_idle / ap_ready). `scale_x/y` are passed as pre-divided `float` so the kernel does **not** synthesize a divider — the PS computes `in_w/out_w` once per frame config.

### coordinate & math (matches the golden exactly)

For every output pixel `(x, y)`:

```
u = clamp(scale_x * x, 0, in_w − 1)
v = clamp(scale_y * y, 0, in_h − 1)
u0 = (int)u,  u1 = min(u0+1, in_w−1)
v0 = (int)v,  v1 = min(v0+1, in_h−1)
du = u − u0,  dv = v − v0
out = clamp(Σ w_ij · p_ij, 0, 255) truncated to uint8
```

All arithmetic is IEEE-754 `float`. The clamp-on-address trick (rather than replicating samples) means the last column/row never triggers out-of-range BRAM reads. Output packing uses `(uint8_t)(int)acc` — integer truncation, same as numpy's `astype(uint8)` after `np.clip` in the Python golden.

### schedule — 2-row ring buffer, II=1

Because `v = scale_y · y` is **monotonic non-decreasing** in `y`, any output row needs only two consecutive input rows: `v0` and `v0+1`. So a `pixel_t line_buf[2][RESIZE_MAX_IN_W]` suffices — no full-frame storage.

```
for r in 0..in_h−1:
    LOAD_ROW: stream in row r into line_buf[r & 1][:]        # II=1
    EMIT: while next_y < out_h and v1(next_y) ≤ r:
        compute row-constant (v_f, v0, v1, dv, row slots)
        EMIT_PIXEL: for xo in 0..out_w−1:                    # II=1
            compute (u_f, u0, u1, du)
            fetch 4 taps from line_buf[row0_slot/row1_slot]
            write bilinear(...) to out_stream
        next_y++
```

Key pragmas ([resize.cpp:65-77](apply_hardware_design_project/hls/resize.cpp#L65-L77), [resize.cpp:104-106](apply_hardware_design_project/hls/resize.cpp#L104-L106)):

- `ARRAY_PARTITION line_buf dim=1 complete` — the two-row dimension is fully split so both tap rows are read in the **same cycle** → 2×2 neighborhood at II=1.
- `BIND_STORAGE line_buf type=ram_t2p impl=bram` — true dual-port BRAM: write (from LOAD_ROW) and read (from EMIT_PIXEL) can happen concurrently; on a single BRAM this would cost 1 extra cycle per pixel.
- `PIPELINE II=1` on both `LOAD_ROW` and `EMIT_PIXEL` → steady-state throughput is **1 pixel/cycle** on both the input load and output emit phases.
- `LOOP_TRIPCOUNT` on every bounded-dynamic loop so Vitis HLS reports meaningful latency with the `RESIZE_MAX_*` bounds.

Fast-path correctness argument: after streaming in row `r`, any output row `y` with `v1(y) ≤ r` is serviceable (both taps in `line_buf`), and because `v` is monotonic in `y` we can drain them greedily in order without ever needing an old row again. Each input row is therefore read **once** from the AXI-Stream, and each output pixel written **once** — no backpressure loops, no slow path.

### testbench

[hls/resize_tb.cpp](apply_hardware_design_project/hls/resize_tb.cpp) runs **two tiers** and fails on any byte mismatch:

1. **Synthetic cases** ([resize_tb.cpp:235-239](apply_hardware_design_project/hls/resize_tb.cpp#L235-L239)) — deterministic `rand()` images for 5 shape regimes (downscale / upscale / identity / non-integer / wide-skinny). The golden is the in-file `golden_resize()` — a straight C++ port of `sw/resize.py` with the same float32 order of ops.
2. **Real-data cases** — `find_testdata_dir()` walks a few candidate paths so the tb works both from `hls/` directly and from Vitis's deep `csim/build` cwd. Then it reads `testdata/cases.txt`, streams each `_in.bin` through `resize_kernel`, and `memcmp`-s against `_gold.bin` (produced by the Python `resize_vectorized`). If `cases.txt` is missing the real-data phase is skipped with a warning — the synthetic cases still run.

Every case prints a single `PASS (…)` or `FAIL (…): N mismatches, max|diff|=…, first (y,x) hw=… gold=…` line, and the driver exits non-zero if any case fails.

### build / run

From [apply_hardware_design_project/hls/](apply_hardware_design_project/hls/):

```
vitis_hls -f run_hls.tcl            # csim + csynth + cosim
vitis_hls -f run_hls.tcl csim       # C simulation only
vitis_hls -f run_hls.tcl csynth     # C synthesis only
vitis_hls -f run_hls.tcl cosim      # RTL/C co-simulation (needs csynth first)
```

Target in the script is `xczu7ev-ffvc1156-2-e` (Zynq UltraScale+ ZCU104-class) at a **5 ns** clock — adjust `PART` / `PERIOD` in [run_hls.tcl](apply_hardware_design_project/hls/run_hls.tcl) for a different board. The project tree is written to `hls/resize_hls/sol1/`; synthesis and cosim reports land under `sol1/syn/report/` and `sol1/sim/report/`.

Before running cosim with real images, regenerate the fixtures once:

```
python apply_hardware_design_project/sw/prepare_real_data.py
```

### verification & synthesis (latest run)

| phase | result |
|---|---|
| Python golden tests ([sw/test_resize.py](apply_hardware_design_project/sw/test_resize.py)) | 5/5 PASS (loop ≡ vectorized bit-exact, identity, shapes, hand-computed 2×2, OpenCV cross-check skipped when `cv2` absent) |
| HLS C-simulation (synthetic) | `downscale` / `upscale` / `identity` / `non-integer` / `wide` — 5/5 PASS |
| HLS C-simulation (NYU real-data) | `nyu_downscale_320x240` / `nyu_half` / `nyu_upscale_640x480` / `nyu_identity` / `nyu_square_96` — 5/5 PASS, byte-exact against `_gold.bin` from `resize_vectorized` |
| HLS C-synthesis | completes; top-module slack reported as `−0.00 ns` at 5 ns target (marginal, see below) |
| HLS C/RTL co-simulation | 10/10 PASS (both tiers); max `hls::stream` depth = 307,200 (= 640×480, largest real-data case) |

Resource estimates from `hls/resize_hls/sol1/syn/report/csynth.rpt` (target `xczu7ev-ffvc1156-2-e`):

| resource | usage | % of part |
|---|---|---|
| BRAM | 2 | ~0% |
| DSP  | 37 | 2% |
| FF   | 5,206 | 1% |
| LUT  | 6,406 | 2% |

Schedule numbers from the same report: `LOAD_ROW` inner loop achieves **II = 1** (1922 cycles for 1920 input pixels); `EMIT_PIXEL` inner loop achieves **II = 1** with 79-cycle iteration latency (1997 cycles for 1920 output pixels). The conservative top-module latency in `csynth.rpt` (≈ 2.38 G cycles) is HLS's static worst-case bound for a full 1920×1080 pass — it assumes the outer `EMIT` re-traverses all output rows every input row, which the algorithm does not actually do; cosim confirms the realised throughput tracks `in_h × in_w + out_h × out_w` pixels at II=1.

## rectify

The rectify stage runs after resize and remaps every output pixel `(x, y)` from a source coordinate `(u, v) = (map_x[y, x], map_y[y, x])`. It reuses the resize stage's bilinear sampler, BORDER_REPLICATE clamp, and uint8 truncation, so the Python golden and the HLS kernel share one arithmetic convention end-to-end and the testbench can `memcmp` outputs with zero tolerance.

### python golden model

[apply_hardware_design_project/sw/rectify.py](apply_hardware_design_project/sw/rectify.py) reuses `bilinear_sample()` from `resize.py`, so border-clamp and float32 accumulation are inherited unchanged. Three entry points:

- `rectify(img_in, map_x, map_y)` — plain double loop over output pixels; maps 1:1 to the HLS schedule. Returns `img_in.dtype` after `np.clip(0, 255).astype(...)` (truncation, not rounding — matches `(uint8_t)(int)acc` in the kernel).
- `identity_maps(h, w)` — returns `(map_x, map_y)` with `map_x[y, x] = x` and `map_y[y, x] = y`. The rectified output equals the input bit-for-bit.
- `shifted_maps(h, w, dx, dy)` — identity map plus a constant `(dx, dy)` shift. Drives the fractional-pixel tests.

Run the unit tests (identity copy / fractional shape & dtype / border-replicate clamp / hand-computed half-pixel average = 25):

```
python apply_hardware_design_project/sw/test_rectify.py
```

Regenerate the binary fixtures under `hls/testdata/rectify_*.bin` (an 8×8 `np.arange` ramp sampled with `(dx=0.25, dy=0.5)`):

```
python apply_hardware_design_project/sw/gen_rectify_testdata.py
```

These fixtures are a Python-side cross-check artifact — feeding the committed `rectify_input.bin` + `rectify_map_{x,y}.bin` through `rectify()` reproduces `rectify_expected.bin` byte-for-byte. They are **not** consumed by the HLS C-testbench today; the testbench is self-contained (see below).

### hls hardware implementation

Vitis HLS C++ kernel for the rectify stage lives in [hls/rectify.cpp](apply_hardware_design_project/hls/rectify.cpp). Same float32 math, same uint8 truncation, same BORDER_REPLICATE clamp as the Python golden — the in-file `golden_sample()` in the testbench is a line-by-line port.

#### files

- [hls/rectify.h](apply_hardware_design_project/hls/rectify.h) — top-level declarations, `pixel_t = ap_uint<8>`, `dim_t = ap_uint<13>`, compile-time bounds `RECTIFY_MAX_IN_W/H`, `RECTIFY_MAX_OUT_W/H` (default 1920×1080).
- [hls/rectify.cpp](apply_hardware_design_project/hls/rectify.cpp) — `rectify_kernel(...)`, the synthesizable top.
- [hls/rectify_tb.cpp](apply_hardware_design_project/hls/rectify_tb.cpp) — self-contained C-testbench: 5 deterministic cases, no external fixtures.
- [hls/run_rectify_hls.tcl](apply_hardware_design_project/hls/run_rectify_hls.tcl) — Vitis HLS build/flow script (csim / csynth / cosim).

#### interface

```
void rectify_kernel(
    const pixel_t *img_in,             // m_axi gmem_img
    const float   *map_x,              // m_axi gmem_mapx
    const float   *map_y,              // m_axi gmem_mapy
    pixel_t       *img_out,            // m_axi gmem_out
    dim_t in_w, dim_t in_h,            // s_axilite (bundle=ctrl)
    dim_t out_w, dim_t out_h);
```

- **Data plane**: four independent `m_axi` master ports — input image, two `float` remap tables, output image. Each table is sized to `out_w * out_h` and addressed row-major (`y*out_w + x`). Splitting `map_x` / `map_y` / `img_in` / `img_out` onto separate `gmem_*` bundles lets HLS issue their reads in parallel rather than serialize them on one shared port.
- **Control plane**: all dims plus 64-bit base pointers exposed through one `s_axilite` `ctrl` bundle alongside the standard `return` register (`ap_start` / `ap_done` / `ap_idle` / `ap_ready` + `interrupt`).
- Unlike `resize_kernel`, this is a baseline memory-mapped design: no on-chip line buffer, every input tap goes back to DDR through `gmem_img`. The optimized line-buffered + output-queue path with DDR slow-path fallback is described in [plan.md](apply_hardware_design_project/plan.md) and is **not** implemented yet.

#### coordinate & math (matches the golden exactly)

For every output pixel `(x, y)`:

```
u = clamp(map_x[y*out_w + x], 0, in_w − 1)
v = clamp(map_y[y*out_w + x], 0, in_h − 1)
u0 = (int)u,  u1 = min(u0+1, in_w−1)
v0 = (int)v,  v1 = min(v0+1, in_h−1)
du = u − u0,  dv = v − v0
out = clamp(Σ w_ij · p_ij, 0, 255) truncated to uint8
```

The clamp-on-address trick handles arbitrary map values (BORDER_REPLICATE) without ever issuing an out-of-range pointer read. Float accumulation order matches the Python golden, so `memcmp` against the testbench gold succeeds bit-exactly with no tolerance window.

#### schedule — m_axi-per-pixel baseline, II=4

The kernel is two nested loops with `PIPELINE II=1` requested on the inner body ([rectify.cpp:84](apply_hardware_design_project/hls/rectify.cpp#L84)):

```
OUT_Y:  for y in 0..out_h−1:
OUT_X:    for x in 0..out_w−1:                       # PIPELINE II=1 requested
            idx = y*out_w + x
            u   = map_x[idx]                         # m_axi gmem_mapx
            v   = map_y[idx]                         # m_axi gmem_mapy
            img_out[idx] = bilinear_sample_mem(      # 4 reads on m_axi gmem_img
                img_in, in_w, in_h, u, v)            # + write on m_axi gmem_out
```

Each iteration issues two `float` map reads, four 8-bit image taps, one 8-bit output write, plus the bilinear arithmetic. HLS settles on **II = 4 cycles/pixel** in `csynth.rpt` — the inner loop is bottlenecked by memory dependencies, not by the float chain. No line buffer is instantiated.

For a full 1920×1080 frame the report measures **8,294,489 cycles ≈ 41 ms @ 200 MHz**. That fits a ~24 fps budget but not 60 fps. The route to II=1 is the line-buffered + output-queue variant described in `plan.md` (deferred).

#### testbench

[hls/rectify_tb.cpp](apply_hardware_design_project/hls/rectify_tb.cpp) drives 5 deterministic cases and fails on any byte mismatch:

| case | in → out | map style | coverage |
|---|---|---|---|
| `identity`      | 32×24 → 32×24 | `(x, y)` | sanity: rectify ≡ copy |
| `fractional`    | 32×24 → 32×24 | `(x+0.25, y+0.50)` | bilinear interpolation path |
| `crop`          | 48×36 → 24×18 | `(x+3, y+2)` | sub-window with integer offset |
| `border_clamp`  | 32×24 → 32×24 | rotating `−5` / `in_w+8` / `in_h+9` / both-OOB | replicate-clamp stress |
| `scale_map`     | 23×17 → 29×11 | half-pixel-centered scale | non-integer in & out dims |

Failures print `errors=N first=I hw=… gold=… max_abs=…` and the driver returns non-zero. The 8×8 `.bin` fixtures under [hls/testdata/](apply_hardware_design_project/hls/testdata/) are **not** wired into this testbench — they are a Python-side cross-check only.

#### build / run

From [apply_hardware_design_project/hls/](apply_hardware_design_project/hls/):

```
vitis_hls -f run_rectify_hls.tcl            # csim + csynth + cosim
vitis_hls -f run_rectify_hls.tcl csim       # C simulation only
vitis_hls -f run_rectify_hls.tcl csynth     # C synthesis only
vitis_hls -f run_rectify_hls.tcl cosim      # RTL/C co-simulation (needs csynth first)
```

Target part and clock match the resize project: `xczu7ev-ffvc1156-2-e` @ **5 ns**, configurable via `PART` / `PERIOD` in [run_rectify_hls.tcl](apply_hardware_design_project/hls/run_rectify_hls.tcl). The build tree lands under `hls/rectify_hls/sol1/`; synthesis reports under `sol1/syn/report/`, cosim logs under `sol1/sim/report/`.

Latest verified state (xczu7ev @ 5 ns): csim 5/5 PASS, csynth completes with zero timing-slack violations, cosim 5/5 PASS. Resource estimates from `csynth.rpt`:

| resource | usage | % of part |
|---|---|---|
| BRAM | 12 | 1% |
| DSP  | 13 | ~0% |
| FF   | 7,542 | 1% |
| LUT  | 8,903 | 3% |
