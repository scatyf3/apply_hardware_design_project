#include "resize.h"

#include <hls_math.h>

// Baseline streaming resize with bilinear interpolation.
//
// Coordinate model (matches sw/resize.py):
//   u = scale_x * x   (no half-pixel centering)
//   v = scale_y * y
// then sample img_in at (u, v) with bilinear interpolation. Coordinates are
// clamped to [0, W-1] × [0, H-1] so out-of-range lookups replicate borders.
//
// Schedule: we keep a 2-row ring buffer of input rows. Because v is monotonic
// non-decreasing in the output row index y, any output row y only needs input
// rows v0(y) and v0(y)+1. After reading input row r, we emit every output row
// y whose v0(y)+1 == r (or for upscale, every y whose pair is [r-1, r] that
// we haven't emitted yet). This makes each input row readable exactly once
// from the AXI-Stream and each output pixel writable exactly once — the fast
// path the plan calls for.

static pixel_t bilinear(
    pixel_t p00, pixel_t p10, pixel_t p01, pixel_t p11,
    float du, float dv)
{
#pragma HLS INLINE
    float one_du = 1.0f - du;
    float one_dv = 1.0f - dv;
    float w00 = one_du * one_dv;
    float w10 = du     * one_dv;
    float w01 = one_du * dv;
    float w11 = du     * dv;

    float acc = w00 * (float)p00
              + w10 * (float)p10
              + w01 * (float)p01
              + w11 * (float)p11;

    if (acc < 0.0f)   acc = 0.0f;
    if (acc > 255.0f) acc = 255.0f;
    return (pixel_t)(ap_uint<8>)(int)acc;  // truncate-to-uint8 like Python's astype
}

void resize_kernel(
    hls::stream<pixel_t> &in_stream,
    hls::stream<pixel_t> &out_stream,
    dim_t in_w,
    dim_t in_h,
    dim_t out_w,
    dim_t out_h,
    float scale_x,
    float scale_y)
{
#pragma HLS INTERFACE axis port=in_stream
#pragma HLS INTERFACE axis port=out_stream
#pragma HLS INTERFACE s_axilite port=in_w     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=in_h     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=out_w    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=out_h    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=scale_x  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=scale_y  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return   bundle=ctrl

    // 2-row ring buffer, partitioned so both rows can be read in the same
    // cycle to evaluate a 2x2 neighborhood at II=1.
    static pixel_t line_buf[2][RESIZE_MAX_IN_W];
#pragma HLS ARRAY_PARTITION variable=line_buf dim=1 complete
#pragma HLS BIND_STORAGE variable=line_buf type=ram_t2p impl=bram

    dim_t next_y = 0;

    INPUT_ROW: for (dim_t r = 0; r < in_h; r++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=RESIZE_MAX_IN_H

        // Load input row r into the ring slot r&1.
        LOAD_ROW: for (dim_t xi = 0; xi < in_w; xi++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=RESIZE_MAX_IN_W
#pragma HLS PIPELINE II=1
            line_buf[r & 1][xi] = in_stream.read();
        }

        // Emit every output row whose (v0, v1) pair is now in the buffer.
        // After loading row r, rows (r-1, r) and (r, r) (bottom-edge clamp)
        // are available.
        EMIT: while (next_y < out_h) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=RESIZE_MAX_OUT_H

            float v_f = scale_y * (float)(int)next_y;
            // clamp v_f to [0, in_h - 1]
            float v_max = (float)((int)in_h - 1);
            if (v_f < 0.0f)    v_f = 0.0f;
            if (v_f > v_max)   v_f = v_max;

            int v0i = (int)v_f;  // floor for non-negative
            int v1i = v0i + 1;
            if (v1i > (int)in_h - 1) v1i = (int)in_h - 1;

            // Need rows v0i and v1i both <= r before emitting.
            if (v1i > (int)r) break;

            float dv = v_f - (float)v0i;
            ap_uint<1> row0_slot = v0i & 1;
            ap_uint<1> row1_slot = v1i & 1;

            EMIT_PIXEL: for (dim_t xo = 0; xo < out_w; xo++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=RESIZE_MAX_OUT_W
#pragma HLS PIPELINE II=1
                float u_f = scale_x * (float)(int)xo;
                float u_max = (float)((int)in_w - 1);
                if (u_f < 0.0f)  u_f = 0.0f;
                if (u_f > u_max) u_f = u_max;

                int u0i = (int)u_f;
                int u1i = u0i + 1;
                if (u1i > (int)in_w - 1) u1i = (int)in_w - 1;

                float du = u_f - (float)u0i;

                pixel_t p00 = line_buf[row0_slot][u0i];
                pixel_t p10 = line_buf[row0_slot][u1i];
                pixel_t p01 = line_buf[row1_slot][u0i];
                pixel_t p11 = line_buf[row1_slot][u1i];

                out_stream.write(bilinear(p00, p10, p01, p11, du, dv));
            }

            next_y++;
        }
    }
}
