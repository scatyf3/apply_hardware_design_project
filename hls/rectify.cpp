#include "rectify.h"

#include <hls_math.h>

// Baseline streaming rectify with bilinear interpolation.
//
// Coordinate model (matches sw/rectify.py):
//   u = map_x[y * out_w + x]   (fractional column into img_resize)
//   v = map_y[y * out_w + x]   (fractional row into img_resize)
//   I_out[y,x] = bilinear(img_resize, u, v)
// Out-of-bounds -> 0 (border-constant, same as the Python golden model).
//
// Baseline memory access pattern:
//   map_x / map_y   — sequential raster order, friendly to AXI burst reads.
//   img_resize      — scatter access (arbitrary u,v per pixel); no burst.
//
// For the optional optimised path (line-buffer + output-queue) see plan.md.
// That extension can replace the img_resize m_axi port with a narrow
// line-buffer BRAM and a stream from resize_kernel, but the AXI-Lite control
// registers and the out_stream interface remain identical.

// -----------------------------------------------------------------------
// Shared bilinear helper — same logic as resize.cpp so both stages use
// identical fixed-point semantics.
// -----------------------------------------------------------------------
static pixel_t bilinear_rect(
    pixel_t p00, pixel_t p10, pixel_t p01, pixel_t p11,
    float du, float dv)
{
#pragma HLS INLINE
    float one_du = 1.0f - du;
    float one_dv = 1.0f - dv;
    float acc = one_du * one_dv * (float)p00
              + du     * one_dv * (float)p10
              + one_du * dv     * (float)p01
              + du     * dv     * (float)p11;
    if (acc < 0.0f)   acc = 0.0f;
    if (acc > 255.0f) acc = 255.0f;
    return (pixel_t)(ap_uint<8>)(int)acc;
}

// -----------------------------------------------------------------------
// rectify_kernel
// -----------------------------------------------------------------------
void rectify_kernel(
    const pixel_t             *img_resize,
    const float               *map_x,
    const float               *map_y,
    hls::stream<pixel_t>      &out_stream,
    dim_t resize_w,
    dim_t resize_h,
    dim_t out_w,
    dim_t out_h)
{
// AXI master ports — three independent bundles so the tool can issue
// concurrent burst requests for the two map arrays while img_resize
// uses a separate bus for scatter reads.
#pragma HLS INTERFACE m_axi port=img_resize offset=slave bundle=gmem0 \
        depth=RECT_MAX_RESIZE_PIXELS max_read_burst_length=16
#pragma HLS INTERFACE m_axi port=map_x      offset=slave bundle=gmem1 \
        depth=RECT_MAX_OUT_PIXELS    max_read_burst_length=256
#pragma HLS INTERFACE m_axi port=map_y      offset=slave bundle=gmem2 \
        depth=RECT_MAX_OUT_PIXELS    max_read_burst_length=256
#pragma HLS INTERFACE axis      port=out_stream
#pragma HLS INTERFACE s_axilite port=resize_w  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=resize_h  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=out_w     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=out_h     bundle=ctrl
#pragma HLS INTERFACE s_axilite port=img_resize bundle=ctrl
#pragma HLS INTERFACE s_axilite port=map_x      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=map_y      bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return     bundle=ctrl

    const float rw_max = (float)((int)resize_w - 1);
    const float rh_max = (float)((int)resize_h - 1);

    OUTPUT_ROW: for (dim_t y = 0; y < out_h; y++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=RESIZE_MAX_OUT_H

        OUTPUT_COL: for (dim_t x = 0; x < out_w; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=RESIZE_MAX_OUT_W
// Pipeline the column loop. Each iteration performs two sequential map reads
// then up to four img_resize reads. II will be > 1 when DDR latency > 1 cycle
// (expected for the m_axi scatter path) but the tool schedules them as a
// pipeline with backpressure.
#pragma HLS PIPELINE

            int map_idx = (int)y * (int)out_w + (int)x;
            float u = map_x[map_idx];
            float v = map_y[map_idx];

            // Out-of-bounds check: emit border constant 0.
            if (u < 0.0f || u > rw_max || v < 0.0f || v > rh_max) {
                out_stream.write((pixel_t)0);
                continue;
            }

            int u0 = (int)u;
            int v0 = (int)v;
            int u1 = (u0 + 1 <= (int)resize_w - 1) ? u0 + 1 : (int)resize_w - 1;
            int v1 = (v0 + 1 <= (int)resize_h - 1) ? v0 + 1 : (int)resize_h - 1;
            float du = u - (float)u0;
            float dv = v - (float)v0;

            pixel_t p00 = img_resize[v0 * (int)resize_w + u0];
            pixel_t p10 = img_resize[v0 * (int)resize_w + u1];
            pixel_t p01 = img_resize[v1 * (int)resize_w + u0];
            pixel_t p11 = img_resize[v1 * (int)resize_w + u1];

            out_stream.write(bilinear_rect(p00, p10, p01, p11, du, dv));
        }
    }
}
