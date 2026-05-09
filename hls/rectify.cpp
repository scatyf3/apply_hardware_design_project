#include "rectify.h"

static float clamp_float(float x, float lo, float hi) {
#pragma HLS INLINE
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static pixel_t bilinear_sample_mem(
    const pixel_t *img,
    dim_t in_w,
    dim_t in_h,
    float u,
    float v) {
#pragma HLS INLINE
    float u_max = (float)((int)in_w - 1);
    float v_max = (float)((int)in_h - 1);

    // BORDER_REPLICATE behavior: out-of-range map coordinates clamp to edge.
    u = clamp_float(u, 0.0f, u_max);
    v = clamp_float(v, 0.0f, v_max);

    int u0 = (int)u;  // floor after non-negative clamp
    int v0 = (int)v;
    int u1 = u0 + 1;
    int v1 = v0 + 1;
    if (u1 > (int)in_w - 1) u1 = (int)in_w - 1;
    if (v1 > (int)in_h - 1) v1 = (int)in_h - 1;

    float du = u - (float)u0;
    float dv = v - (float)v0;

    float one_du = 1.0f - du;
    float one_dv = 1.0f - dv;

    pixel_t p00 = img[v0 * (int)in_w + u0];
    pixel_t p10 = img[v0 * (int)in_w + u1];
    pixel_t p01 = img[v1 * (int)in_w + u0];
    pixel_t p11 = img[v1 * (int)in_w + u1];

    float acc = one_du * one_dv * (float)p00
              + du     * one_dv * (float)p10
              + one_du * dv     * (float)p01
              + du     * dv     * (float)p11;

    if (acc < 0.0f) acc = 0.0f;
    if (acc > 255.0f) acc = 255.0f;

    // Match the existing resize convention: truncate to uint8, not round.
    return (pixel_t)(ap_uint<8>)(int)acc;
}

void rectify_kernel(
    const pixel_t *img_in,
    const float *map_x,
    const float *map_y,
    pixel_t *img_out,
    dim_t in_w,
    dim_t in_h,
    dim_t out_w,
    dim_t out_h) {
#pragma HLS INTERFACE m_axi port=img_in  offset=slave bundle=gmem_img  depth=2048
#pragma HLS INTERFACE m_axi port=map_x   offset=slave bundle=gmem_mapx depth=2048
#pragma HLS INTERFACE m_axi port=map_y   offset=slave bundle=gmem_mapy depth=2048
#pragma HLS INTERFACE m_axi port=img_out offset=slave bundle=gmem_out  depth=2048

#pragma HLS INTERFACE s_axilite port=img_in  bundle=ctrl
#pragma HLS INTERFACE s_axilite port=map_x   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=map_y   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=img_out bundle=ctrl
#pragma HLS INTERFACE s_axilite port=in_w    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=in_h    bundle=ctrl
#pragma HLS INTERFACE s_axilite port=out_w   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=out_h   bundle=ctrl
#pragma HLS INTERFACE s_axilite port=return  bundle=ctrl

OUT_Y:
    for (dim_t y = 0; y < out_h; y++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=RECTIFY_MAX_OUT_H
OUT_X:
        for (dim_t x = 0; x < out_w; x++) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=RECTIFY_MAX_OUT_W
#pragma HLS PIPELINE II=1
            int out_idx = (int)y * (int)out_w + (int)x;
            float u = map_x[out_idx];
            float v = map_y[out_idx];
            img_out[out_idx] = bilinear_sample_mem(img_in, in_w, in_h, u, v);
        }
    }
}

