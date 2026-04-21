#ifndef RESIZE_H
#define RESIZE_H

#include <ap_int.h>
#include <hls_stream.h>

// Compile-time upper bounds — line buffers and loop trip counts are sized by
// these, so they must be large enough for any runtime frame the PS configures.
#ifndef RESIZE_MAX_IN_W
#define RESIZE_MAX_IN_W  1920
#endif
#ifndef RESIZE_MAX_IN_H
#define RESIZE_MAX_IN_H  1080
#endif
#ifndef RESIZE_MAX_OUT_W
#define RESIZE_MAX_OUT_W 1920
#endif
#ifndef RESIZE_MAX_OUT_H
#define RESIZE_MAX_OUT_H 1080
#endif

typedef ap_uint<8>  pixel_t;
typedef ap_uint<13> dim_t;   // supports up to 8191

void resize_kernel(
    hls::stream<pixel_t> &in_stream,
    hls::stream<pixel_t> &out_stream,
    dim_t in_w,
    dim_t in_h,
    dim_t out_w,
    dim_t out_h,
    float scale_x,  // in_w / out_w (pre-computed by PS to avoid an HLS divider)
    float scale_y); // in_h / out_h

#endif
