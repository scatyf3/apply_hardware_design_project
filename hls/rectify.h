#ifndef RECTIFY_H
#define RECTIFY_H

#include <ap_int.h>

#ifndef RECTIFY_MAX_IN_W
#define RECTIFY_MAX_IN_W 1920
#endif
#ifndef RECTIFY_MAX_IN_H
#define RECTIFY_MAX_IN_H 1080
#endif
#ifndef RECTIFY_MAX_OUT_W
#define RECTIFY_MAX_OUT_W 1920
#endif
#ifndef RECTIFY_MAX_OUT_H
#define RECTIFY_MAX_OUT_H 1080
#endif

typedef ap_uint<8> pixel_t;
typedef ap_uint<13> dim_t;  // supports up to 8191

// Baseline rectify / remap kernel.
// img_in:  source image, normally the resized frame, row-major grayscale uint8
// map_x:   output-sized table; map_x[y*out_w+x] gives source u coordinate
// map_y:   output-sized table; map_y[y*out_w+x] gives source v coordinate
// img_out: rectified output image, row-major grayscale uint8
void rectify_kernel(
    const pixel_t *img_in,
    const float *map_x,
    const float *map_y,
    pixel_t *img_out,
    dim_t in_w,
    dim_t in_h,
    dim_t out_w,
    dim_t out_h);

#endif
