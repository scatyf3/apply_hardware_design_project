#ifndef RECTIFY_H
#define RECTIFY_H

// Reuse the shared pixel / dim types and MAX dimension constants from resize.h.
#include "resize.h"

// Flat-array sizes used for m_axi depth hints (compile-time constants only).
#define RECT_MAX_RESIZE_PIXELS (RESIZE_MAX_IN_W  * RESIZE_MAX_IN_H)
#define RECT_MAX_OUT_PIXELS    (RESIZE_MAX_OUT_W * RESIZE_MAX_OUT_H)

// Baseline rectify kernel.
//
// Reads img_resize and the two rectification maps from memory (AXI master),
// and produces the remapped image as an AXI4-Stream to the output.
//
// Interface:
//   img_resize — flat row-major pixel array of the resize-stage output,
//                size resize_h × resize_w.
//   map_x      — float32 flat array [out_h × out_w]: fractional column (u)
//                to sample in img_resize for each output pixel.
//   map_y      — float32 flat array [out_h × out_w]: fractional row (v).
//   out_stream — AXI4-Stream of output pixels, row-major raster order.
//   resize_w / resize_h — dimensions of img_resize.
//   out_w / out_h       — dimensions of the output (and of the maps).
//
// Coordinate convention matches sw/rectify.py:
//   u = map_x[y, x],  v = map_y[y, x]
//   I_out[y,x] = bilinear(img_resize, u, v)
// Out-of-bounds coordinates produce pixel value 0 (border constant).

void rectify_kernel(
    const pixel_t             *img_resize,
    const float               *map_x,
    const float               *map_y,
    hls::stream<pixel_t>      &out_stream,
    dim_t resize_w,
    dim_t resize_h,
    dim_t out_w,
    dim_t out_h);

#endif // RECTIFY_H
