#include "rectify.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// -----------------------------------------------------------------------
// C++ golden model — mirrors sw/rectify.py rectify() exactly.
// -----------------------------------------------------------------------
static void golden_rectify(
    const std::vector<uint8_t> &img_resize,
    int resize_w, int resize_h,
    const std::vector<float>   &map_x_data,
    const std::vector<float>   &map_y_data,
    std::vector<uint8_t>       &out_img,
    int out_w, int out_h)
{
    out_img.resize(out_w * out_h);

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            int map_idx = y * out_w + x;
            float u = map_x_data[map_idx];
            float v = map_y_data[map_idx];

            if (u < 0.f || u > (float)(resize_w - 1) ||
                v < 0.f || v > (float)(resize_h - 1)) {
                out_img[y * out_w + x] = 0;
                continue;
            }

            int u0 = (int)u;
            int v0 = (int)v;
            int u1 = (u0 + 1 < resize_w) ? u0 + 1 : resize_w - 1;
            int v1 = (v0 + 1 < resize_h) ? v0 + 1 : resize_h - 1;
            float du = u - (float)u0;
            float dv = v - (float)v0;

            float p00 = (float)img_resize[v0 * resize_w + u0];
            float p10 = (float)img_resize[v0 * resize_w + u1];
            float p01 = (float)img_resize[v1 * resize_w + u0];
            float p11 = (float)img_resize[v1 * resize_w + u1];

            float one_du = 1.f - du, one_dv = 1.f - dv;
            float acc = one_du * one_dv * p00
                      + du     * one_dv * p10
                      + one_du * dv     * p01
                      + du     * dv     * p11;
            if (acc < 0.f)   acc = 0.f;
            if (acc > 255.f) acc = 255.f;
            out_img[y * out_w + x] = (uint8_t)(int)acc;
        }
    }
}

// -----------------------------------------------------------------------
// Test runner
// -----------------------------------------------------------------------
static int run_case(
    const char *name,
    int resize_w, int resize_h,
    int out_w, int out_h,
    const std::vector<float> &map_x_data,
    const std::vector<float> &map_y_data,
    unsigned seed)
{
    // Build random source image.
    std::vector<uint8_t> img(resize_w * resize_h);
    srand(seed);
    for (int i = 0; i < resize_w * resize_h; i++) img[i] = (uint8_t)(rand() & 0xFF);

    // Drive HLS kernel.
    hls::stream<pixel_t> out_stream("out_stream");
    rectify_kernel(
        img.data(),
        map_x_data.data(),
        map_y_data.data(),
        out_stream,
        (dim_t)resize_w, (dim_t)resize_h,
        (dim_t)out_w,    (dim_t)out_h);

    std::vector<uint8_t> hw_out(out_w * out_h);
    for (int i = 0; i < out_w * out_h; i++)
        hw_out[i] = (uint8_t)(ap_uint<8>)out_stream.read();

    if (!out_stream.empty()) {
        printf("[%s] FAIL: out_stream has extra data\n", name);
        return 1;
    }

    // Compare against golden.
    std::vector<uint8_t> gold_out;
    golden_rectify(img, resize_w, resize_h,
                   map_x_data, map_y_data,
                   gold_out, out_w, out_h);

    int mismatches = 0, max_abs = 0, first_y = -1, first_x = -1;
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            int idx = y * out_w + x;
            int d = (int)hw_out[idx] - (int)gold_out[idx];
            if (d != 0) {
                mismatches++;
                int ad = d < 0 ? -d : d;
                if (ad > max_abs) max_abs = ad;
                if (first_y < 0) { first_y = y; first_x = x; }
            }
        }
    }

    if (mismatches == 0) {
        printf("[%s] PASS  (resize %dx%d -> out %dx%d)\n",
               name, resize_w, resize_h, out_w, out_h);
        return 0;
    } else {
        printf("[%s] FAIL  (resize %dx%d -> out %dx%d): "
               "%d mismatches, max|diff|=%d, first at (%d,%d) hw=%u gold=%u\n",
               name, resize_w, resize_h, out_w, out_h,
               mismatches, max_abs, first_y, first_x,
               (unsigned)hw_out[first_y * out_w + first_x],
               (unsigned)gold_out[first_y * out_w + first_x]);
        return 1;
    }
}

// -----------------------------------------------------------------------
// Build map helpers
// -----------------------------------------------------------------------

// Identity map: map_x[y,x]=x, map_y[y,x]=y  =>  output == source.
static void make_identity_maps(int w, int h,
                                std::vector<float> &mx,
                                std::vector<float> &my)
{
    mx.resize(w * h);
    my.resize(w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            mx[y * w + x] = (float)x;
            my[y * w + x] = (float)y;
        }
}

// Sub-pixel shift map: map_x[y,x]=x+dx, map_y[y,x]=y+dy.
static void make_shift_maps(int w, int h, float dx, float dy,
                             std::vector<float> &mx,
                             std::vector<float> &my)
{
    mx.resize(w * h);
    my.resize(w * h);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            mx[y * w + x] = (float)x + dx;
            my[y * w + x] = (float)y + dy;
        }
}

// Random maps in [0, rw-1] × [0, rh-1].
static void make_random_maps(int out_w, int out_h,
                              int rw, int rh,
                              std::vector<float> &mx,
                              std::vector<float> &my,
                              unsigned seed)
{
    mx.resize(out_w * out_h);
    my.resize(out_w * out_h);
    srand(seed + 1000);
    for (int i = 0; i < out_w * out_h; i++) {
        mx[i] = (float)(rand() % (rw * 100)) / 100.0f;  // sub-pixel
        my[i] = (float)(rand() % (rh * 100)) / 100.0f;
    }
}

// -----------------------------------------------------------------------
// main
// -----------------------------------------------------------------------
int main()
{
    int failures = 0;
    std::vector<float> mx, my;

    // 1. Identity: output must equal source exactly.
    make_identity_maps(16, 12, mx, my);
    failures += run_case("identity-16x12", 16, 12, 16, 12, mx, my, 1);

    // 2. Identity on a non-square image.
    make_identity_maps(24, 8, mx, my);
    failures += run_case("identity-24x8", 24, 8, 24, 8, mx, my, 2);

    // 3. Sub-pixel shift of 0.5 px in both axes.
    make_shift_maps(14, 10, 0.5f, 0.5f, mx, my);
    failures += run_case("shift-0.5-14x10", 16, 12, 14, 10, mx, my, 3);

    // 4. Output smaller than source (crop-like maps).
    {
        int rw = 20, rh = 16, ow = 10, oh = 8;
        // Sample from the centre quarter of the source.
        mx.resize(ow * oh); my.resize(ow * oh);
        for (int y = 0; y < oh; y++)
            for (int x = 0; x < ow; x++) {
                mx[y * ow + x] = 5.0f + (float)x;
                my[y * ow + x] = 4.0f + (float)y;
            }
        failures += run_case("crop-10x8-from-20x16", rw, rh, ow, oh, mx, my, 4);
    }

    // 5. Out-of-bounds coordinates -> 0 (border constant).
    {
        int rw = 8, rh = 6, ow = 4, oh = 3;
        mx.assign(ow * oh, -5.0f);   // all OOB
        my.assign(ow * oh, -5.0f);
        failures += run_case("oob-border", rw, rh, ow, oh, mx, my, 5);
    }

    // 6. Random fractional maps — stress test bilinear path.
    make_random_maps(18, 14, 20, 16, mx, my, 6);
    failures += run_case("random-frac-18x14", 20, 16, 18, 14, mx, my, 6);

    // 7. Larger synthetic case.
    make_random_maps(60, 45, 64, 48, mx, my, 7);
    failures += run_case("random-frac-60x45", 64, 48, 60, 45, mx, my, 7);

    printf("\n%s  (%d failure(s))\n",
           failures == 0 ? "ALL PASS" : "SOME TESTS FAILED", failures);
    return failures ? 1 : 0;
}
