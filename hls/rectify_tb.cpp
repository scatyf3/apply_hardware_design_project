#include "rectify.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#define MAX_PIXELS 2048

static pixel_t hw_in[MAX_PIXELS];
static pixel_t hw_out[MAX_PIXELS];
static float map_x_buf[MAX_PIXELS];
static float map_y_buf[MAX_PIXELS];
static uint8_t gold[MAX_PIXELS];

static float clamp_float_tb(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static int clamp_int_tb(int v, int lo, int hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

static uint8_t golden_sample(
    const pixel_t *img,
    int w,
    int h,
    float u,
    float v
) {
    float uc = clamp_float_tb(u, 0.0f, (float)(w - 1));
    float vc = clamp_float_tb(v, 0.0f, (float)(h - 1));

    int x0 = (int)uc;
    int y0 = (int)vc;
    int x1 = clamp_int_tb(x0 + 1, 0, w - 1);
    int y1 = clamp_int_tb(y0 + 1, 0, h - 1);

    float dx = uc - (float)x0;
    float dy = vc - (float)y0;

    float p00 = (float)(uint8_t)img[y0 * w + x0];
    float p10 = (float)(uint8_t)img[y0 * w + x1];
    float p01 = (float)(uint8_t)img[y1 * w + x0];
    float p11 = (float)(uint8_t)img[y1 * w + x1];

    float one_dx = 1.0f - dx;
    float one_dy = 1.0f - dy;

    float acc =
        one_dx * one_dy * p00 +
        dx     * one_dy * p10 +
        one_dx * dy     * p01 +
        dx     * dy     * p11;

    if (acc < 0.0f) acc = 0.0f;
    if (acc > 255.0f) acc = 255.0f;

    return (uint8_t)((int)acc);
}

static void clear_buffers() {
    for (int i = 0; i < MAX_PIXELS; i++) {
        hw_in[i] = 0;
        hw_out[i] = 0;
        map_x_buf[i] = 0.0f;
        map_y_buf[i] = 0.0f;
        gold[i] = 0;
    }
}

static void make_input(int w, int h, unsigned seed) {
    for (int i = 0; i < w * h; i++) {
        unsigned v = (unsigned)(i * 17 + seed * 29 + (i % 13) * 7);
        hw_in[i] = (pixel_t)(v & 0xFF);
    }
}

static int run_case(
    const char *name,
    int in_w,
    int in_h,
    int out_w,
    int out_h,
    int mode
) {
    if (in_w * in_h > MAX_PIXELS || out_w * out_h > MAX_PIXELS) {
        printf("[%s] FAIL: case exceeds MAX_PIXELS\n", name);
        return 1;
    }

    clear_buffers();
    make_input(in_w, in_h, (unsigned)(mode + 1));

    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            int idx = y * out_w + x;

            float u = 0.0f;
            float v = 0.0f;

            if (mode == 0) {
                // Identity map.
                u = (float)x;
                v = (float)y;
            } else if (mode == 1) {
                // Fractional shift, exercises bilinear interpolation.
                u = (float)x + 0.25f;
                v = (float)y + 0.50f;
            } else if (mode == 2) {
                // Crop from a larger image.
                u = (float)x + 3.0f;
                v = (float)y + 2.0f;
            } else if (mode == 3) {
                // Border clamp stress test.
                if ((x + y) % 4 == 0) {
                    u = -5.0f;
                    v = -3.0f;
                } else if ((x + y) % 4 == 1) {
                    u = (float)in_w + 8.0f;
                    v = (float)y;
                } else if ((x + y) % 4 == 2) {
                    u = (float)x;
                    v = (float)in_h + 9.0f;
                } else {
                    u = (float)in_w + 10.0f;
                    v = (float)in_h + 10.0f;
                }
            } else {
                // Scale-like map.
                u = ((float)x + 0.5f) * ((float)in_w / (float)out_w) - 0.5f;
                v = ((float)y + 0.5f) * ((float)in_h / (float)out_h) - 0.5f;
            }

            map_x_buf[idx] = u;
            map_y_buf[idx] = v;
            gold[idx] = golden_sample(hw_in, in_w, in_h, u, v);
        }
    }

    rectify_kernel(
        hw_in,
        map_x_buf,
        map_y_buf,
        hw_out,
        (dim_t)in_w,
        (dim_t)in_h,
        (dim_t)out_w,
        (dim_t)out_h
    );

    int errors = 0;
    int first = -1;
    int max_abs = 0;

    for (int i = 0; i < out_w * out_h; i++) {
        int got = (int)(uint8_t)hw_out[i];
        int exp = (int)gold[i];
        int d = got - exp;
        int ad = d < 0 ? -d : d;

        if (ad != 0) {
            errors++;
            if (first < 0) first = i;
            if (ad > max_abs) max_abs = ad;
        }
    }

    if (errors == 0) {
        printf("[%s] PASS (%dx%d -> %dx%d)\n", name, in_w, in_h, out_w, out_h);
        return 0;
    }

    printf("[%s] FAIL errors=%d first=%d hw=%u gold=%u max_abs=%d\n",
           name,
           errors,
           first,
           first >= 0 ? (unsigned)(uint8_t)hw_out[first] : 0,
           first >= 0 ? (unsigned)gold[first] : 0,
           max_abs);

    return 1;
}

// Read expected_count elements of T from path into a vector. Empty on failure.
template <typename T>
static std::vector<T> load_bin_vec(const std::string &path, size_t expected_count) {
    std::vector<T> buf;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return buf;
    buf.resize(expected_count);
    size_t got = fread(buf.data(), sizeof(T), expected_count, f);
    fclose(f);
    if (got != expected_count) buf.clear();
    return buf;
}

// Drives rectify_kernel with real .bin fixtures produced by
// sw/prepare_real_rectify_data.py and memcmps the result against the Python
// golden's _gold.bin output. Buffers are heap-allocated so the static
// MAX_PIXELS budget for the synthetic cases above doesn't apply.
static int run_real_case(
    const char *name,
    const std::string &in_path,
    const std::string &mx_path,
    const std::string &my_path,
    const std::string &gold_path,
    int in_w, int in_h, int out_w, int out_h
) {
    std::vector<uint8_t> in_img =
        load_bin_vec<uint8_t>(in_path, (size_t)in_w * in_h);
    if (in_img.empty()) {
        printf("[real-data %s] FAIL: could not read %s\n", name, in_path.c_str());
        return 1;
    }
    std::vector<float> map_x_buf =
        load_bin_vec<float>(mx_path, (size_t)out_w * out_h);
    if (map_x_buf.empty()) {
        printf("[real-data %s] FAIL: could not read %s\n", name, mx_path.c_str());
        return 1;
    }
    std::vector<float> map_y_buf =
        load_bin_vec<float>(my_path, (size_t)out_w * out_h);
    if (map_y_buf.empty()) {
        printf("[real-data %s] FAIL: could not read %s\n", name, my_path.c_str());
        return 1;
    }
    std::vector<uint8_t> gold_out =
        load_bin_vec<uint8_t>(gold_path, (size_t)out_w * out_h);
    if (gold_out.empty()) {
        printf("[real-data %s] FAIL: could not read %s\n", name, gold_path.c_str());
        return 1;
    }

    std::vector<pixel_t> in_buf((size_t)in_w * in_h);
    for (size_t i = 0; i < in_buf.size(); i++) in_buf[i] = (pixel_t)in_img[i];
    std::vector<pixel_t> out_buf((size_t)out_w * out_h);

    rectify_kernel(
        in_buf.data(),
        map_x_buf.data(),
        map_y_buf.data(),
        out_buf.data(),
        (dim_t)in_w, (dim_t)in_h,
        (dim_t)out_w, (dim_t)out_h
    );

    int errors = 0;
    int first = -1;
    int max_abs = 0;
    for (int i = 0; i < out_w * out_h; i++) {
        int got = (int)(uint8_t)out_buf[i];
        int exp = (int)gold_out[i];
        int d = got - exp;
        int ad = d < 0 ? -d : d;
        if (ad != 0) {
            errors++;
            if (first < 0) first = i;
            if (ad > max_abs) max_abs = ad;
        }
    }

    if (errors == 0) {
        printf("[real-data %s] PASS (%dx%d -> %dx%d)\n",
               name, in_w, in_h, out_w, out_h);
        return 0;
    }
    printf("[real-data %s] FAIL (%dx%d -> %dx%d): errors=%d first=%d hw=%u gold=%u max_abs=%d\n",
           name, in_w, in_h, out_w, out_h, errors, first,
           first >= 0 ? (unsigned)(uint8_t)out_buf[first] : 0,
           first >= 0 ? (unsigned)gold_out[first] : 0,
           max_abs);
    return 1;
}

// Vitis HLS C-sim launches from .../rectify_hls/sol1/csim/build, so walk a few
// candidate paths until we find testdata/rectify_cases.txt — same trick as the
// resize testbench.
static std::string find_rectify_testdata_dir() {
    static const char *candidates[] = {
        "testdata",
        "../testdata",
        "../../testdata",
        "../../../testdata",
        "../../../../testdata",
        "../../../../../testdata",
    };
    for (const char *c : candidates) {
        std::string p = std::string(c) + "/rectify_cases.txt";
        FILE *f = fopen(p.c_str(), "r");
        if (f) { fclose(f); return c; }
    }
    return std::string();
}

static int run_real_data_tests() {
    // Allow callers (run_rectify_hls.tcl cosim path) to skip this tier — cosim's
    // m_axi pointer wrapper cannot handle the 384x288 buffers we use for the
    // real-data cases and segfaults the C TB process. csim still exercises them
    // end-to-end, so RTL/C equivalence on the synthetic cases is sufficient.
    const char *skip = getenv("RECTIFY_TB_SKIP_REAL_DATA");
    if (skip && skip[0] && skip[0] != '0') {
        printf("[real-data] skipped (RECTIFY_TB_SKIP_REAL_DATA=%s)\n", skip);
        return 0;
    }

    std::string dir = find_rectify_testdata_dir();
    if (dir.empty()) {
        printf("[real-data] testdata/rectify_cases.txt not found — "
               "skipping (run sw/prepare_real_rectify_data.py to generate)\n");
        return 0;
    }
    printf("[real-data] using %s/\n", dir.c_str());

    std::string manifest = dir + "/rectify_cases.txt";
    FILE *f = fopen(manifest.c_str(), "r");
    if (!f) {
        printf("[real-data] open failed: %s\n", manifest.c_str());
        return 1;
    }

    int fails = 0;
    char name[128];
    int in_h, in_w, out_h, out_w;
    while (fscanf(f, "%127s %d %d %d %d", name, &in_h, &in_w, &out_h, &out_w) == 5) {
        std::string in_p   = dir + "/" + name + "_in.bin";
        std::string mx_p   = dir + "/" + name + "_map_x.bin";
        std::string my_p   = dir + "/" + name + "_map_y.bin";
        std::string gold_p = dir + "/" + name + "_gold.bin";
        fails += run_real_case(name, in_p, mx_p, my_p, gold_p,
                               in_w, in_h, out_w, out_h);
    }
    fclose(f);
    return fails;
}

int main() {
    int fails = 0;

    fails += run_case("identity",      32, 24, 32, 24, 0);
    fails += run_case("fractional",    32, 24, 32, 24, 1);
    fails += run_case("crop",          48, 36, 24, 18, 2);
    fails += run_case("border_clamp",  32, 24, 32, 24, 3);
    fails += run_case("scale_map",     23, 17, 29, 11, 4);

    fails += run_real_data_tests();

    if (fails == 0) {
        printf("\nAll HLS rectify tests passed.\n");
        return 0;
    }

    printf("\n%d HLS rectify test(s) failed.\n", fails);
    return 1;
}
