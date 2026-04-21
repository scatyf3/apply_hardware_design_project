#include "resize.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// C++ port of sw/resize.py resize() — same arithmetic order and precision
// (float32 throughout) so a bit-exact match with the HLS kernel is expected.
static void golden_resize(
    const std::vector<uint8_t> &in_img,
    int in_w, int in_h,
    std::vector<uint8_t> &out_img,
    int out_w, int out_h)
{
    float scale_x = (float)in_w / (float)out_w;
    float scale_y = (float)in_h / (float)out_h;
    out_img.resize(out_w * out_h);

    for (int y = 0; y < out_h; y++) {
        float v_f = scale_y * (float)y;
        if (v_f < 0.f) v_f = 0.f;
        if (v_f > (float)(in_h - 1)) v_f = (float)(in_h - 1);
        int v0i = (int)v_f;
        int v1i = v0i + 1 > in_h - 1 ? in_h - 1 : v0i + 1;
        float dv = v_f - (float)v0i;

        for (int x = 0; x < out_w; x++) {
            float u_f = scale_x * (float)x;
            if (u_f < 0.f) u_f = 0.f;
            if (u_f > (float)(in_w - 1)) u_f = (float)(in_w - 1);
            int u0i = (int)u_f;
            int u1i = u0i + 1 > in_w - 1 ? in_w - 1 : u0i + 1;
            float du = u_f - (float)u0i;

            float p00 = (float)in_img[v0i * in_w + u0i];
            float p10 = (float)in_img[v0i * in_w + u1i];
            float p01 = (float)in_img[v1i * in_w + u0i];
            float p11 = (float)in_img[v1i * in_w + u1i];

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

static int run_case(const char *name, int in_w, int in_h, int out_w, int out_h,
                    unsigned seed)
{
    std::vector<uint8_t> in_img(in_w * in_h);
    srand(seed);
    for (int i = 0; i < in_w * in_h; i++) in_img[i] = (uint8_t)(rand() & 0xFF);

    hls::stream<pixel_t> in_stream("in_stream");
    hls::stream<pixel_t> out_stream("out_stream");
    for (int i = 0; i < in_w * in_h; i++) in_stream.write((pixel_t)in_img[i]);

    float sx = (float)in_w / (float)out_w;
    float sy = (float)in_h / (float)out_h;
    resize_kernel(in_stream, out_stream, in_w, in_h, out_w, out_h, sx, sy);

    std::vector<uint8_t> hw_out(out_w * out_h);
    for (int i = 0; i < out_w * out_h; i++) hw_out[i] = (uint8_t)(ap_uint<8>)out_stream.read();

    if (!in_stream.empty()) {
        printf("[%s] FAIL: in_stream not fully consumed\n", name);
        return 1;
    }
    if (!out_stream.empty()) {
        printf("[%s] FAIL: out_stream has extra data\n", name);
        return 1;
    }

    std::vector<uint8_t> gold_out;
    golden_resize(in_img, in_w, in_h, gold_out, out_w, out_h);

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
        printf("[%s] PASS  (%dx%d -> %dx%d)\n", name, in_w, in_h, out_w, out_h);
        return 0;
    } else {
        printf("[%s] FAIL  (%dx%d -> %dx%d): %d mismatches, max|diff|=%d, first at (%d,%d) hw=%u gold=%u\n",
               name, in_w, in_h, out_w, out_h, mismatches, max_abs, first_y, first_x,
               (unsigned)hw_out[first_y * out_w + first_x],
               (unsigned)gold_out[first_y * out_w + first_x]);
        return 1;
    }
}

// Read expected_bytes from path into a vector. Returns empty on any failure.
static std::vector<uint8_t> load_bin(const std::string &path, size_t expected_bytes)
{
    std::vector<uint8_t> buf;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return buf;
    buf.resize(expected_bytes);
    size_t got = fread(buf.data(), 1, expected_bytes, f);
    fclose(f);
    if (got != expected_bytes) buf.clear();
    return buf;
}

// Streams a real image through resize_kernel and compares against the golden
// bin produced by sw/prepare_real_data.py (both files use the same Python
// golden that already matched the HLS kernel bit-exactly on synthetic data).
static int run_real_case(const char *name,
                         const std::string &in_path, const std::string &gold_path,
                         int in_w, int in_h, int out_w, int out_h)
{
    std::vector<uint8_t> in_img = load_bin(in_path, (size_t)in_w * in_h);
    if (in_img.empty()) {
        printf("[real-data %s] FAIL: could not read %s (%dx%d bytes)\n",
               name, in_path.c_str(), in_h, in_w);
        return 1;
    }
    std::vector<uint8_t> gold_out = load_bin(gold_path, (size_t)out_w * out_h);
    if (gold_out.empty()) {
        printf("[real-data %s] FAIL: could not read %s\n", name, gold_path.c_str());
        return 1;
    }

    hls::stream<pixel_t> in_stream("in_stream");
    hls::stream<pixel_t> out_stream("out_stream");
    for (int i = 0; i < in_w * in_h; i++) in_stream.write((pixel_t)in_img[i]);

    float sx = (float)in_w / (float)out_w;
    float sy = (float)in_h / (float)out_h;
    resize_kernel(in_stream, out_stream, in_w, in_h, out_w, out_h, sx, sy);

    std::vector<uint8_t> hw_out(out_w * out_h);
    for (int i = 0; i < out_w * out_h; i++)
        hw_out[i] = (uint8_t)(ap_uint<8>)out_stream.read();

    if (!in_stream.empty() || !out_stream.empty()) {
        printf("[real-data %s] FAIL: stream mis-sized after kernel\n", name);
        return 1;
    }

    int mismatches = 0, max_abs = 0, first_y = -1, first_x = -1;
    for (int y = 0; y < out_h; y++) {
        for (int x = 0; x < out_w; x++) {
            int d = (int)hw_out[y * out_w + x] - (int)gold_out[y * out_w + x];
            if (d != 0) {
                mismatches++;
                int ad = d < 0 ? -d : d;
                if (ad > max_abs) max_abs = ad;
                if (first_y < 0) { first_y = y; first_x = x; }
            }
        }
    }

    if (mismatches == 0) {
        printf("[real-data %s] PASS  (%dx%d -> %dx%d)\n", name, in_w, in_h, out_w, out_h);
        return 0;
    }
    printf("[real-data %s] FAIL  (%dx%d -> %dx%d): %d mismatches, max|diff|=%d, first (%d,%d) hw=%u gold=%u\n",
           name, in_w, in_h, out_w, out_h, mismatches, max_abs, first_y, first_x,
           (unsigned)hw_out[first_y * out_w + first_x],
           (unsigned)gold_out[first_y * out_w + first_x]);
    return 1;
}

// Vitis HLS C-sim runs from .../resize_hls/sol1/csim/build — so the testdata
// lives several levels up. Try a handful of relative paths so the same tb
// works whether we're launched from hls/ directly or from inside a sim tree.
static std::string find_testdata_dir()
{
    static const char *candidates[] = {
        "testdata",
        "../testdata",
        "../../testdata",
        "../../../testdata",
        "../../../../testdata",
        "../../../../../testdata",
    };
    for (const char *c : candidates) {
        std::string p = std::string(c) + "/cases.txt";
        FILE *f = fopen(p.c_str(), "r");
        if (f) { fclose(f); return c; }
    }
    return std::string();
}

static int run_real_data_tests()
{
    std::string dir = find_testdata_dir();
    if (dir.empty()) {
        printf("[real-data] testdata/cases.txt not found — "
               "skipping (run sw/prepare_real_data.py to generate)\n");
        return 0;
    }
    printf("[real-data] using %s/\n", dir.c_str());

    std::string manifest = dir + "/cases.txt";
    FILE *f = fopen(manifest.c_str(), "r");
    if (!f) { printf("[real-data] open failed: %s\n", manifest.c_str()); return 1; }

    int fails = 0;
    char name[128];
    int in_h, in_w, out_h, out_w;
    while (fscanf(f, "%127s %d %d %d %d", name, &in_h, &in_w, &out_h, &out_w) == 5) {
        std::string in_p   = dir + "/" + name + "_in.bin";
        std::string gold_p = dir + "/" + name + "_gold.bin";
        fails += run_real_case(name, in_p, gold_p, in_w, in_h, out_w, out_h);
    }
    fclose(f);
    return fails;
}

int main()
{
    int fails = 0;
    fails += run_case("downscale",     64, 48, 32, 24, 1);
    fails += run_case("upscale",       16, 12, 48, 36, 2);
    fails += run_case("identity",      32, 32, 32, 32, 3);
    fails += run_case("non-integer",   23, 17, 29, 11, 4);
    fails += run_case("wide",         128, 16, 64, 16, 5);

    fails += run_real_data_tests();

    if (fails == 0) {
        printf("\nAll HLS resize tests passed.\n");
        return 0;
    }
    printf("\n%d test(s) failed.\n", fails);
    return 1;
}
