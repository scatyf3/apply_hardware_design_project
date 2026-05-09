# Results

This document summarizes the verification and synthesis results for the camera preprocessing HLS project.

The project implements two image-processing kernels:

1. `resize_kernel`: bilinear image resize.
2. `rectify_kernel`: map-based image rectification / remap using `map_x` and `map_y`.

The goal of this project is to implement and verify HLS IP kernels using a course-style hardware design flow:

```text
Python golden model
→ Python tests
→ HLS C++ implementation
→ C++ testbench
→ Vitis HLS C simulation
→ HLS synthesis to RTL
→ Vitis HLS C/RTL co-simulation
