# Real-Time ROS 2 Camera Preprocessing Accelerator

## Project Team

- Jason Pan
- Yifei Feng 

## Motivation

Robotics and edge AI systems often need to preprocess camera images before sending them to downstream perception modules such as object detection, visual odometry, SLAM, or navigation. Common preprocessing steps include resizing the image to a target resolution and rectifying the image to correct lens distortion or camera geometry. These operations must often be performed in real time, frame after frame, making them a good candidate for hardware acceleration.

![](.asserts/Overview.png)

## IP definition

In this project, we design a hardware IP that accelerates two common image preprocessing operations:
1.	Image resize
2.	Image rectify / remap

The goal is to **offload** these operations from the processor system (PS) to the programmable logic (PL), so that the CPU can spend less time on repetitive per-pixel computation and more time on higher-level robotics tasks.

The input to the IP is a camera frame $I_{in}$, together with configuration parameters such as input/output image sizes and rectification maps. The output is a processed frame $I_{out}$ that has been resized and rectified, ready for use by downstream ROS 2 nodes.


### resize

The resize operation maps each output pixel (x,y) to a source coordinate (u,v) in the input image:

$$u = s_x x,\qquad v = s_y y$$

where $s_x = W_{in}/W_{out}$ and $s_y = H_{in}/H_{out}$. Since (u,v) may not lie exactly on an integer pixel location, the output pixel value is computed with bilinear interpolation from nearby source pixels:

$$I_{resize}(x,y)=\sum_{i=0}^{1}\sum_{j=0}^{1} w_{ij} I_{in}(u_i,v_j)$$

```python
# Resize
for y in range(H_resize):
    for x in range(W_resize):
        u = scale_x * x
        v = scale_y * y
        I_resize[y,x] = bilinear_sample(I_in, u, v)
```

### rectify

The rectify operation uses two remap tables, $map_x$ and $map_y$, that specify where each output pixel should sample from the input image:

$$u = map_x(x,y), \qquad v = map_y(x,y)$$

and then computes:

$$I_{rect}(x,y) = I_{resize}(u,v)$$

again using bilinear interpolation.

```python
# Rectify / remap
for y in range(H_out):
    for x in range(W_out):
        u = map_x[y,x]
        v = map_y[y,x]
        I_out[y,x] = bilinear_sample(I_resize, u, v)
```

In practice, $map_x$ and $map_y$ are usually generated offline from camera calibration parameters, then reused at runtime. As an optimization direction (if schedule permits), we can further constrain remap access to a bounded local window so that a small on-chip line buffer can satisfy most 2×2 bilinear neighborhoods without external memory reads. We can also introduce a schedulable output queue that defers pixel emission until required source rows are resident, decoupling map-driven access from strict raster output order. Before adopting this optimization path, an offline map-quality check (per-pixel displacement bounds and expected buffer hit rate) can be used to estimate feasibility; rare misses may still be handled through a slow-path DDR fallback to preserve functional correctness.



### overall pipeline

These operations are well-suited for hardware acceleration because **they are highly regular, operate on every pixel in a similar way**, and can be organized as a streaming pipeline. The same arithmetic is repeated across the entire image, which makes the design amenable to pipelining, buffering, and parallel processing in hardware. AMD’s ROS 2 Perception Node accelerated application specifically uses resize and rectify as hardware-accelerated image pipeline stages, which supports this choice as a realistic robotics use case.  

The operation in hardware is as follows:
-	PS receives a camera frame from a ROS 2 image topic or a replayed dataset.
-	PS loads frame pointers, image dimensions, and rectification map addresses into the IP.
-	IP reads the input image and processes it through resize and rectify stages.
-	IP writes the processed image to output memory.
-	PS publishes the processed image as a new ROS 2 topic for downstream perception modules.


```python
# Offline-validated displacement bounds for stream-friendly remap
D_x, D_y = map_displacement_bounds

# Stage 1: generate resized pixels as a stream (and keep optional backing store)
for yr in range(H_resize):
    for xr in range(W_resize):
        u0 = xr * W_in / W_resize
        v0 = yr * H_in / H_resize
        p_resize = bilinear_sample(I_in, u0, v0)
        stream_resize.write((xr, yr, p_resize))
        I_resize_ddr[yr, xr] = p_resize  # Slow-path source for rare misses

# Stage 2: remap with local-window buffering and schedulable output queue
linebuf = SlidingWindowRows(width=W_resize, depth=D_y + 2)
ready_q = OutputQueue()

while stream_resize.not_empty():
    xr, yr, p_resize = stream_resize.read()
    linebuf.push(xr, yr, p_resize)

    for (x, y) in outputs_activated_by_row(yr, D_y):
        ready_q.push((x, y))

    while ready_q.not_empty() and dependencies_ready(ready_q.front(), yr, D_y):
        x, y = ready_q.pop()
        u1 = map_x[y, x]
        v1 = map_y[y, x]

        if in_local_window(u1, v1, yr, D_x, D_y):
            I_out[y, x] = bilinear_sample_from_linebuf(linebuf, u1, v1, border_mode="clamp")
        else:
            I_out[y, x] = bilinear_sample(I_resize_ddr, u1, v1, border_mode="clamp")
```




## IP Architecture

The preprocessing can be performed as follows:
-	PS stores the input frame in shared memory and supplies source and destination buffer addresses to the IP.
-	PS writes control registers for frame size, output size, and rectification map addresses through AXI4-Lite (with optional displacement-bound parameters $D_x$, $D_y$ for the advanced streaming optimization).
-	IP reads image pixels from memory and converts them into an internal AXI4-Stream pixel stream.
-	A resize module performs coordinate generation and bilinear interpolation to produce a resized pixel stream.
-	The resized stream feeds the rectify stage. In the baseline design, rectify accesses the resized frame through standard memory reads.
-	As a time-permitting optimization, rectify can be upgraded to a line-buffered design with depth $D_y + 2$, an output scheduling queue, and local-window hit checks.
-	In that optimized mode, rare misses outside the local window can use a slow-path read from the backing store ($I_{resize}$ in DDR) to preserve correctness.
-	A frame output module writes the final processed image to shared memory.
-	The IP reports completion and optional timing/performance counters back to the PS through status registers.

A modular version of the architecture is:
-	Frame ingress module
Reads image data from PS-visible memory and formats it as an AXI4-Stream pixel sequence for downstream compute blocks.
-	Resize module
Computes source coordinates for each output pixel and performs bilinear interpolation. This module is parameterized by input and output image dimensions.
-	Rectify module
Applies geometric remapping using calibration maps. Baseline implementation uses standard memory access; if time permits, it can be extended with bounded local displacement, line buffers, a schedulable output queue, and local-window hit testing, with DDR-backed $I_{resize}$ used on misses.
-	Frame egress module
Collects the processed pixel stream and writes the result back to shared memory so the PS can access it.
-	Control / status module
Exposes configuration registers such as source address, destination address, dimensions, map addresses, displacement bounds ($D_x$, $D_y$), start, done, and optional cycle counters.

The interfaces are chosen as follows:
-	Shared memory is used between PS and IP for input/output frame buffers and rectification maps. Optional backing storage of resized pixels is reserved for the advanced optimized rectify path.
-	AXI4-Lite is used for configuration and status.
-	AXI4-Stream is used between internal compute modules for the primary fast path.

This modular decomposition is preferred over one large monolithic module because it makes the design easier to test incrementally and easier to verify against a software golden model while still supporting a high-throughput streaming fast path. The resize and rectify stages are mathematically distinct and can be validated separately before being integrated into a full end-to-end hardware image pipeline.

