# Apple GPU Support Plan (Inference-First)

## Goal
Deliver macOS inference acceleration using Apple GPUs (Metal + MPS) with a CPU fallback for unsupported layers. Training is explicitly out of scope for the initial milestone.

## Scope and Success Criteria
- Scope: inference only (single image + batch), CPU fallback for unsupported ops.
- Platforms: macOS 13+ on Apple Silicon (M1/M2/M3/M4). Intel + AMD eGPU is optional and not required.
- Success: run a small model (e.g., YOLOv4-tiny) end-to-end on GPU with outputs matching CPU within tolerance and measurable speedup.

## Status (Jan 2026)
Completed:
- Added `DARKNET_TRY_MPS` CMake option and `DARKNET_USE_MPS` compile definition.
- Added Objective-C++ MPS wrapper: `src-lib/apple_mps.hpp` + `src-lib/apple_mps.mm`.
- Added MPS device info output in `src-lib/darknet.cpp`.
- Added MPS GEMM in `src-lib/gemm.cpp` (`mps_gemm()` is attempted before CPU GEMM).
- Added MPS convolution forward path (inference-only) in `src-lib/convolutional_layer.cpp`.
- Added MPS pooling forward paths (max/avg) in `src-lib/maxpool_layer.cpp` with padding support (within kernel size).
- Added MPS batchnorm for conv layers plus ReLU/leaky/linear activations and cached MPS inputs for consecutive MPS layers.
- Added deferred GPU readback with per-thread MPS caches and CPU flush on demand.
- Added command buffer pipelining (commit per layer, wait only on CPU readback).
- Added MPS shortcut add for the simple case (1 input, no weights, matching shapes).
- Added Metal shortcut support for PER_FEATURE/PER_CHANNEL weights (single input, no/relu/softmax normalization) to avoid CPU fallback.
- Added MPS route concatenation with grouped-route support (4-channel aligned slices per group).
- Added MPS upsample forward path (nearest neighbor) via Metal kernel to avoid CPU fallback.
- Added Metal activation kernels for swish/mish/hard-mish to avoid CPU readbacks.
- Added Metal softmax and reorg kernels; wired into the forward paths.
- Audited `yolov4-tiny.cfg`: route + upsample are the only relevant non-conv ops (no reorg/softmax).
- Reduced MPS memory pressure by allocating per-layer input staging images only when needed.
- Added buffer reuse and serialization to avoid MPS memory churn.
- Added macOS build notes in `README_GPU_APPLE_METAL.md`.
- Added benchmarks in `APPLE_GPU_BENCHMARKS.md`.
- Added benchmark run instructions in `benchmarks.md`.
- Fixed macOS OpenBLAS include to use `cblas.h` in `src-lib/blas.hpp`.
- Fixed per-frame image leak in `src-examples/darknet_05_process_videos_multithreaded.cpp`.
- Added benchmark output naming + FPS overlay in `src-examples/darknet_05_process_videos_multithreaded.cpp`.
- Added Metal backend scaffolding (`src-lib/metal_backend.hpp`/`.mm`) with kernel dispatch and a basic self-test hook.
- Added a Metal kernel registry doc (`METAL_KERNEL_REGISTRY.md`) to keep kernel names and arguments in sync.
- Added runtime MPS coverage summary (`DARKNET_MPS_COVERAGE=1`) and parity table (`APPLE_GPU_PARITY_MATRIX.md`).
- Added GPU YOLO post-processing (decode + candidate filter + per-class NMS) via Metal, controlled by `DARKNET_MPS_POSTPROC` (compare mode via `=2` or `compare`).

Deferred:
- Metal backend exists and is used for several inference ops; broader kernel coverage is still pending.
- MPSGraph support is currently not available.
- No training support.

## Phase 1: Build System and Flags (Completed)
- Added `DARKNET_TRY_MPS` (default ON on Apple) and `DARKNET_USE_MPS` compile def.
- Linked against `Metal`, `MetalPerformanceShaders`, and `Foundation` on macOS.
- Added Objective-C++ sources to `src-lib/CMakeLists.txt`.

## Phase 2: Minimal Metal Backend Layer (In progress)
Goal: introduce a small, standalone Metal runtime layer so future GPU work is not tied to CUDA-only assumptions.

Proposed interface (minimal surface area):
- Device/context lifecycle (`metal_is_available`, `metal_init`, `metal_shutdown`).
- Per-thread execution context (`metal_begin_frame`, `metal_end_frame`, command buffer management).
- Buffer management (`metal_buffer_alloc`, `metal_buffer_free`, `metal_buffer_upload`, `metal_buffer_download`, `metal_buffer_fill`).
- Kernel dispatch wrapper (`metal_dispatch_1d/2d`, pipeline cache by name, argument binding).
- Optional helper for MTLTexture creation if needed later for image-based ops.

Implementation outline (partially completed):
1. **New files**
   - `src-lib/metal_backend.hpp` (C++ header with the minimal API).
   - `src-lib/metal_backend.mm` (ObjC++ implementation).
   - Optionally `src-lib/metal_kernels.metal` for built-in kernels (scale, swish, mish, hard-mish, upsample).
2. **CMake integration**
   - Add `metal_backend.mm` (and `.metal` if used) under `DARKNET_USE_MPS`.
   - Compile ObjC++ with ARC; keep non-Apple builds untouched (stub inline functions in header).
3. **Context + caching**
   - Use `MTLDevice` + `MTLCommandQueue`.
   - Cache `MTLComputePipelineState` by kernel name for dispatch.
   - Cache `MTLLibrary` (from embedded source or precompiled metallib).
4. **Command buffer policy**
   - Per-thread command buffer with explicit `metal_begin_frame` / `metal_end_frame`.
   - Expose a `metal_flush()` for CPU readback synchronization.
5. **Error handling**
   - Centralized error logging with clear fallbacks to CPU.
   - Avoid throwing exceptions (match current style).

Integration plan (in progress):
- Ported several Metal kernels to use the backend wrapper; keep MPS-based GEMM/conv/pool as-is.
- Continue migrating remaining custom kernels out of `apple_mps.mm` to centralize dispatch/caching.
- Ensure no symbol/behavior changes on CUDA/ROCm builds.

Validation:
- Add a small runtime self-check (optional) to validate dispatch + readback.
- Compare GPU vs CPU outputs for a trivial kernel (scale or add) in debug builds.

## Phase 3: Core Kernel Coverage for Inference (In progress)
Prioritize kernels required for forward pass only:
1. Elementwise activations and basic math ops (`activation_kernels.cu`, `blas_kernels.cu`).
2. `im2col` / `col2im` (if using custom GEMM path).
3. Pooling (`maxpool_layer_kernels.cu`, `avgpool_layer_kernels.cu`).
4. Softmax + miscellaneous network ops (`network_kernels.cu`).
5. Layer glue kernels (reorg/route/shortcut/upsample) as needed.

## Phase 4: Use MPS Where Possible (Partially Completed)
- GEMM: `MPSMatrixMultiplication` is now used in `gemm.cpp` via `mps_gemm()`.
- Convolution: forward path uses `MPSCNNConvolution` for common inference cases.
- Pooling: forward path uses `MPSCNNPoolingMax`/`MPSCNNPoolingAverage` with padding support (within kernel size).
- Supported: stride, padding, dilation (when MPS supports it), and grouped conv when the MPS API exposes it.
- Batchnorm uses `MPSCNNBatchNormalization` for conv layers.
- Upsample: forward path uses a Metal compute kernel (nearest neighbor) to keep data on GPU.

## Phase 4b: Push Beyond ~2x Speedup (Next Focus)
Goal: reduce CPU time and CPU↔GPU copies so that overall inference speed exceeds 2x vs CPU OpenBLAS on real models.

Immediate steps (ranked):
1. **MPS Convolution Forward** (Completed)
   - Add an optional forward path in `src-lib/convolutional_layer.cpp` using `MPSConvolution`.
   - Keep CPU fallback for unsupported shapes.
   - Map Darknet weights to MPS layout; validate NCHW↔MPS tensor layout.
2. **MPS Pooling + Activation** (Completed)
   - Use `MPSCNNPoolingMax/Avg` and `MPSCNNNeuronReLU/Linear` for common layers.
   - Keep CPU fallback for non-covered cases.
3. **Reduce CPU↔GPU Copies** (Partially completed)
   - Persist MTLBuffers for activations and weights across layers.
   - Only copy back to CPU at the end (or when a CPU-only layer is hit).
   - Add a small GPU cache for temporary tensors to avoid per-layer allocations.
4. **Batching + Command Buffer Reuse** (Completed, wait-on-demand)
   - Use a single command buffer per frame (or per batch) to reduce overhead.
   - Avoid `waitUntilCompleted` per layer; synchronize once per frame.

Stretch targets:
5. **Metal Kernels for Custom Ops**
   - Implement Metal compute kernels for reorg/route/shortcut layers.
   - Remove CPU fallback for those ops when possible.

## Phase 5: Integration Points (Partial)
- MPS device info added to `src-lib/darknet.cpp`.
- No Metal backend integration in CUDA/HIP paths yet.

## Phase 6: Validation and Tests (Partial)
- Benchmarks recorded in `APPLE_GPU_BENCHMARKS.md`.
- No automated parity tests yet.
- Future benchmark runs should compare MPS vs CPU OpenBLAS only.

## Performance Gate for Next Milestone
- Achieve **>2.5x** on `yolov4-tiny.cfg` and **>3x** on `LegoGears.cfg` vs CPU OpenBLAS for both `video1.MOV` and `video2.MOV`.
- Track CPU time spent in non-GEMM layers and the number of CPU↔GPU copies per frame.

## Deliverable Milestone (Achieved for MPS GEMM)
- Build on macOS with MPS enabled.
- Inference-only run succeeds on Apple Silicon with CPU fallback for unsupported ops.
- Documented build + run steps and known limitations.
