# Apple GPU Support Plan (Metal/MPS)

## Purpose
Add Apple GPU acceleration to Darknet on macOS using Metal + Metal Performance Shaders (MPS), with a staged path from minimal inference to full training support.

## Current Implementation (Jan 2026)
- Implemented MPS GEMM (`MPSMatrixMultiplication`) for inference via `src-lib/apple_mps.mm`.
- Implemented MPS convolution forward path (`MPSCNNConvolution`) for inference via `src-lib/apple_mps.mm`.
- Implemented MPS batchnorm (`MPSCNNBatchNormalization`) for conv layers using rolling mean/variance.
- Implemented MPS max/avg pooling forward paths (`MPSCNNPoolingMax`/`MPSCNNPoolingAverage`) with padding support (within kernel size).
- Added GPU ReLU/leaky/linear activation for conv layers and reuse of previous MPS outputs to reduce CPU→GPU copies.
- Added Metal activation kernels for swish/mish/hard-mish (GPU-only).
- Added MPS route concatenation and upsample forward paths to keep data on GPU.
- Added Metal softmax + reorg kernels and wired into the forward paths.
- Added Metal shortcut support for PER_FEATURE/PER_CHANNEL weights (single input, no/relu/softmax normalization).
- Added GPU YOLO post-processing (decode + candidate filter + per-class NMS) via Metal, enabled with `DARKNET_MPS_POSTPROC=1` (compare mode via `=2` or `compare`).
- Added runtime MPS coverage summary (`DARKNET_MPS_COVERAGE=1`) and parity table (`APPLE_GPU_PARITY_MATRIX.md`).
- Supports stride/padding and dilation (when supported by MPS). Grouped conv is supported when the MPS API exposes it.
- Added build flag `DARKNET_TRY_MPS` and compile define `DARKNET_USE_MPS` in `CM_dependencies.cmake`.
- Added MPS device info output in `src-lib/darknet.cpp`.
- Added buffer reuse + thread serialization to avoid MPS memory churn.
- Fixed macOS OpenBLAS include to use `cblas.h` in `src-lib/blas.hpp`.
- Added benchmark doc `APPLE_GPU_BENCHMARKS.md` and overlay support in `src-examples/darknet_05_process_videos_multithreaded.cpp`.

## Current Codebase Notes (GPU Architecture)
- GPU enablement is build-time via CMake in `CM_dependencies.cmake` and `src-lib/CMakeLists.txt`.
- GPU backends are CUDA and ROCm, selected by `DARKNET_USE_CUDA` / `DARKNET_USE_ROCM` and compile defs `DARKNET_GPU`, `DARKNET_GPU_CUDA`, `DARKNET_GPU_ROCM`.
- CUDA/HIP API is used directly throughout GPU code; ROCm works by macro-mapping CUDA symbols in `src-lib/darknet_gpu.hpp`.
- Core GPU runtime helpers live in `src-lib/dark_cuda.hpp` and `src-lib/dark_cuda.cpp` (streams, malloc, memcpy, cublas, cuDNN).
- GPU kernels live in these `.cu` files in `src-lib/`:
  - `activation_kernels.cu`
  - `avgpool_layer_kernels.cu`
  - `blas_kernels.cu`
  - `col2im_kernels.cu`
  - `convolutional_kernels.cu`
  - `crop_layer_kernels.cu`
  - `dropout_layer_kernels.cu`
  - `im2col_kernels.cu`
  - `maxpool_layer_kernels.cu`
  - `network_kernels.cu`
- `src-lib/darknet.cpp` prints CUDA/ROCm info; this will need a Metal path for device reporting.

## Recommended Apple Backend Choice
- Use Metal for custom compute kernels and MPS for high-performance convolution, pooling, and GEMM.
- Current implementation uses MPS for GEMM and convolution (forward only), plus small Metal kernels for activations and upsample.
- Avoid OpenCL (deprecated) and Vulkan/MoltenVK unless Metal proves insufficient.

## Phased Plan

### Phase 0: Scope and Success Criteria
- Decide target scope: inference-only first (recommended), then training.
- Define supported macOS versions and Apple GPU targets (e.g., M1/M2/M3, Intel + AMD eGPU optional).
- Establish correctness/perf baselines vs CPU on a small model (e.g., YOLOv4-tiny).

### Phase 1: Build System + Feature Flags (Partial)
- Added `DARKNET_TRY_MPS` (default ON for APPLE) and `DARKNET_USE_MPS`.
- Detect Metal + MPS via `find_library` (Metal, MetalPerformanceShaders, Foundation).
- Optional future: `DARKNET_TRY_METAL` and `DARKNET_GPU_METAL` for a custom Metal kernel backend.
- Objective-C++ sources added for MPS (`src-lib/apple_mps.mm`).
- Updated `README_GPU_APPLE_METAL.md` and `README_CMake_flags.md`.

### Phase 2: GPU Backend Abstraction Layer
- Introduce a minimal backend interface to reduce CUDA-only assumptions.
  - Example: `gpu_backend.hpp/.cpp` with functions for device selection, stream/queue, memory alloc, memcpy, kernel dispatch, and RNG.
- Implement adapters:
  - CUDA/ROCm: wrappers that call existing `cuda_*` / cuDNN / cublas.
  - Metal: new implementation using `MTLDevice`, `MTLCommandQueue`, `MTLBuffer`, and `MTLComputePipelineState`.
- Keep the existing `DARKNET_GPU` code paths, but route through the backend interface where possible.

### Phase 3: Kernel Porting Strategy (Deferred)
- Port kernels in the following order to unlock inference:
  1. `activation_kernels.cu` (activations + elementwise ops)
  2. `blas_kernels.cu` (axpy, scal, copy, etc.)
  3. `im2col_kernels.cu` + `col2im_kernels.cu`
  4. `maxpool_layer_kernels.cu` / `avgpool_layer_kernels.cu`
  5. `convolutional_kernels.cu` (or replace with MPSConvolution)
  6. `network_kernels.cu` (softmax, batchnorm helpers)
  7. `dropout_layer_kernels.cu` / `crop_layer_kernels.cu`
- Prefer MPS for convolution and matrix multiply:
  - Use `MPSMatrixMultiplication` for GEMM paths in `src-lib/gemm.cpp`.
  - Use `MPSConvolution` / `MPSCNNPooling*` for conv/pooling in `src-lib/convolutional_layer.cpp` and `src-lib/maxpool_layer.cpp`.
- For custom ops (YOLO, reorg, shortcut, route, scale-channels), implement Metal compute kernels.

### Phase 4: Layer Integration and Runtime Hooks (Partial)
- Add `#ifdef DARKNET_GPU_METAL` branches where CUDA/HIP-specific behavior is assumed.
- Ensure device selection replaces `cuda_set_device` with a backend-agnostic entry point (or provide a Metal-backed `cuda_set_device` shim when `DARKNET_GPU_METAL` is defined).
- Add Metal device info reporting in `src-lib/darknet.cpp`, similar to `show_cuda_cudnn_info()` and `show_rocm_info()`.

### Phase 5: Validation and Testing (Partial)
- Add GPU-vs-CPU parity tests for key ops (activation, GEMM, conv, softmax).
- Add a small inference smoke test in `src-test/` with deterministic inputs and tolerance checks.
- Benchmark a known model vs CPU and record speedups + memory usage.

### Phase 6: Documentation and Packaging
- Document build instructions for macOS (Xcode CLI tools, Metal SDK, CMake flags).
- Note limitations (unsupported layers, training status, precision differences).
- Add example commands for building and running with Metal.

## Risks and Unknowns
- Metal API does not map cleanly to CUDA; a thin macro shim is unlikely to be sufficient.
- Full training support requires many kernels and optimizer ops from `src-lib/blas.hpp` and `src-lib/gemm.cpp`.
- MPSGraph is not currently supported; future adoption would need parity validation.
- Shader compilation and caching need to be integrated cleanly into CMake for a smooth developer experience.

## Suggested Initial Milestone
- macOS inference-only build using Metal + MPS for conv/GEMM, with CPU fallback for unsupported layers.
- Validation on a small model with matching outputs to CPU within tolerance.
