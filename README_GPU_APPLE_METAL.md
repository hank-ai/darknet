# Apple GPU (Metal/MPS) Inference

This build uses Metal Performance Shaders (MPS) to accelerate GEMM and convolution forward paths on macOS. It does not enable the CUDA/ROCm GPU backend.

## Requirements
- macOS 13+ on Apple Silicon (M1/M2/M3/M4)
- Xcode Command Line Tools
- CMake 3.24+

## Build
```
cmake -S . -B build -DDARKNET_TRY_MPS=ON
cmake --build build
```

## Notes
- MPS is used for GEMM, convolution forward, batchnorm, and max/avg pooling in inference paths.
- Metal kernels are used for route/concat, reorg, upsample, shortcut add, softmax, and elementwise activations (swish/mish/hard-mish + most others).
- Pooling uses MPSCNNPooling* for standard cases; larger padding falls back to a Metal kernel or CPU where needed.
- ReLU/leaky/linear run on GPU; normalize_channels and some edge cases still fall back to CPU.
- Consecutive MPS layers reuse cached MPS outputs as inputs to reduce CPU→GPU copies.
- Grouped convolution is supported when the local MPS API exposes grouped conv; otherwise it falls back to CPU. Dilation falls back to CPU if unsupported by MPS.
- YOLO post-processing can run on GPU with `DARKNET_MPS_POSTPROC=1` (decode + candidate filter + NMS). Use `DARKNET_MPS_POSTPROC=2` or `DARKNET_MPS_POSTPROC=compare` to compare GPU vs CPU NMS.
- MPSGraph is not currently supported in this build.
- Training is not targeted in this milestone.

## Packaging (macOS)
CPack is already configured to produce a DMG on Apple platforms via the `DragNDrop` generator.

```
cmake -S . -B build -DDARKNET_TRY_MPS=ON
cmake --build build
cmake --build build --target package
```

This will emit a `.dmg` in the build directory (default format `UDZO`).
If you prefer to call CPack directly, use the build config file:
`cpack --config build/CPackConfig.cmake -G DragNDrop`.

The macOS package bundles the OpenCV dylibs used at build time. If you update OpenCV via Homebrew, reinstall or rebuild to refresh the bundled libraries.

## Cleaning Install Prefix (Optional)
If you package from `/opt/lib/darknet` and want to wipe `bin/`, `include/`, and `lib/` before each package build, enable:

```
cmake -S . -B build -DDARKNET_TRY_MPS=ON -DDARKNET_CLEAN_INSTALL_PREFIX=ON
```
