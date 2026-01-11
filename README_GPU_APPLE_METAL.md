# Apple GPU (Metal Performance Shaders) Inference

This build uses Metal Performance Shaders (MPS) to accelerate GEMM and convolution forward paths on MacOS. It does not enable the CUDA/ROCm GPU backend.

## Requirements
- MacOS 13+ (Apple Silicon recommended; Intel Macs may work depending on GPU, but are untested)
- Xcode Command Line Tools
- CMake 3.24+

## Build
```
mkdir build
cd build
cmake -DDARKNET_TRY_MPS=ON ..
make -j4
```

## Notes
- MPS is used for GEMM, convolution forward, batchnorm, and max/avg pooling in inference paths.
- Metal kernels are used for route/concat, reorg, upsample, shortcut add, softmax, and elementwise activations (swish/mish/hard-mish + most others).
- Pooling uses MPSCNNPooling* for standard cases; larger padding falls back to a Metal kernel or CPU where needed.
- ReLU/leaky/linear run on GPU; normalize_channels and some edge cases still fall back to CPU.
- Consecutive MPS layers reuse cached MPS outputs as inputs to reduce CPU→GPU copies.
- Grouped convolution is supported when the local MPS API exposes grouped conv; otherwise it falls back to CPU. Dilation falls back to CPU if unsupported by MPS.
- MPSGraph is not currently supported in this build.
- Training is not targeted in this milestone.

## Packaging (macOS)
CPack is already configured to produce a DMG on Apple platforms via the `DragNDrop` generator.

```
mkdir build
cd build
cmake -DDARKNET_TRY_MPS=ON ..
make -j4 package
```

This will emit a `.dmg` in the build directory (default format `UDZO`).
If you prefer to call CPack directly, use the build config file:
`cpack --config build/CPackConfig.cmake -G DragNDrop`.

The MacOS package bundles the OpenCV dylibs used at build time. If you update OpenCV via Homebrew, reinstall or rebuild to refresh the bundled libraries.
