# Notice

> [!CAUTION]
> **This is not the primary "readme" file!**  Please start with [README.md](README.md#Building).  This file contains a description of the various CMake flags.

# CMake

The tool CMake is used to generate the necessary build files.  The output of running CMake is then used to build Darknet/YOLO.

Various flags are exposed via CMake to influence how Darknet/YOLO is built.  This readme will attempt to document all of these flags in a single convenient location.  Many of these flags can be combined together.

## Flags

### NVIDIA GPU

Attempt to discover if the system has a NVIDIA GPU and CUDA installed.  The default is `ON`.

	cmake -DDARKNET_TRY_CUDA=ON ...
	cmake -DDARKNET_TRY_CUDA=OFF ...

The NVIDIA GPU that should be used when compiling CUDA source files.  The default of `native` will cause the GPU to be auto-detected.

	cmake -DDARKNET_CUDA_ARCHITECTURES=native ...

### AMD GPU

Attempt to discover if the system has a AMD GPU and ROCm installed.  The default is `ON`.

	cmake -DDARKNET_TRY_ROCM=ON ...
	cmake -DDARKNET_TRY_ROCM=OFF ...

The AMD GPU that should be used when compiling ROCm source files.  This needs to be set to your AMD GPU architecture, such as `gfx1101`.

	cmake -DCMAKE_HIP_ARCHITECTURES=gfx1101 ...

### Apple GPU (Metal/MPS)

Attempt to discover if the system has Apple MPS available.  The default is `ON` on MacOS.

	cmake -DDARKNET_TRY_MPS=ON ...
	cmake -DDARKNET_TRY_MPS=OFF ...

### Other Optimizations

Some older computers don't have the full SSE and AVX implementation.  This results in a "illegal instruction" trap when running Darknet/YOLO.  (See the [FAQ](https://www.ccoderun.ca/programming/yolo_faq/#illegal_instruction).)  On those older computers, you'll need to disable SSE and AVX.  The default is `ON`.

	cmake -DENABLE_SSE_AND_AVX=ON ...
	cmake -DENABLE_SSE_AND_AVX=OFF ...

With CPU-only builds, this determines if OpenBLAS should be auto-detected.  Systems with GPUs will ignore this setting.  The default is `ON`.

	cmake -DDARKNET_TRY_OPENBLAS=ON ...
	cmake -DDARKNET_TRY_OPENBLAS=OFF ...

Toggle creating profile-guided optimizations.  See [README_PGO.md](README_PGO.md).  The default is `OFF`.

	cmake -DDARKNET_PROFILE_GEN=ON ...
	cmake -DDARKNET_PROFILE_GEN=OFF ...

Toggle using the previously-generated profile-guided optimizations.  See [README_PGO.md](README_PGO.md).  The default is `OFF`.

	cmake -DDARKNET_PROFILE_USE=ON ...
	cmake -DDARKNET_PROFILE_USE=OFF ...

Toggle looking for Protocol Buffers and building the ONNX export tool.  See [README_ONNX.md](README_ONNX.md).  The default in V5.0 was `OFF`.  The default in V5.1 was changed to `ON`.

	cmake -DDARKNET_TRY_ONNX=ON ...
	cmake -DDARKNET_TRY_ONNX=OFF ...

### Debugging

Tell CMake if the build is `Release` or `Debug`.  The default is `Release`.

	cmake -DCMAKE_BUILD_TYPE=Release ...
	cmake -DCMAKE_BUILD_TYPE=Debug ...

Toggle the "timing and tracking" functionality to time how long functions take to run.  This is meant for developers while debugging since it has a huge negative impact on performance.  The default is `OFF`.

	cmake -DENABLE_TIMING_AND_TRACKING=ON ...
	cmake -DENABLE_TIMING_AND_TRACKING=OFF ...

### Linux

By default, CPack (part of CMake) will install to `/usr/bin/` and `/usr/lib/`.  To force Darknet to be installed to a different path, you must set _two_ variables, `CPACK_SET_DESTDIR` and `CMAKE_INSTALL_PREFIX`.

	cmake -DCPACK_SET_DESTDIR=ON -DCMAKE_INSTALL_PREFIX=/usr/local/ ...

(Also see `CPACK_PACKAGING_INSTALL_PREFIX` which may provide an alterate way to install to a different location.)

## Environment Variables

CUDA and cuDNN on Windows can be complicated.  Several environment variables exist to help CMake find where CUDA and cuDNN are installed.

- `CUDNN_LIBRARY_DIR`
- `CUDA_PATH`
- `CUDNN_HOME`
