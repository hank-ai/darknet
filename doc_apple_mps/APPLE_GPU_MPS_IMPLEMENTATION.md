# Apple MPS Inference Implementation Notes (Maintainer)

## Audience
This document is for the core maintainer who is an experienced C++ developer but does not have access to macOS hardware or MPS knowledge.

## Executive Summary
We added Apple MPS-backed GEMM, convolution, pooling, shortcut add, route concatenation, and upsample forward paths for inference. Swish/Mish/Hard‑Mish activations and upsample run on GPU via small Metal compute kernels compiled at runtime. The change is minimal and isolated: Darknet still uses the CPU code paths, but `gemm_cpu()` now tries an MPS GEMM first (when compiled with `DARKNET_USE_MPS`), and several layer forward functions now try an MPS fast path before falling back to CPU. This provides measurable inference speedups on Apple Silicon without a full Metal kernel port.

Key takeaways:
- MPS is used for GEMM, conv, max/avg pooling, shortcut add, route concat, and upsample forward.
- Swish/Mish/Hard‑Mish activations and upsample run on GPU via embedded Metal compute kernels.
- MPSGraph is not currently supported.
- A minimal Metal backend scaffold exists (`src-lib/metal_backend.hpp`/`.mm`) for future kernel ports.
- CPU fallback remains the default if MPS is unavailable or if layer shapes are unsupported.
- Memory churn in MPS was fixed by reusing `MTLBuffer` objects, per-thread MPS caches, and command buffer pipelining.

## What Was Added
- Build enablement for MPS in `CM_dependencies.cmake` (`DARKNET_TRY_MPS`, `DARKNET_USE_MPS`, MPS frameworks).
- Objective-C++ MPS wrapper (`src-lib/apple_mps.hpp`, `src-lib/apple_mps.mm`).
- GEMM integration in `src-lib/gemm.cpp` (MPS first, CPU fallback).
- Convolution integration in `src-lib/convolutional_layer.cpp` (MPS forward fast path).
- Pooling integration in `src-lib/maxpool_layer.cpp` (MPS max/avg forward fast path).
- Shortcut add integration in `src-lib/shortcut_layer.cpp` (MPS add + optional activation).
- Route concat integration in `src-lib/route_layer.cpp` (MPS blit concat).
- Upsample integration in `src-lib/upsample_layer.cpp` (MPS forward fast path).
- Metal backend wrapper (`src-lib/metal_backend.hpp`/`.mm`) used for Metal activation + upsample kernels.
- Weighted shortcut (PER_FEATURE/PER_CHANNEL, no/relu/softmax normalization, single input) uses Metal elementwise kernels to stay on GPU.
- Optional runtime check via `DARKNET_METAL_SELF_TEST=1` prints a simple backend self-test result at startup.
- Kernel list and argument conventions are tracked in `METAL_KERNEL_REGISTRY.md`.
- Runtime coverage summary is available via `DARKNET_MPS_COVERAGE=1` (prints MPS vs CPU per layer type).
- Metal activation kernels for swish/mish/hard-mish (compiled at runtime).
- Deferred readback in `src-lib/apple_mps.mm` to avoid CPU syncs between MPS-friendly layers.
- MPS device info in `src-lib/darknet.cpp`.
- OpenBLAS include fix on macOS (`src-lib/blas.hpp`).
- Benchmark summary in `APPLE_GPU_BENCHMARKS.md`.

## Code Added (Excerpts)

### CMake Enablement
`CM_dependencies.cmake`:
```cmake
IF (APPLE)
	CMAKE_DEPENDENT_OPTION (DARKNET_TRY_MPS "Attempt to find Apple Metal/MPS support" ON "" ON)
	IF (DARKNET_TRY_MPS)
		FIND_LIBRARY (APPLE_METAL Metal)
		FIND_LIBRARY (APPLE_MPS MetalPerformanceShaders)
		FIND_LIBRARY (APPLE_FOUNDATION Foundation)
		IF (APPLE_METAL AND APPLE_MPS AND APPLE_FOUNDATION)
			MESSAGE (STATUS "Apple Metal/MPS detected. Darknet will use MPS for inference acceleration.")
			SET (DARKNET_USE_MPS ON)
			SET (CMAKE_OBJCXX_STANDARD 17)
			SET (CMAKE_OBJCXX_STANDARD_REQUIRED ON)
			ENABLE_LANGUAGE (OBJCXX)
			ADD_COMPILE_DEFINITIONS (DARKNET_USE_MPS)
			LIST (APPEND DARKNET_LINK_LIBS ${APPLE_METAL} ${APPLE_MPS} ${APPLE_FOUNDATION})
		ELSE ()
			MESSAGE (WARNING "Apple Metal/MPS not found.")
		ENDIF ()
	ELSE ()
		MESSAGE (WARNING "Apple Metal/MPS support is disabled.")
	ENDIF ()
ENDIF ()
```

`src-lib/CMakeLists.txt`:
```cmake
IF (DARKNET_USE_MPS)
	LIST (APPEND LIBSRC apple_mps.mm)
	SET_SOURCE_FILES_PROPERTIES (apple_mps.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
ENDIF ()
```

### Public Interface
`src-lib/apple_mps.hpp`:
```cpp
#ifdef DARKNET_USE_MPS
bool mps_is_available();

bool mps_gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc);

namespace Darknet { struct Layer; struct Network; }

bool mps_convolution_forward(const Darknet::Layer & l, const float *input, float *output, const char **reason);
bool mps_maxpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev, const float *input, float *output, bool defer_readback, const char **reason);
bool mps_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev, const float *input, float *output, bool defer_readback, const char **reason);
bool mps_shortcut_forward(const Darknet::Layer & l, const Darknet::Layer *prev, const Darknet::Layer *from,
	const float *input, float *output, bool defer_readback, bool *activation_applied, const char **reason);
bool mps_route_forward(const Darknet::Layer & l, const Darknet::Network & net,
	float *output, bool defer_readback, const char **reason);
bool mps_upsample_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason);
static inline bool mps_convolution_forward(const Darknet::Layer & l, const float *input, float *output)
{
	return mps_convolution_forward(l, input, output, nullptr);
}

namespace Darknet { void show_mps_info(); }
#endif
```

### MPS GEMM
`src-lib/apple_mps.mm` (excerpt):
```cpp
bool mps_gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc)
{
	std::scoped_lock<std::mutex> lock(get_mps_gemm_mutex());

	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		return false;
	}

	// ... create/reuse buffers, encode MPSMatrixMultiplication ...
	[command_buffer commit];
	[command_buffer waitUntilCompleted];

	std::memcpy(C, [ctx.bufferC contents], c_bytes);
	return true;
}
```

### MPS Convolution Forward
`src-lib/apple_mps.mm` (excerpt):
```cpp
bool mps_convolution_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool *activation_applied, const char **reason)
{
	auto & ctx = get_mps_context();
	if (!ctx.ready)
	{
		if (reason) *reason = "MPS not available";
		return false;
	}

	if (l.groups <= 0 || (l.c % l.groups) != 0 || (l.n % l.groups) != 0)
	{
		if (reason) *reason = "invalid groups";
		return false;
	}

	if (l.groups > 1 && !mps_supports_grouped_conv())
	{
		if (reason) *reason = "grouped convolution not supported";
		return false;
	}

	if (l.pad < 0 || l.pad > (l.size / 2))
	{
		if (reason) *reason = "unsupported padding";
		return false;
	}

	// Build or reuse cached MPSCNNConvolution, batchnorm, neuron (ReLU/leaky), and packed weights
	if (!cache_matches_layer(cache, l))
	{
		if (!build_conv_cache(cache, l, ctx.device))
		{
			if (reason) *reason = (l.dilation != 1) ? "dilation not supported" : "failed to build convolution";
			return false;
		}
	}

	// MPSImage upload (NCHW), encode, optional batchnorm + neuron, and read back
	[cache.conv encodeToCommandBuffer:command_buffer sourceImage:input_image destinationImage:output_image];
	if (cache.batchnorm)
	{
		[cache.batchnorm encodeToCommandBuffer:command_buffer sourceImage:output_image destinationImage:output_image];
	}
	if (cache.neuron)
	{
		[cache.neuron encodeToCommandBuffer:command_buffer sourceImage:output_image destinationImage:output_image];
	}
	[command_buffer commit];
	[command_buffer waitUntilCompleted];

	if (reason) *reason = "ok";
	return true;
}
```

### MPS Pooling (Max/Avg)
`src-lib/apple_mps.mm` (excerpt):
```cpp
bool mps_maxpool_forward(const Darknet::Layer & l, const float *input, float *output, const char **reason)
{
	if (l.maxpool_depth || l.antialiasing)
	{
		if (reason) *reason = "unsupported maxpool mode";
		return false;
	}
	if (l.pad < 0 || l.pad > l.size)
	{
		if (reason) *reason = "unsupported padding";
		return false;
	}
	// build/reuse MPSCNNPoolingMax and cached MPSImage buffers
	[cache.pool encodeToCommandBuffer:command_buffer sourceImage:input_image destinationImage:output_image];
	[command_buffer commit];
	[command_buffer waitUntilCompleted];
	return true;
}
```

`MPSCNNPoolingAverage` is used for local avgpool with the same padding restriction.

### MPS Upsample (Metal Kernel)
`src-lib/apple_mps.mm` (excerpt):
```cpp
if (!encode_mps_upsample(command_buffer, input_image, cache.output_image, l.stride, l.scale, reason))
{
	return false;
}
```
The upsample kernel is a small Metal compute shader (`upsample_kernel`) that does nearest‑neighbor expansion and optional scaling.

### Convolution Fast Path Hook
`src-lib/convolutional_layer.cpp` (excerpt):
```cpp
#ifdef DARKNET_USE_MPS
if (!state.train && !l.binary && !l.xnor)
{
	const char *reason = nullptr;
	if (mps_convolution_forward(l, state.input, l.output, &reason))
	{
		log_mps_conv_once(l, true, reason);
		// CPU-side batchnorm/bias/activation
		return;
	}
	log_mps_conv_once(l, false, reason);
}
else
{
	log_mps_conv_once(l, false, "training/binary/xnor");
}
#endif
```

### Pooling Fast Path Hooks
`src-lib/maxpool_layer.cpp` (excerpt):
```cpp
#ifdef DARKNET_USE_MPS
if (!state.train && !l.maxpool_depth && !l.antialiasing)
{
	if (mps_maxpool_forward(l, state.input, l.output))
	{
		return;
	}
}
#endif
```

### Device Reporting Hook
`src-lib/darknet.cpp`:
```cpp
#elif defined(DARKNET_USE_MPS)
	Darknet::show_mps_info();
```

### OpenBLAS Include Fix (macOS)
`src-lib/blas.hpp`:
```cpp
#ifdef DARKNET_USE_OPENBLAS
	#ifdef WIN32
		#include <openblas/cblas.h>
	#elif defined(__APPLE__)
		#include <cblas.h>
	#else
		#include <cblas-openblas64.h>
	#endif
#endif
```

## Functionality Summary
- **Build time:** On Apple, CMake probes for Metal/MPS and enables ObjC++ with `DARKNET_USE_MPS` if found.
- **Runtime startup:** `Darknet::show_mps_info()` prints the device name and recommended working set size.
- **Inference hot path (GEMM):** `gemm_cpu()` calls `mps_gemm()`. If it returns `true`, MPS has performed the multiplication and CPU work is skipped.
- **Inference hot path (Conv):** `forward_convolutional_layer()` calls `mps_convolution_forward()` for compatible layers. Bias is applied by MPS when batchnorm is disabled; batchnorm is applied on MPS for batchnorm layers. ReLU/leaky/linear use `MPSCNNNeuron`, and swish/mish/hard‑mish use embedded Metal kernels.
- **Inference hot path (Pooling):** `forward_maxpool_layer()` and `forward_local_avgpool_layer()` call MPS pooling when the layer is compatible (including padding within kernel size). Input upload is skipped when the previous layer already produced an MPS image of matching shape.
- **Inference hot path (Shortcut):** `forward_shortcut_layer()` can use `MPSCNNAdd` when the shortcut is a simple 1-input add (no weights), then apply ReLU/leaky on MPS or swish/mish/hard‑mish via Metal kernels when possible.
- **Inference hot path (Route):** `forward_route_layer()` can use MPS to concatenate feature maps by blitting MPSImage texture slices into a single output image.
- **Inference hot path (Upsample):** `forward_upsample_layer()` can use a Metal kernel for nearest‑neighbor upsample and keep data on GPU.
- **Memory behavior:** MPS input staging images are now allocated lazily (only when a layer cannot reuse the previous MPS output), reducing GPU memory pressure for large models.
- **Deferred readback:** MPS layers can skip CPU readback when the next layer can also run on MPS; CPU readback happens only when a CPU-only layer is encountered or output is needed.

## Current Limitations
- Only swish/mish/hard‑mish were added via Metal kernels; other activations still run on CPU.
- Grouped convolution requires MPS APIs that expose `groups`; otherwise it falls back to CPU.
- Dilation is supported only if the underlying MPS descriptor supports it; otherwise it falls back to CPU.
- Route concat is limited to `groups=1`, `group_id=0`, and output channels aligned to 4 (MPSImage slice layout).
- Shortcut add is only supported for the simple case (1 input, no weights, matching shapes).
- Upsample supports forward‑only nearest‑neighbor (no reverse/downsample path).
- MPSGraph is not currently supported.

## Build and Usage
On macOS:
```
cmake -S . -B build -DDARKNET_TRY_MPS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
cmake --install build --config Release --prefix /Users/denizz/darknet
```

No Apple machine is required to review this code; it is guarded by `#ifdef DARKNET_USE_MPS` and does not affect non-Apple builds.

## Suggested Next Steps (If You Continue)
1. Add a small CPU vs MPS parity test in `src-test/`.
2. Expand Metal kernel coverage for remaining CPU fallbacks.

## Notes for Non-Apple Maintainers
- You can review the ObjC++ code as plain C++ with ObjC syntax.
- Keep changes wrapped in `#ifdef DARKNET_USE_MPS` to avoid impacting non-Apple builds.
- If needed, you can stub the MPS functions on non-Apple platforms without changing public headers.
