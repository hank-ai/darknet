#pragma once

#include <cstddef>
#include <cstdint>

/**
 * \file apple_mps.hpp
 * \brief Apple MPS/Metal inference entry points (optional at runtime).
 */

#ifdef DARKNET_USE_MPS

/** \defgroup mps_backend Apple MPS Backend
 *  \brief MPS/Metal inference entry points and helpers.
 *  @{
 */

/**
 * \brief Returns true if MPS is available and initialized.
 */
bool mps_is_available();

/**
 * \brief Try to execute GEMM using MPS. Returns true if MPS handled the op.
 */
bool mps_gemm(int TA, int TB, int M, int N, int K, float ALPHA,
	float *A, int lda,
	float *B, int ldb,
	float BETA,
	float *C, int ldc);

namespace Darknet
{
	struct Layer;
	struct Network;
	struct NetworkState;
}

struct DarknetDetection;

/**
 * \brief Try to execute convolution forward using MPS.
 *
 * \param defer_readback If true, output may remain on GPU for a subsequent MPS layer.
 * \param activation_applied Set to true if MPS applied the activation.
 * \param reason Optional fallback reason string when returning false.
 */
bool mps_convolution_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, bool *activation_applied, const char **reason);

static inline bool mps_convolution_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, const char **reason)
{
	return mps_convolution_forward(l, prev, input, output, false, nullptr, reason);
}

static inline bool mps_convolution_forward(const Darknet::Layer & l, const float *input, float *output, const char **reason)
{
	return mps_convolution_forward(l, nullptr, input, output, false, nullptr, reason);
}

static inline bool mps_convolution_forward(const Darknet::Layer & l, const float *input, float *output)
{
	return mps_convolution_forward(l, nullptr, input, output, false, nullptr, nullptr);
}

bool mps_maxpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason);
bool mps_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason);
/**
 * \brief Try to execute global avgpool forward using MPS.
 */
bool mps_global_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason);
/**
 * \brief Try to execute shortcut add using MPS.
 */
bool mps_shortcut_forward(const Darknet::Layer & l, const Darknet::Layer *prev, const Darknet::Layer *from,
	const float *input, float *output, bool defer_readback, bool *activation_applied, const char **reason);
/**
 * \brief Try to concatenate route inputs using MPS.
 */
bool mps_route_forward(const Darknet::Layer & l, const Darknet::Network & net,
	float *output, bool defer_readback, const char **reason);
/**
 * \brief Try to upsample using a Metal kernel on GPU.
 */
bool mps_upsample_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason);
/**
 * \brief Try to execute reorg using a Metal kernel on GPU.
 */
bool mps_reorg_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason);
/**
 * \brief Try to execute connected layer forward using MPS (batchnorm/activation).
 */
bool mps_connected_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, bool *activation_applied, const char **reason);
/**
 * \brief Try to execute softmax on GPU.
 */
bool mps_softmax_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, const char **reason);

static inline bool mps_maxpool_forward(const Darknet::Layer & l, const float *input, float *output)
{
	return mps_maxpool_forward(l, nullptr, input, output, false, nullptr);
}

static inline bool mps_avgpool_forward(const Darknet::Layer & l, const float *input, float *output)
{
	return mps_avgpool_forward(l, nullptr, input, output, false, nullptr);
}

static inline bool mps_global_avgpool_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, const char **reason)
{
	return mps_global_avgpool_forward(l, prev, input, output, false, reason);
}

static inline bool mps_shortcut_forward(const Darknet::Layer & l, const Darknet::Layer *from,
	const float *input, float *output, const char **reason)
{
	return mps_shortcut_forward(l, nullptr, from, input, output, false, nullptr, reason);
}

static inline bool mps_route_forward(const Darknet::Layer & l, const Darknet::Network & net,
	float *output, const char **reason)
{
	return mps_route_forward(l, net, output, false, reason);
}

static inline bool mps_upsample_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, const char **reason)
{
	return mps_upsample_forward(l, prev, input, output, false, reason);
}

static inline bool mps_reorg_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, const char **reason)
{
	return mps_reorg_forward(l, prev, input, output, false, reason);
}

static inline bool mps_connected_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, const char **reason)
{
	return mps_connected_forward(l, prev, input, output, false, nullptr, reason);
}


bool mps_layer_can_run(const Darknet::Layer & l, bool train);
bool mps_is_output_deferred(const Darknet::Layer *layer);
void mps_flush_output_if_needed(const Darknet::Layer *layer, float *output);
const Darknet::Layer *mps_prev_layer(const Darknet::NetworkState &state);
bool mps_should_defer_readback(const Darknet::NetworkState &state);
void mps_flush_deferred_output(const Darknet::Layer *layer);
namespace Darknet
{
	void show_mps_info();
}

/** @} */
#else

static inline bool mps_is_available()
{
	return false;
}

static inline bool mps_gemm(int, int, int, int, int, float,
	float *, int,
	float *, int,
	float,
	float *, int)
{
	return false;
}

namespace Darknet
{
	struct Layer;
	struct Network;
	struct NetworkState;
}

static inline bool mps_convolution_forward(const Darknet::Layer &, const float *, float *, const char **)
{
	return false;
}

static inline bool mps_convolution_forward(const Darknet::Layer &, const float *, float *)
{
	return false;
}

static inline bool mps_convolution_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, bool, bool *, const char **)
{
	return false;
}

static inline bool mps_maxpool_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, bool, const char **)
{
	return false;
}

static inline bool mps_avgpool_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, bool, const char **)
{
	return false;
}

static inline bool mps_global_avgpool_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, bool, const char **)
{
	return false;
}

static inline bool mps_shortcut_forward(const Darknet::Layer &, const Darknet::Layer *,
	const Darknet::Layer *, const float *, float *, bool, bool *, const char **)
{
	return false;
}

static inline bool mps_route_forward(const Darknet::Layer &, const Darknet::Network &,
	float *, bool, const char **)
{
	return false;
}

static inline bool mps_upsample_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, bool, const char **)
{
	return false;
}

static inline bool mps_reorg_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, bool, const char **)
{
	return false;
}

static inline bool mps_connected_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, bool, bool *, const char **)
{
	return false;
}

static inline bool mps_softmax_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, const char **)
{
	return false;
}

static inline bool mps_maxpool_forward(const Darknet::Layer &, const float *, float *)
{
	return false;
}

static inline bool mps_avgpool_forward(const Darknet::Layer &, const float *, float *)
{
	return false;
}

static inline bool mps_shortcut_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, const char **)
{
	return false;
}

static inline bool mps_layer_can_run(const Darknet::Layer &, bool)
{
	return false;
}

static inline bool mps_is_output_deferred(const Darknet::Layer *)
{
	return false;
}

static inline void mps_flush_output_if_needed(const Darknet::Layer *, float *)
{
	return;
}

static inline const Darknet::Layer *mps_prev_layer(const Darknet::NetworkState &)
{
	return nullptr;
}

static inline bool mps_should_defer_readback(const Darknet::NetworkState &)
{
	return false;
}

static inline void mps_flush_deferred_output(const Darknet::Layer *)
{
	return;
}

namespace Darknet
{
	static inline void show_mps_info()
	{
		return;
	}
}

#endif
