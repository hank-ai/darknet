#pragma once

#include <cstddef>
#include <cstdint>

#ifdef DARKNET_USE_MPS

bool mps_is_available();

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
bool mps_shortcut_forward(const Darknet::Layer & l, const Darknet::Layer *prev, const Darknet::Layer *from,
	const float *input, float *output, bool defer_readback, bool *activation_applied, const char **reason);
bool mps_route_forward(const Darknet::Layer & l, const Darknet::Network & net,
	float *output, bool defer_readback, const char **reason);
bool mps_upsample_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason);
bool mps_reorg_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, bool defer_readback, const char **reason);
bool mps_softmax_forward(const Darknet::Layer & l, const Darknet::Layer *prev,
	const float *input, float *output, const char **reason);
bool mps_yolo_activate(const Darknet::Layer & l, const float *input, float *output, const char **reason);
bool mps_yolo_decode_boxes(const Darknet::Layer & l, const float *input, int netw, int neth, float *boxes, const char **reason);
bool mps_yolo_collect_candidates(const Darknet::Layer & l, const float *input, float thresh,
	uint32_t *indices, uint32_t max_candidates, uint32_t *count, const char **reason);
bool mps_nms_suppress(const Darknet::Box *boxes, float *scores, const uint32_t *order,
	uint32_t order_count, uint32_t total, float thresh, const char **reason);
bool mps_nms_sort(DarknetDetection *dets, int total, int classes, float thresh, const char **reason);

static inline bool mps_maxpool_forward(const Darknet::Layer & l, const float *input, float *output)
{
	return mps_maxpool_forward(l, nullptr, input, output, false, nullptr);
}

static inline bool mps_avgpool_forward(const Darknet::Layer & l, const float *input, float *output)
{
	return mps_avgpool_forward(l, nullptr, input, output, false, nullptr);
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


bool mps_layer_can_run(const Darknet::Layer & l, bool train);
bool mps_is_output_deferred(const Darknet::Layer *layer);
void mps_flush_output_if_needed(const Darknet::Layer *layer, float *output);
const Darknet::Layer *mps_prev_layer(const Darknet::NetworkState &state);
bool mps_should_defer_readback(const Darknet::NetworkState &state);
void mps_flush_deferred_output(const Darknet::Layer *layer);
bool mps_coverage_enabled();
void mps_coverage_record(const Darknet::Layer & l, bool used_mps);
void mps_coverage_report();

namespace Darknet
{
	void show_mps_info();
}

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

static inline bool mps_softmax_forward(const Darknet::Layer &, const Darknet::Layer *,
	const float *, float *, const char **)
{
	return false;
}

static inline bool mps_yolo_activate(const Darknet::Layer &, const float *, float *, const char **)
{
	return false;
}

static inline bool mps_yolo_decode_boxes(const Darknet::Layer &, const float *, int, int, float *, const char **)
{
	return false;
}

static inline bool mps_yolo_collect_candidates(const Darknet::Layer &, const float *, float,
	uint32_t *, uint32_t, uint32_t *, const char **)
{
	return false;
}

static inline bool mps_nms_sort(DarknetDetection *, int, int, float, const char **)
{
	return false;
}

static inline bool mps_nms_suppress(const Darknet::Box *, float *, const uint32_t *,
	uint32_t, uint32_t, float, const char **)
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

static inline bool mps_coverage_enabled()
{
	return false;
}

static inline void mps_coverage_record(const Darknet::Layer &, bool)
{
	return;
}

static inline void mps_coverage_report()
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
