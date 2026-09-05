#include "darknet_internal.hpp"
#include "apple_training_internal.hpp"

#include "apple_mps.hpp"
#include "darknet_layers.hpp"
#include "darknet_network.hpp"
#include "metal_backend.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace Darknet::Apple::TrainingDetail
{
	static size_t bytes_for_float_count(const int count)
	{
		return sizeof(float) * static_cast<size_t>(std::max(0, count));
	}

	static bool is_detector_head_layer_type(const ELayerType type)
	{
		return type == ELayerType::YOLO;
	}

		bool gpu_yolo_loss_supported_by_current_contract(const Network & net, std::string & reason)
		{
			if (net.adversarial)
			{
				reason = "adversarial YOLO loss is not in the Apple GPU-loss scope";
				return false;
			}
			if (net.badlabels_rejection_percentage ||
				net.num_sigmas_reject_badlabels ||
				net.equidistant_point)
			{
				reason = "bad-label rejection and equidistant-point filtering are CPU YOLO loss features";
				return false;
			}

			for (int i = 0; i < net.n; ++i)
			{
				const Layer & layer = net.layers[i];
				if (!is_detector_head_layer_type(layer.type) || !training_is_terminal_detector_head(net, i))
				{
					continue;
				}
				if (layer.type != Darknet::ELayerType::YOLO)
				{
					reason = "Apple GPU YOLO loss is limited to plain YOLO heads";
					return false;
				}
				if (layer.onlyforward)
				{
					reason = "onlyforward YOLO heads do not run the CUDA/CPU training loss";
					return false;
				}
				if (layer.iou_thresh < 1.0f && layer.iou_thresh_kind != IOU)
				{
					reason = "Apple GPU YOLO loss supports iou_thresh<1 only with iou_thresh_kind=iou";
					return false;
				}
				if (std::fabs(layer.truth_thresh - 1.0f) > 1e-6f)
				{
					reason = "truth_thresh values other than 1 are not in the Apple GPU YOLO loss scope";
					return false;
				}
				if (layer.focal_loss)
				{
					reason = "focal_loss is not in the Apple GPU YOLO loss scope";
					return false;
				}
				if (layer.label_smooth_eps != 0.0f)
				{
					reason = "label_smooth_eps is not in the Apple GPU YOLO loss scope";
					return false;
				}
				if (layer.classes_multipliers)
				{
					reason = "classes_multipliers are not in the Apple GPU YOLO loss scope";
					return false;
				}
				if (layer.map)
				{
					reason = "class remapping is not in the Apple GPU YOLO loss scope";
					return false;
				}
				if (layer.iou_loss != CIOU)
				{
					reason = "the first Apple GPU YOLO loss scope is limited to ciou";
					return false;
				}
			}

			reason = "ok";
			return true;
		}

		bool host_parameter_consumers_supported_by_current_contract(const Network & net, std::string & reason)
		{
			if (net.ema_alpha != 0.0f)
			{
				reason = "ema_alpha is not supported by Apple MPSGraph training until parameter sync is implemented";
				return false;
			}
			if (net.weights_reject_freq != 0)
			{
				reason = "weights_reject_freq is not supported by Apple MPSGraph training until parameter sync is implemented";
				return false;
			}
			reason = "ok";
			return true;
		}

		bool validate_gpu_yolo_truth_buffer(const Network & net, const float *truth, std::string & reason)
		{
			if (!truth)
			{
				reason = "Apple GPU YOLO loss requires a truth buffer";
				return false;
			}

			for (int layer_index = 0; layer_index < net.n; ++layer_index)
			{
				const Layer & layer = net.layers[layer_index];
				if (!is_detector_head_layer_type(layer.type) || !training_is_terminal_detector_head(net, layer_index))
				{
					continue;
				}
				if (layer.truth_size <= 4 || layer.truths <= 0 || layer.max_boxes <= 0 || layer.batch <= 0)
				{
					std::ostringstream oss;
					oss << "YOLO layer #" << layer_index << " has invalid truth geometry";
					reason = oss.str();
					return false;
				}

				for (int b = 0; b < layer.batch; ++b)
				{
					for (int t = 0; t < layer.max_boxes; ++t)
					{
						const int truth_index = b * layer.truths + t * layer.truth_size;
						const float x = truth[truth_index + 0];
						if (!x)
						{
							break;
						}
						const float y = truth[truth_index + 1];
						const float w = truth[truth_index + 2];
						const float h = truth[truth_index + 3];
						const float raw_class_id = truth[truth_index + 4];
						if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(w) ||
							!std::isfinite(h) || !std::isfinite(raw_class_id))
						{
							reason = "non-finite YOLO truth value at batch " + std::to_string(b) +
								", truth " + std::to_string(t) + ", layer #" + std::to_string(layer_index);
							return false;
						}
						const int class_id = static_cast<int>(raw_class_id);
						if (class_id >= layer.classes || class_id < 0)
						{
							std::ostringstream oss;
							oss << "invalid class ID #" << class_id
								<< " for Apple GPU YOLO loss at batch " << b
								<< ", truth " << t
								<< ", layer #" << layer_index;
							reason = oss.str();
							return false;
						}
						if (x < 0.0f || y < 0.0f || x >= 1.0f || y >= 1.0f || w <= 0.0f || h <= 0.0f)
						{
							std::ostringstream oss;
							oss << "invalid coordinates, width, or height for Apple GPU YOLO loss"
								<< " at batch " << b
								<< ", truth " << t
								<< ", layer #" << layer_index
								<< " (x=" << x << ", y=" << y << ", w=" << w << ", h=" << h << ")";
							reason = oss.str();
							return false;
						}
					}
				}
			}

			reason = "ok";
			return true;
		}

		struct AppleYoloPlainLossParams
		{
			uint32_t batch = 0;
			uint32_t w = 0;
			uint32_t h = 0;
			uint32_t n = 0;
			uint32_t total = 0;
			uint32_t classes = 0;
			uint32_t outputs = 0;
			uint32_t truths = 0;
			uint32_t truth_size = 0;
			uint32_t max_boxes = 0;
			uint32_t net_w = 0;
			uint32_t net_h = 0;
			uint32_t new_coords = 0;
			uint32_t objectness_smooth = 0;
			float scale_x_y = 1.0f;
			float ignore_thresh = 0.0f;
			float truth_thresh = 1.0f;
			float iou_thresh = 1.0f;
			float obj_normalizer = 1.0f;
			float cls_normalizer = 1.0f;
			float iou_normalizer = 1.0f;
			float max_delta = std::numeric_limits<float>::max();
		};

		struct AppleYoloPlainLossStats
		{
			float total_sq = 0.0f;
			float class_sq = 0.0f;
			float iou_loss = 0.0f;
			float iou_sum = 0.0f;
			float count = 0.0f;
			float rewritten = 0.0f;
		};

		// These PODs cross the Objective-C++/Metal ABI. Keep their layout in sync
		// with apple_yolo_plain_params/apple_yolo_plain_loss_stats in metal_backend.mm.
		static_assert(std::is_standard_layout_v<AppleYoloPlainLossParams>);
		static_assert(sizeof(AppleYoloPlainLossParams) == 88);
		static_assert(offsetof(AppleYoloPlainLossParams, scale_x_y) == 56);
		static_assert(std::is_standard_layout_v<AppleYoloPlainLossStats>);
		static_assert(sizeof(AppleYoloPlainLossStats) == 24);

		struct AppleYoloPlainLossReadback
		{
			Layer *layer = nullptr;
			size_t stats_offset = 0;
			size_t stats_bytes = 0;
		};

		struct AppleYoloPlainLossReadbacks
		{
			// One cached shared stats buffer covers all YOLO heads in the iteration.
			// The stats are tiny, but this avoids one temporary Metal buffer and
			// one host read per detector head.
			MetalBuffer stats_buffer{};
			size_t stats_bytes = 0;
			size_t next_stats_offset = 0;
			std::vector<AppleYoloPlainLossReadback> entries;
		};

		struct AppleYoloPlainLossStatsBufferEntry
		{
			const Network *network = nullptr;
			const void *details_identity = nullptr;
			const Layer *layers_identity = nullptr;
			MetalBuffer buffer{};
			size_t bytes = 0;
		};

		size_t gpu_yolo_loss_stats_bytes_for_layer(const Layer & layer)
		{
			return sizeof(AppleYoloPlainLossStats) * static_cast<size_t>(std::max(0, layer.batch));
		}

		void release_gpu_yolo_loss_readbacks(AppleYoloPlainLossReadbacks & readbacks)
		{
			readbacks.stats_buffer = {};
			readbacks.stats_bytes = 0;
			readbacks.next_stats_offset = 0;
			readbacks.entries.clear();
		}

		AppleYoloPlainLossStatsBufferEntry & gpu_yolo_loss_stats_buffer_cache()
		{
			static AppleYoloPlainLossStatsBufferEntry cache;
			return cache;
		}

		std::mutex & gpu_yolo_loss_stats_buffer_mutex()
		{
			static std::mutex mutex;
			return mutex;
		}

		void reset_gpu_yolo_loss_stats_buffer_cache(const Network * net)
		{
			std::scoped_lock lock(gpu_yolo_loss_stats_buffer_mutex());
			auto & cache = gpu_yolo_loss_stats_buffer_cache();
			if (net && cache.network != net)
			{
				return;
			}
			metal_buffer_free(&cache.buffer);
			cache = {};
		}

		bool cached_gpu_yolo_loss_stats_buffer(const Network & net,
			const size_t bytes,
			MetalBuffer *buffer,
			std::string & reason)
		{
			if (!buffer || bytes == 0)
			{
				reason = "GPU YOLO loss stats buffer request is invalid";
				return false;
			}

			std::scoped_lock lock(gpu_yolo_loss_stats_buffer_mutex());
			auto & entry = gpu_yolo_loss_stats_buffer_cache();
			const bool identity_matches = entry.network == &net &&
				entry.details_identity == net.details && entry.layers_identity == net.layers;
			if (entry.network && !identity_matches)
			{
				metal_buffer_free(&entry.buffer);
				entry = {};
			}
			entry.network = &net;
			entry.details_identity = net.details;
			entry.layers_identity = net.layers;
			if (!entry.buffer.handle || entry.bytes < bytes)
			{
				metal_buffer_free(&entry.buffer);
				if (!metal_buffer_alloc(bytes, &entry.buffer))
				{
					entry.bytes = 0;
					reason = "GPU YOLO loss failed to allocate stats buffer";
					return false;
				}
				entry.bytes = bytes;
			}

			*buffer = entry.buffer;
			return true;
		}

		bool prepare_gpu_yolo_loss_readbacks(Network & net,
			AppleYoloPlainLossReadbacks & readbacks,
			std::string & reason)
		{
			release_gpu_yolo_loss_readbacks(readbacks);

			size_t total_stats_bytes = 0;
			int head_count = 0;
			for (int i = 0; i < net.n; ++i)
			{
				const Layer & layer = net.layers[i];
				if (!is_detector_head_layer_type(layer.type) || !training_is_terminal_detector_head(net, i))
				{
					continue;
				}

				const size_t stats_bytes = gpu_yolo_loss_stats_bytes_for_layer(layer);
				if (stats_bytes == 0)
				{
					reason = "GPU YOLO loss stats buffer size is zero";
					return false;
				}
				if (stats_bytes > std::numeric_limits<size_t>::max() - total_stats_bytes)
				{
					reason = "GPU YOLO loss stats buffer is too large";
					return false;
				}
				total_stats_bytes += stats_bytes;
				++head_count;
			}

			if (head_count == 0)
			{
				reason = "no terminal YOLO heads found";
				return false;
			}
			if (!cached_gpu_yolo_loss_stats_buffer(net, total_stats_bytes, &readbacks.stats_buffer, reason))
			{
				return false;
			}
			if (!metal_buffer_fill(&readbacks.stats_buffer, 0, 0, total_stats_bytes))
			{
				reason = "GPU YOLO loss failed to clear stats buffer";
				return false;
			}

			readbacks.stats_bytes = total_stats_bytes;
			readbacks.next_stats_offset = 0;
			readbacks.entries.reserve(static_cast<size_t>(head_count));
			return true;
		}

		MetalBuffer metal_buffer_from_training_view(const TrainingBufferView & view)
		{
			return {view.buffer_handle, view.offset + view.bytes};
		}

		bool dispatch_clear_training_layer_delta_with_encoder(Layer & layer, void *encoder_handle, std::string & reason)
		{
			if (!layer.delta)
			{
				return true;
			}
			const int item_count = layer.outputs * layer.batch;
			if (item_count <= 0)
			{
				return true;
			}

			TrainingBufferView delta_view{};
			const size_t delta_bytes = bytes_for_float_count(item_count);
			if (!load_required_training_view(layer.delta,
				delta_bytes,
				TrainingBufferAccess::kWrite,
				"layer delta",
				delta_view,
				reason))
			{
				return false;
			}

			const MetalBuffer delta_buffer = metal_buffer_from_training_view(delta_view);
			const float zero = 0.0f;
			const uint32_t total = static_cast<uint32_t>(item_count);
			const MetalDispatchBuffer buffers[] = {
				{&delta_buffer, delta_view.offset},
			};
			const MetalDispatchBytes bytes[] = {
				{&zero, sizeof(zero)},
				{&total, sizeof(total)},
			};
			if (!dispatch_training_kernel_1d_with_encoder("buffer_fill_float",
				static_cast<size_t>(item_count),
				buffers, 1,
				bytes, 2,
				encoder_handle,
				reason))
			{
				return false;
			}
			note_training_shared_buffer_updated_from_gpu(layer.delta);
			return true;
		}

		bool dispatch_clear_yolo_boundary_deltas_with_encoder(Network & net, void *encoder_handle, std::string & reason)
		{
			std::unordered_set<int> cleared_layers;
			for (int i = 0; i < net.n; ++i)
			{
				Layer & layer = net.layers[i];
				if (!is_detector_head_layer_type(layer.type) || !training_is_terminal_detector_head(net, i))
				{
					continue;
				}
				if (i <= 0)
				{
					reason = "GPU YOLO loss requires a previous layer delta buffer";
					return false;
				}
				// The terminal YOLO head receives its input from the immediately
				// preceding layer, so the backward graph only needs that boundary
				// delta buffer zeroed before the GPU YOLO loss writes into it.
				const int boundary_layer_index = i - 1;
				if (!cleared_layers.insert(boundary_layer_index).second)
				{
					continue;
				}
				if (!dispatch_clear_training_layer_delta_with_encoder(net.layers[boundary_layer_index], encoder_handle, reason))
				{
					if (reason.empty())
					{
						reason = "failed to clear Apple YOLO boundary delta buffers";
					}
					return false;
				}
			}
			return true;
		}

		const MPSGraphTensorPlan *find_forward_result_plan(const MPSGraphExecutableBundle & bundle,
			const int layer_index,
			const MPSGraphTensorBindingKind binding_kind)
		{
			for (const auto & tensor_plan : bundle.result_plans())
			{
				if (tensor_plan.owner_layer_index == layer_index && tensor_plan.binding_kind == binding_kind)
				{
					return &tensor_plan;
				}
			}
			return nullptr;
		}

		bool graph_storage_dispatch_buffer(const MPSGraphExecutableBundle & bundle,
			const MPSGraphTensorPlan & tensor_plan,
			const std::string & label,
			MetalBuffer *buffer,
			size_t *offset,
			std::string & reason)
		{
			if (!buffer || !offset)
			{
				reason = std::string(not label.empty() ? label : "graph tensor") + " dispatch output is unavailable";
				return false;
			}

			MPSGraphBufferBridge bridge;
			if (!training_graph_tensor_storage_bridge(bundle, tensor_plan, bridge, reason) ||
				!bridge.buffer_handle ||
				bridge.bytes < tensor_plan.bytes)
			{
				if (reason.empty())
				{
					reason = std::string(not label.empty() ? label : "graph tensor") + " is missing graph-owned storage";
				}
				return false;
			}

			*buffer = {bridge.buffer_handle, bridge.offset + bridge.bytes};
			*offset = bridge.offset;
			return true;
		}

		bool batchnorm_stats_dispatch_buffer(const MPSGraphExecutableBundle & bundle,
			Layer & layer,
			const MPSGraphTensorPlan & tensor_plan,
			const std::string & label,
			MetalBuffer *buffer,
			size_t *offset,
			std::string & reason)
		{
			if (!tensor_plan.materialize_to_darknet_buffer)
			{
				return graph_storage_dispatch_buffer(bundle, tensor_plan, label, buffer, offset, reason);
			}

			const void *ptr = nullptr;
			switch (tensor_plan.binding_kind)
			{
				case MPSGraphTensorBindingKind::kBatchMeanTape:
					ptr = layer.mean;
					break;
				case MPSGraphTensorBindingKind::kBatchVarianceTape:
					ptr = layer.variance;
					break;
				default:
					reason = std::string(not label.empty() ? label : "batchnorm stat") + " has unsupported materialized binding";
					return false;
			}

			TrainingBufferView view{};
			if (!load_required_training_view(ptr,
				tensor_plan.bytes,
				TrainingBufferAccess::kRead,
				label,
				view,
				reason))
			{
				return false;
			}
			*buffer = metal_buffer_from_training_view(view);
			*offset = view.offset;
			return true;
		}

		bool dispatch_update_running_batchnorm_stats_with_encoder(Network & net, void *encoder_handle, std::string & reason)
		{
			const auto bundle = training_graph_executable_bundle(&net, false);
			if (!bundle.available())
			{
				reason = bundle.reason().empty() ? "MPSGraph forward bundle is unavailable" : bundle.reason();
				return false;
			}

			for (int i = 0; i < net.n; ++i)
			{
				Layer & layer = net.layers[i];
				if (layer.type != Darknet::ELayerType::CONVOLUTIONAL || !layer.batch_normalize)
				{
					continue;
				}

				const int channels = layer.n;
				const size_t bytes = bytes_for_float_count(channels);
				if (channels <= 0 || bytes == 0)
				{
					continue;
				}

				const auto *mean_plan = find_forward_result_plan(bundle, i, MPSGraphTensorBindingKind::kBatchMeanTape);
				const auto *variance_plan = find_forward_result_plan(bundle, i, MPSGraphTensorBindingKind::kBatchVarianceTape);
				if (!mean_plan || !variance_plan)
				{
					reason = "batch-normalization graph mean/variance tapes are missing for layer #" + std::to_string(i);
					return false;
				}

				TrainingBufferView rolling_mean_view{};
				TrainingBufferView rolling_variance_view{};
				TrainingBufferView unbiased_variance_view{};
				if (!load_required_training_view(layer.rolling_mean,
						bytes,
						TrainingBufferAccess::kReadWrite,
						"rolling mean",
						rolling_mean_view,
						reason) ||
					!load_required_training_view(layer.rolling_variance,
						bytes,
						TrainingBufferAccess::kReadWrite,
						"rolling variance",
						rolling_variance_view,
						reason) ||
					!load_required_training_view(layer.variance_delta,
						bytes,
						TrainingBufferAccess::kWrite,
						"unbiased variance scratch",
						unbiased_variance_view,
						reason))
				{
					return false;
				}

				MetalBuffer mean_buffer{};
				MetalBuffer variance_buffer{};
				size_t mean_offset = 0;
				size_t variance_offset = 0;
				if (!batchnorm_stats_dispatch_buffer(bundle, layer, *mean_plan, "batch mean", &mean_buffer, &mean_offset, reason) ||
					!batchnorm_stats_dispatch_buffer(bundle, layer, *variance_plan, "batch variance", &variance_buffer, &variance_offset, reason))
				{
					return false;
				}

				const MetalBuffer rolling_mean_buffer = metal_buffer_from_training_view(rolling_mean_view);
				const MetalBuffer rolling_variance_buffer = metal_buffer_from_training_view(rolling_variance_view);
				const MetalBuffer unbiased_variance_buffer = metal_buffer_from_training_view(unbiased_variance_view);
				const uint32_t total = static_cast<uint32_t>(channels);
				const MetalDispatchBuffer copy_buffers[] = {
					{&variance_buffer, variance_offset},
					{&unbiased_variance_buffer, unbiased_variance_view.offset},
				};
				const MetalDispatchBytes copy_bytes[] = {
					{&total, sizeof(total)},
				};
				if (!dispatch_training_kernel_1d_with_encoder("buffer_copy",
					static_cast<size_t>(channels),
					copy_buffers, 2,
					copy_bytes, 1,
					encoder_handle,
					reason))
				{
					return false;
				}
				const int sample_count = layer.batch * layer.out_h * layer.out_w;
				const float unbiased_scale = sample_count > 1
					? static_cast<float>(sample_count) / static_cast<float>(sample_count - 1)
					: 1.0f;
				const MetalDispatchBuffer scale_buffers[] = {
					{&unbiased_variance_buffer, unbiased_variance_view.offset},
				};
				const MetalDispatchBytes scale_bytes[] = {
					{&unbiased_scale, sizeof(unbiased_scale)},
					{&total, sizeof(total)},
				};
				if (!dispatch_training_kernel_1d_with_encoder("buffer_scal",
					static_cast<size_t>(channels),
					scale_buffers, 1,
					scale_bytes, 2,
					encoder_handle,
					reason))
				{
					return false;
				}
				const MetalDispatchBuffer buffers[] = {
					{&rolling_mean_buffer, rolling_mean_view.offset},
					{&rolling_variance_buffer, rolling_variance_view.offset},
					{&mean_buffer, mean_offset},
					{&unbiased_variance_buffer, unbiased_variance_view.offset},
				};
				const float keep = 0.99f;
				const float add = 0.01f;
				const MetalDispatchBytes kernel_bytes[] = {
					{&keep, sizeof(keep)},
					{&add, sizeof(add)},
					{&total, sizeof(total)},
				};
				if (!dispatch_training_kernel_1d_with_encoder("bn_rolling_update_kernel",
					static_cast<size_t>(channels),
					buffers, 4,
					kernel_bytes, 3,
					encoder_handle,
					reason))
				{
					return false;
				}
				Darknet::Apple::mps_notify_layer_running_stats_updated_gpu(&layer);
			}

			return true;
		}

		bool dispatch_gpu_yolo_plain_loss_layer(Network & net,
			Layer & layer,
			float *detector_input,
			float *prev_delta,
			const float *truth,
			void *encoder_handle,
			AppleYoloPlainLossReadbacks & readbacks,
			std::string & reason)
		{
			if (!detector_input || !prev_delta || !truth)
			{
				reason = "GPU YOLO loss requires detector input, previous-layer delta, and truth buffers";
				return false;
			}
			if (!layer.output || !layer.delta || !layer.biases || !layer.mask || !layer.labels || !layer.class_ids)
			{
				reason = "GPU YOLO loss requires YOLO output, delta, biases, mask, labels, and class-id buffers";
				return false;
			}
			if (layer.inputs != layer.outputs)
			{
				reason = "GPU YOLO loss requires matching YOLO input and output sizes";
				return false;
			}

			const size_t head_bytes = bytes_for_float_count(layer.outputs * layer.batch);
			const size_t truth_bytes = bytes_for_float_count(layer.batch * layer.truths);
			const size_t bias_bytes = bytes_for_float_count(layer.total * 2);
			const size_t mask_bytes = sizeof(int) * static_cast<size_t>(std::max(0, layer.n));
			const size_t labels_count = static_cast<size_t>(layer.batch) * static_cast<size_t>(layer.n) * static_cast<size_t>(layer.w) * static_cast<size_t>(layer.h);
			const size_t labels_bytes = sizeof(int) * labels_count;

			note_training_shared_buffer_updated_from_host(truth);
			note_training_shared_buffer_updated_from_host(layer.biases);
			note_training_shared_buffer_updated_from_host(layer.mask);

			TrainingBufferView detector_input_view{};
			TrainingBufferView output_view{};
			TrainingBufferView delta_view{};
			TrainingBufferView prev_delta_view{};
			TrainingBufferView truth_view{};
			TrainingBufferView biases_view{};
			TrainingBufferView mask_view{};
			TrainingBufferView labels_view{};
			TrainingBufferView class_ids_view{};
			if (!load_required_training_view(detector_input, head_bytes, TrainingBufferAccess::kRead,
					"YOLO detector input", detector_input_view, reason) ||
				!load_required_training_view(layer.output, head_bytes, TrainingBufferAccess::kWrite,
					"YOLO output", output_view, reason) ||
				!load_required_training_view(layer.delta, head_bytes, TrainingBufferAccess::kWrite,
					"YOLO delta", delta_view, reason) ||
				!load_required_training_view(prev_delta, head_bytes, TrainingBufferAccess::kReadWrite,
					"YOLO previous-layer delta", prev_delta_view, reason) ||
				!load_required_training_view(truth, truth_bytes, TrainingBufferAccess::kRead,
					"YOLO truth", truth_view, reason) ||
				!load_required_training_view(layer.biases, bias_bytes, TrainingBufferAccess::kRead,
					"YOLO biases", biases_view, reason) ||
				!load_required_training_view(layer.mask, mask_bytes, TrainingBufferAccess::kRead,
					"YOLO mask", mask_view, reason) ||
				!load_required_training_view(layer.labels, labels_bytes, TrainingBufferAccess::kWrite,
					"YOLO labels", labels_view, reason) ||
				!load_required_training_view(layer.class_ids, labels_bytes, TrainingBufferAccess::kWrite,
					"YOLO class IDs", class_ids_view, reason))
			{
				return false;
			}

			const size_t stats_bytes = gpu_yolo_loss_stats_bytes_for_layer(layer);
			if (!readbacks.stats_buffer.handle ||
				stats_bytes == 0 ||
				readbacks.next_stats_offset > readbacks.stats_bytes ||
				stats_bytes > readbacks.stats_bytes - readbacks.next_stats_offset)
			{
				reason = "GPU YOLO loss stats buffer is incomplete";
				return false;
			}
			const size_t stats_offset = readbacks.next_stats_offset;

			const MetalBuffer detector_input_buffer = metal_buffer_from_training_view(detector_input_view);
			const MetalBuffer output_buffer = metal_buffer_from_training_view(output_view);
			const MetalBuffer delta_buffer = metal_buffer_from_training_view(delta_view);
			const MetalBuffer prev_delta_buffer = metal_buffer_from_training_view(prev_delta_view);
			const MetalBuffer truth_buffer = metal_buffer_from_training_view(truth_view);
			const MetalBuffer biases_buffer = metal_buffer_from_training_view(biases_view);
			const MetalBuffer mask_buffer = metal_buffer_from_training_view(mask_view);
			const MetalBuffer labels_buffer = metal_buffer_from_training_view(labels_view);
			const MetalBuffer class_ids_buffer = metal_buffer_from_training_view(class_ids_view);
			const MetalBuffer stats_buffer = readbacks.stats_buffer;

			const AppleYoloPlainLossParams params{
				static_cast<uint32_t>(layer.batch),
				static_cast<uint32_t>(layer.w),
				static_cast<uint32_t>(layer.h),
				static_cast<uint32_t>(layer.n),
				static_cast<uint32_t>(layer.total),
				static_cast<uint32_t>(layer.classes),
				static_cast<uint32_t>(layer.outputs),
				static_cast<uint32_t>(layer.truths),
				static_cast<uint32_t>(layer.truth_size),
				static_cast<uint32_t>(layer.max_boxes),
				static_cast<uint32_t>(net.w),
				static_cast<uint32_t>(net.h),
				static_cast<uint32_t>(layer.new_coords ? 1 : 0),
				static_cast<uint32_t>(layer.objectness_smooth ? 1 : 0),
				layer.scale_x_y,
				layer.ignore_thresh,
				layer.truth_thresh,
				layer.iou_thresh,
				layer.obj_normalizer,
				layer.cls_normalizer,
				layer.iou_normalizer,
				layer.max_delta
			};
			const MetalDispatchBytes params_bytes[] = {
				{&params, sizeof(params)},
			};

			const size_t prediction_count = static_cast<size_t>(layer.batch) * static_cast<size_t>(layer.n) * static_cast<size_t>(layer.w) * static_cast<size_t>(layer.h);
			const MetalDispatchBuffer base_buffers[] = {
				{&detector_input_buffer, detector_input_view.offset},
				{&output_buffer, output_view.offset},
				{&delta_buffer, delta_view.offset},
				{&labels_buffer, labels_view.offset},
				{&class_ids_buffer, class_ids_view.offset},
			};
			if (!dispatch_training_kernel_1d_with_encoder("yolo_plain_activate_base_delta",
				prediction_count,
				base_buffers, 5,
				params_bytes, 1,
				encoder_handle,
				reason))
			{
				return false;
			}

			const MetalDispatchBuffer ignore_buffers[] = {
				{&output_buffer, output_view.offset},
				{&delta_buffer, delta_view.offset},
				{&truth_buffer, truth_view.offset},
				{&biases_buffer, biases_view.offset},
				{&mask_buffer, mask_view.offset},
			};
			if (!dispatch_training_kernel_1d_with_encoder("yolo_plain_ignore_delta",
				prediction_count,
				ignore_buffers, 5,
				params_bytes, 1,
				encoder_handle,
				reason))
			{
				return false;
			}

			const MetalDispatchBuffer assign_buffers[] = {
				{&output_buffer, output_view.offset},
				{&delta_buffer, delta_view.offset},
				{&truth_buffer, truth_view.offset},
				{&biases_buffer, biases_view.offset},
				{&mask_buffer, mask_view.offset},
				{&labels_buffer, labels_view.offset},
				{&class_ids_buffer, class_ids_view.offset},
				{&stats_buffer, stats_offset},
			};
			if (!dispatch_training_kernel_1d_with_encoder("yolo_plain_assign_truth_delta",
				static_cast<size_t>(layer.batch),
				assign_buffers, 8,
				params_bytes, 1,
				encoder_handle,
				reason))
			{
				return false;
			}

			if (layer.iou_thresh < 1.0f)
			{
				const MetalDispatchBuffer average_buffers[] = {
					{&delta_buffer, delta_view.offset},
				};
				if (!dispatch_training_kernel_1d_with_encoder("yolo_plain_average_thresh_deltas",
					prediction_count,
					average_buffers, 1,
					params_bytes, 1,
					encoder_handle,
					reason))
				{
					return false;
				}
			}

			const MetalDispatchBuffer stats_buffers[] = {
				{&delta_buffer, delta_view.offset},
				{&stats_buffer, stats_offset},
			};
			if (!dispatch_training_kernel_1d_with_encoder("yolo_plain_batch_loss_stats",
				static_cast<size_t>(layer.batch),
				stats_buffers, 2,
				params_bytes, 1,
				encoder_handle,
				reason))
			{
				return false;
			}

			const float alpha = net.loss_scale * layer.delta_normalizer;
			const uint32_t total = static_cast<uint32_t>(layer.batch * layer.inputs);
			const MetalDispatchBytes axpy_bytes[] = {
				{&alpha, sizeof(alpha)},
				{&total, sizeof(total)},
			};
			const MetalDispatchBuffer axpy_buffers[] = {
				{&delta_buffer, delta_view.offset},
				{&prev_delta_buffer, prev_delta_view.offset},
			};
			if (!dispatch_training_kernel_1d_with_encoder("buffer_axpy",
				total,
				axpy_buffers, 2,
				axpy_bytes, 2,
				encoder_handle,
				reason))
			{
				return false;
			}

			note_training_shared_buffer_updated_from_gpu(layer.output);
			note_training_shared_buffer_updated_from_gpu(layer.delta);
			note_training_shared_buffer_updated_from_gpu(prev_delta);
			note_training_shared_buffer_updated_from_gpu(layer.labels);
			note_training_shared_buffer_updated_from_gpu(layer.class_ids);

			AppleYoloPlainLossReadback readback;
			readback.layer = &layer;
			readback.stats_offset = stats_offset;
			readback.stats_bytes = stats_bytes;
			readbacks.next_stats_offset += stats_bytes;
			readbacks.entries.push_back(readback);
			reason = "ok";
			return true;
		}

		bool collect_gpu_yolo_loss_stats(Network & net,
			AppleYoloPlainLossReadbacks & readbacks,
			std::string & reason)
		{
			if (!readbacks.stats_buffer.handle || readbacks.stats_bytes == 0 || readbacks.entries.empty())
			{
				reason = "GPU YOLO loss stats readback is incomplete";
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			}
			if (readbacks.stats_bytes % sizeof(AppleYoloPlainLossStats) != 0)
			{
				reason = "GPU YOLO loss stats readback is misaligned";
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			}

			std::vector<AppleYoloPlainLossStats> all_stats(readbacks.stats_bytes / sizeof(AppleYoloPlainLossStats));
			if (!metal_buffer_download(&readbacks.stats_buffer, 0, all_stats.data(), readbacks.stats_bytes))
			{
				reason = "failed to read GPU YOLO loss stats";
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			}

			for (auto & readback : readbacks.entries)
			{
				if (!readback.layer ||
					readback.stats_bytes == 0 ||
					readback.stats_offset % sizeof(AppleYoloPlainLossStats) != 0 ||
					readback.stats_bytes % sizeof(AppleYoloPlainLossStats) != 0 ||
					readback.stats_offset > readbacks.stats_bytes ||
					readback.stats_bytes > readbacks.stats_bytes - readback.stats_offset)
				{
					reason = "GPU YOLO loss stats readback is incomplete";
					release_gpu_yolo_loss_readbacks(readbacks);
					return false;
				}

				Layer & layer = *readback.layer;
				const size_t stats_begin = readback.stats_offset / sizeof(AppleYoloPlainLossStats);
				const size_t stats_count = readback.stats_bytes / sizeof(AppleYoloPlainLossStats);
				const AppleYoloPlainLossStats *stats = all_stats.data() + stats_begin;

				double class_sq = 0.0;
				double iou_loss = 0.0;
				double count = 0.0;
				double rewritten = 0.0;
				for (size_t i = 0; i < stats_count; ++i)
				{
					const auto & item = stats[i];
					class_sq += item.class_sq;
					iou_loss += item.iou_loss;
					count += item.count;
					rewritten += item.rewritten;
				}

				if (net.total_bbox)
				{
					*net.total_bbox += static_cast<int>(std::lround(count));
				}
				if (net.rewritten_bbox)
				{
					*net.rewritten_bbox += static_cast<int>(std::lround(rewritten));
				}
				if (layer.cost)
				{
					const double positive_count = count > 0.0 ? count : 1.0;
					const double avg_iou_loss = static_cast<double>(layer.iou_normalizer) * (iou_loss / positive_count);
					*layer.cost = static_cast<float>(avg_iou_loss + class_sq);
				}
			}

			release_gpu_yolo_loss_readbacks(readbacks);
			reason = "ok";
			return true;
		}

		bool run_training_graph_gpu_yolo_loss(Network & net,
			const Darknet::NetworkState & state,
			AppleYoloPlainLossReadbacks & readbacks,
			const bool update_rolling_stats_on_gpu,
			std::string & reason,
			void *& command_buffer_handle)
		{
			release_gpu_yolo_loss_readbacks(readbacks);
			auto fail_with_pending_command_buffer = [&]() -> bool
			{
				if (command_buffer_handle)
				{
					commit_and_wait_training_command_buffer(command_buffer_handle, nullptr, nullptr);
				}
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			};

			if (!prepare_gpu_yolo_loss_readbacks(net, readbacks, reason))
			{
				return fail_with_pending_command_buffer();
			}
			const auto backward_bundle = training_graph_executable_bundle(&net, true);
			int boundary_delta_feed_count = 0;
			for (const auto & tensor_plan : backward_bundle.feed_plans())
			{
				if (tensor_plan.binding_kind == MPSGraphTensorBindingKind::kYoloBoundaryDelta)
				{
					++boundary_delta_feed_count;
				}
			}

			void *& active_command_buffer_handle = command_buffer_handle;
			if (!ensure_training_command_buffer(active_command_buffer_handle))
			{
				reason = "failed to allocate GPU YOLO loss command buffer";
				return fail_with_pending_command_buffer();
			}
			void *command_buffer = active_command_buffer_handle;
			void *encoder_handle = metal_begin_compute_encoder(command_buffer);
			if (!encoder_handle)
			{
				reason = "failed to allocate GPU YOLO loss compute encoder";
				commit_and_wait_training_command_buffer(active_command_buffer_handle, nullptr, nullptr);
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			}
			bool encoder_open = true;
			auto close_encoder = [&encoder_handle, &encoder_open]()
			{
				if (encoder_open && encoder_handle)
				{
					metal_end_compute_encoder(encoder_handle);
					encoder_handle = nullptr;
				}
				encoder_open = false;
			};

			auto & stats = training_iteration_stats();
			const auto start = std::chrono::high_resolution_clock::now();
			const bool clear_ok = dispatch_clear_yolo_boundary_deltas_with_encoder(net, encoder_handle, reason);
			if (!clear_ok)
			{
				close_encoder();
				commit_and_wait_training_command_buffer(active_command_buffer_handle, nullptr, nullptr);
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			}
			if (update_rolling_stats_on_gpu &&
				!dispatch_update_running_batchnorm_stats_with_encoder(net, encoder_handle, reason))
			{
				close_encoder();
				commit_and_wait_training_command_buffer(active_command_buffer_handle, nullptr, nullptr);
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			}

			int head_count = 0;
			for (int i = 0; i < net.n; ++i)
			{
				Layer & layer = net.layers[i];
				if (!is_detector_head_layer_type(layer.type) || !training_is_terminal_detector_head(net, i))
				{
					continue;
				}
				if (i <= 0)
				{
					reason = "GPU YOLO loss requires a previous layer delta buffer";
					close_encoder();
					commit_and_wait_training_command_buffer(active_command_buffer_handle, nullptr, nullptr);
					release_gpu_yolo_loss_readbacks(readbacks);
					return false;
				}

				Layer & prev = net.layers[i - 1];
				if (!dispatch_gpu_yolo_plain_loss_layer(net, layer, prev.output, prev.delta, state.truth, encoder_handle, readbacks, reason))
				{
					close_encoder();
					commit_and_wait_training_command_buffer(active_command_buffer_handle, nullptr, nullptr);
					release_gpu_yolo_loss_readbacks(readbacks);
					return false;
				}
				++head_count;
			}

			if (head_count == 0)
			{
				reason = "no terminal YOLO heads found";
				close_encoder();
				commit_and_wait_training_command_buffer(active_command_buffer_handle, nullptr, nullptr);
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			}
			if (head_count != boundary_delta_feed_count)
			{
				reason =
					"YOLO boundary delta feed count mismatch: dispatched " +
					std::to_string(head_count) +
					" head(s), backward graph expects " +
					std::to_string(boundary_delta_feed_count);
				close_encoder();
				commit_and_wait_training_command_buffer(active_command_buffer_handle, nullptr, nullptr);
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			}

			close_encoder();
			// MPSGraph forward/backward and these YOLO kernels now use the same
			// training queue, so Metal queue order makes the deltas visible to
			// backward without blocking the CPU here.
			if (!commit_training_command_buffer(active_command_buffer_handle, nullptr))
			{
				reason = "GPU YOLO loss command buffer failed";
				release_gpu_yolo_loss_readbacks(readbacks);
				return false;
			}
			const double elapsed_ms = std::chrono::duration<double, std::milli>(
				std::chrono::high_resolution_clock::now() - start).count();
			stats.backward_bridge_ms += elapsed_ms;
			reason = "ok";
			return true;
		}

	float train_training_graph_datum(Network & net, float *x, float *y)
	{
		begin_training_iteration(net);
		auto & stats = training_iteration_stats();
		// X/y are owned by the generic train loop and freed after this outer
		// iteration. Refresh mirrored fallbacks and drop their wrappers once the
		// synchronous backward pass has consumed them.
		note_training_shared_buffer_updated_from_host(x);
		note_training_shared_buffer_updated_from_host(y);

		NetworkState state = {0};
		*net.seen += net.batch;
		state.index = 0;
		state.net = net;
		state.input = x;
		state.delta = nullptr;
		state.truth = y;
		state.train = 1;
		state.workspace = net.workspace;

		std::string truth_reason;
		if (!validate_gpu_yolo_truth_buffer(net, y, truth_reason))
		{
			darknet_fatal_error(DARKNET_LOC,
				"Apple GPU YOLO loss truth validation failed: %s",
				truth_reason.c_str());
		}

		AppleYoloPlainLossReadbacks gpu_yolo_loss_readbacks;
		std::string graph_reason;
		void *forward_encoded_command_buffer = nullptr;
		if (!run_training_graph_forward_probe(net,
			state,
			graph_reason,
			forward_encoded_command_buffer))
		{
			darknet_fatal_error(DARKNET_LOC, "Apple training graph forward failed: %s", graph_reason.c_str());
		}
		if (!run_training_graph_gpu_yolo_loss(net,
			state,
			gpu_yolo_loss_readbacks,
			true,
			graph_reason,
			forward_encoded_command_buffer))
		{
			darknet_fatal_error(DARKNET_LOC, "Apple GPU YOLO loss bridge failed: %s", graph_reason.c_str());
		}

		if (!run_training_graph_terminal_backward(net, x, graph_reason))
		{
			release_gpu_yolo_loss_readbacks(gpu_yolo_loss_readbacks);
			darknet_fatal_error(DARKNET_LOC, "Apple training graph backward failed: %s", graph_reason.c_str());
		}
		double backward_wait_ms = 0.0;
		wait_for_training_update_command_buffer(&backward_wait_ms);
		if (const std::string asynchronous_error = take_training_execution_error(); !asynchronous_error.empty())
		{
			release_gpu_yolo_loss_readbacks(gpu_yolo_loss_readbacks);
			darknet_fatal_error(DARKNET_LOC, "Apple training graph execution failed: %s", asynchronous_error.c_str());
		}
		stats.backward_run_ms += backward_wait_ms;
		const auto stats_start = std::chrono::high_resolution_clock::now();
		if (!collect_gpu_yolo_loss_stats(net, gpu_yolo_loss_readbacks, graph_reason))
		{
			darknet_fatal_error(DARKNET_LOC, "Apple GPU YOLO loss stats readback failed: %s", graph_reason.c_str());
		}
		const double stats_ms = std::chrono::duration<double, std::milli>(
			std::chrono::high_resolution_clock::now() - stats_start).count();
		stats.backward_bridge_ms += stats_ms;
		stats.total_batch_ms =
			stats.forward_encode_ms + stats.forward_run_ms +
			stats.backward_encode_ms + stats.backward_run_ms + stats.backward_bridge_ms;
		const int completed_subdivisions = std::max(1, net.current_subdivision + 1);
		stats.images_per_second = stats.total_batch_ms > 0.0
			? (1000.0 * net.batch * completed_subdivisions / stats.total_batch_ms)
			: 0.0;

		erase_training_shared_buffer_storage(x);
		erase_training_shared_buffer_storage(y);

		return get_network_cost(net);
	}
}
