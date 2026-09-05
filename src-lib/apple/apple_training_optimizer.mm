#include "darknet_internal.hpp"
#include "apple_mps.hpp"
#include "apple_training_internal.hpp"
#include "apple_training_optimizer.hpp"
#include "darknet_cfg_and_state.hpp"
#include "metal_backend.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace Darknet::Apple::TrainingDetail
{
	namespace
	{
		bool is_optimizer_gpu_eligible(const Network & net, const Layer & layer)
		{
			(void)net;
			return layer.type == ELayerType::CONVOLUTIONAL;
		}

		MetalBuffer wrap_training_buffer(const TrainingBufferView & view)
		{
			return {view.buffer_handle, view.offset + view.bytes};
		}

		bool dispatch_optimizer_kernel_1d_with_encoder(const std::string & kernel_name,
			size_t threads,
			const MetalDispatchBuffer *buffers,
			size_t buffer_count,
			const MetalDispatchBytes *bytes,
			size_t bytes_count,
			void *encoder_handle)
		{
			if (kernel_name.empty() || threads == 0 || !encoder_handle)
			{
				return false;
			}

			return metal_dispatch_1d_with_encoder(kernel_name,
				threads,
				std::min<size_t>(threads, 256),
				nullptr, 0,
				buffers, buffer_count,
				bytes, bytes_count,
				encoder_handle);
		}

		bool load_training_view(float *ptr,
			const int count,
			const TrainingBufferAccess access,
			TrainingBufferView & view)
		{
			return ptr && count > 0 && get_training_buffer_view(
				ptr,
				static_cast<size_t>(count) * sizeof(float),
				access,
				view);
		}

		bool optimizer_parameter_views_available(const std::string & label, float *parameter, float *updates, const int count, std::string & reason)
		{
			TrainingBufferView parameter_view;
			TrainingBufferView update_view;
			if (!load_training_view(parameter, count, TrainingBufferAccess::kRead, parameter_view) ||
				!load_training_view(updates, count, TrainingBufferAccess::kRead, update_view))
			{
				reason = std::string(label) + "_buffer_unavailable";
				return false;
			}
			return true;
		}

		int channel_count_for_optimizer(const Layer & layer)
		{
			switch (layer.type)
			{
				case ELayerType::CONVOLUTIONAL:
					return layer.n;
				default:
					return 0;
			}
		}

		bool update_parameter_with_encoder(float *parameter,
			float *updates,
			const TrainingSgdStep & step,
			const bool sanitize_weights,
			void *encoder_handle)
		{
			TrainingBufferView parameter_view;
			TrainingBufferView update_view;
			if (!load_training_view(parameter, step.count, TrainingBufferAccess::kReadWrite, parameter_view) ||
				!load_training_view(updates, step.count, TrainingBufferAccess::kReadWrite, update_view))
			{
				return false;
			}

			const MetalBuffer wrapped_parameter = wrap_training_buffer(parameter_view);
			const MetalBuffer wrapped_update = wrap_training_buffer(update_view);
			const MetalDispatchBuffer buffers[] = {
				{&wrapped_parameter, parameter_view.offset},
				{&wrapped_update, update_view.offset},
			};
			const uint32_t total = static_cast<uint32_t>(step.count);
			if (sanitize_weights)
			{
				const MetalDispatchBytes bytes[] = {
					{&step.rate_over_batch, sizeof(step.rate_over_batch)},
					{&step.momentum, sizeof(step.momentum)},
					{&step.decay_term, sizeof(step.decay_term)},
					{&total, sizeof(total)},
				};
				return dispatch_optimizer_kernel_1d_with_encoder(
					"sgd_update_weights", total, buffers, 2, bytes, 4, encoder_handle);
			}

			const MetalDispatchBytes bytes[] = {
				{&step.rate_over_batch, sizeof(step.rate_over_batch)},
				{&step.momentum, sizeof(step.momentum)},
				{&total, sizeof(total)},
			};
			return dispatch_optimizer_kernel_1d_with_encoder(
				"sgd_update_parameter", total, buffers, 2, bytes, 3, encoder_handle);
		}

		bool optimizer_preflight_update_layer(const Layer & layer, std::string & reason)
		{
			if (!metal_kernel_available("sgd_update_weights") || !metal_kernel_available("sgd_update_parameter"))
			{
				reason = "optimizer_kernel_unavailable";
				return false;
			}

			bool has_trainable_buffers = false;
			if (layer.weights && layer.weight_updates && layer.nweights > 0)
			{
				if (!optimizer_parameter_views_available("weight_update", layer.weights, layer.weight_updates, layer.nweights, reason))
				{
					return false;
				}
				has_trainable_buffers = true;
			}

			const int channel_count = channel_count_for_optimizer(layer);
			if (layer.biases && layer.bias_updates && channel_count > 0)
			{
				if (!optimizer_parameter_views_available("bias_update", layer.biases, layer.bias_updates, channel_count, reason))
				{
					return false;
				}
				has_trainable_buffers = true;
			}

			if (layer.scales && layer.scale_updates && channel_count > 0)
			{
				if (!optimizer_parameter_views_available("scale_update", layer.scales, layer.scale_updates, channel_count, reason))
				{
					return false;
				}
				has_trainable_buffers = true;
			}

			if (!has_trainable_buffers)
			{
				reason = "missing_trainable_buffers";
				return false;
			}
			return true;
		}

	}

	std::string training_optimizer_summary(const Network & net)
	{
		(void)net;
		return "gpu_sgd";
	}

	std::string training_precision_summary(const Network & net)
	{
		(void)net;
		return "fp32";
	}

	bool training_optimizer_update_layer(Network & net,
		Layer & layer,
		const int update_batch,
		const float learning_rate,
		void *& command_buffer_handle,
		void *& encoder_handle,
		std::string & reason)
	{
		reason = {};

		if (!is_optimizer_gpu_eligible(net, layer))
		{
			reason = "layer_type_requires_cpu";
			return false;
		}
		if (!optimizer_preflight_update_layer(layer, reason))
		{
			return false;
		}
		if (!ensure_training_command_buffer(command_buffer_handle))
		{
			reason = "no_command_buffer";
			return false;
		}

		void *command_buffer = command_buffer_handle;
		const float loss_scale = net.loss_scale != 0.0f ? net.loss_scale : 1.0f;
		const float rate_over_batch = training_sgd_rate_over_batch(learning_rate / loss_scale, layer.learning_rate_scale, update_batch);
		const float weight_decay_term = training_sgd_decay_term(net.decay * loss_scale, update_batch, layer.type == ELayerType::CONVOLUTIONAL);
		// The caller owns and closes the shared encoder.
		void *optimizer_encoder = encoder_handle;
		if (!optimizer_encoder)
		{
			optimizer_encoder = metal_begin_compute_encoder(command_buffer);
			encoder_handle = optimizer_encoder;
		}
		if (!optimizer_encoder)
		{
			reason = "no_compute_encoder";
			return false;
		}

		bool updated_any = false;
		if (layer.weights && layer.weight_updates && layer.nweights > 0)
		{
			const TrainingSgdStep step{layer.nweights, rate_over_batch, net.momentum, weight_decay_term};
			if (!update_parameter_with_encoder(layer.weights, layer.weight_updates,
				step,
				true,
				optimizer_encoder))
			{
				reason = "weight_update_encode_failed";
				return false;
			}
			updated_any = true;
		}

		const int channel_count = channel_count_for_optimizer(layer);
		if (layer.biases && layer.bias_updates && channel_count > 0)
		{
			const TrainingSgdStep step{channel_count, rate_over_batch, net.momentum, 0.0f};
			if (!update_parameter_with_encoder(layer.biases, layer.bias_updates,
				step,
				false,
				optimizer_encoder))
			{
				reason = "bias_update_encode_failed";
				return false;
			}
			updated_any = true;
		}

		if (layer.scales && layer.scale_updates && channel_count > 0)
		{
			const TrainingSgdStep step{channel_count, rate_over_batch, net.momentum, 0.0f};
			if (!update_parameter_with_encoder(layer.scales, layer.scale_updates,
				step,
				false,
				optimizer_encoder))
			{
				reason = "scale_update_encode_failed";
				return false;
			}
			updated_any = true;
		}

		if (!updated_any)
		{
			reason = "missing_trainable_buffers";
			return false;
		}
		Darknet::Apple::mps_notify_layer_parameters_updated_gpu(&layer);
		Darknet::Apple::mps_notify_layer_optimizer_state_updated_gpu(&layer);
		reason = "gpu_sgd";
		return true;
	}
}
