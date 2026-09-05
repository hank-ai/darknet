#include "darknet_internal.hpp"

#include "apple_training.hpp"
#include "apple_training_internal.hpp"
#include "apple_mps.hpp"
#include "apple_training_optimizer.hpp"

#include "darknet_layers.hpp"
#include "darknet_network.hpp"
#include "metal_backend.hpp"

#include <chrono>
#include <cstring>
#include <string>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace Darknet::Apple
{
	using namespace TrainingDetail;

	bool configure_training_graph(Network & net, std::string *reason)
	{
		std::string graph_reason;
		if (!configure_training_graph_engine(net, &graph_reason))
		{
			invalidate_training_buffers(&net);
			invalidate_training_graph_engine(&net);
			if (reason) *reason = graph_reason.empty() ? "Apple training graph is unavailable" : graph_reason;
			return false;
		}

		std::string buffer_reason;
		if (!prepare_training_buffers(net, &buffer_reason))
		{
			invalidate_training_buffers(&net);
			invalidate_training_graph_engine(&net);
			if (reason) *reason = buffer_reason.empty() ? "Apple training graph runtime buffers are unavailable" : buffer_reason;
			return false;
		}
		if (reason) *reason = "ok";
		return true;
	}

	bool training_graph_enabled(const Network & net)
	{
		return training_graph_engine_enabled(net);
	}

	void invalidate_training_graph(Network * net)
	{
		if (!net || !training_graph_engine_enabled(*net))
		{
			return;
		}

		invalidate_training_buffers(net);
		invalidate_training_graph_engine(net);
	}

	void sync_training_host_state(Network & net, bool sync_rolling_stats, bool sync_parameters, double *wait_ms)
	{
		sync_training_layers_to_host(net, wait_ms, sync_rolling_stats, sync_parameters);
	}

	bool prepare_validation_network(Network & training, Network & validation, std::string *reason)
	{
		if (&training == &validation || !training.layers || !validation.layers ||
			training.n <= 0 || training.n != validation.n)
		{
			if (reason) *reason = "validation requires a distinct network with matching layers";
			return false;
		}
		if (!training_graph_engine_enabled(training))
		{
			if (reason) *reason = "Apple training graph is not configured for the source network";
			return false;
		}

		const auto channel_count = [](const Layer &layer)
		{
			if (layer.type == ELayerType::CONVOLUTIONAL) return layer.n;
			if (layer.type == ELayerType::CONNECTED) return layer.outputs;
			return 0;
		};
		const auto storage_is_dedicated = [](const float *src, const float *dst, const int count)
		{
			return !src || count <= 0 || (dst && dst != src);
		};

		// Validate the entire destination before copying so a configuration error
		// cannot leave a partially refreshed validation network.
		for (int i = 0; i < training.n; ++i)
		{
			const Layer &src = training.layers[i];
			const Layer &dst = validation.layers[i];
			const int channels = channel_count(src);
			if (src.type != dst.type || src.nweights != dst.nweights ||
				!storage_is_dedicated(src.weights, dst.weights, src.nweights) ||
				!storage_is_dedicated(src.biases, dst.biases, channels) ||
				!storage_is_dedicated(src.scales, dst.scales, channels) ||
				!storage_is_dedicated(src.rolling_mean, dst.rolling_mean, channels) ||
				!storage_is_dedicated(src.rolling_variance, dst.rolling_variance, channels))
			{
				if (reason)
				{
					*reason = "validation layer #" + std::to_string(i) +
						" does not have compatible dedicated parameter storage";
				}
				return false;
			}
		}

		sync_training_layers_to_host(training, nullptr, true, true);
		const auto copy = [](const float *src, float *dst, const int count)
		{
			if (src && dst && count > 0)
			{
				std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(float));
			}
		};
		for (int i = 0; i < training.n; ++i)
		{
			const Layer &src = training.layers[i];
			Layer &dst = validation.layers[i];
			const int channels = channel_count(src);
			copy(src.weights, dst.weights, src.nweights);
			copy(src.biases, dst.biases, channels);
			copy(src.scales, dst.scales, channels);
			copy(src.rolling_mean, dst.rolling_mean, channels);
			copy(src.rolling_variance, dst.rolling_variance, channels);
			Darknet::Apple::mps_invalidate_layer_cache(&dst);
		}

		if (reason) *reason = "ok";
		return true;
	}

	float train_graph_network_datum(Network & net, float *x, float *y)
	{
		if (!training_graph_engine_enabled(net))
		{
			const auto info = build_training_graph_info(&net);
			const std::string reason =
				!info.reason.empty() ? info.reason : "Apple graph training is not configured";
			darknet_fatal_error(DARKNET_LOC,
				"Apple training requires the graph engine. graph_reason=%s",
				reason.c_str());
		}

		return train_training_graph_datum(net, x, y);
	}

	void update_training_graph(Network & net)
	{
		if (!training_graph_engine_enabled(net))
		{
			const auto info = build_training_graph_info(&net);
			const std::string reason =
				!info.reason.empty() ? info.reason : "Apple graph training is not configured";
			darknet_fatal_error(DARKNET_LOC,
				"Apple training updates require the graph engine. graph_reason=%s",
				reason.c_str());
		}

		const auto update_start = std::chrono::high_resolution_clock::now();
		const int iteration_num = (*net.seen) / (net.batch * net.subdivisions);
		const int update_batch = net.batch * net.subdivisions * get_sequence_value(net);
		const float rate = get_current_rate(net);
		void *optimizer_command_buffer = nullptr;
		void *optimizer_encoder = nullptr;
		double update_wait_ms = 0.0;
		int gpu_update_count = 0;

		for (int i = 0; i < net.n; ++i)
		{
			Layer & l = net.layers[i];
			if (l.train == 0)
			{
				continue;
			}
			l.t = get_current_batch(net);
			if (!training_update_gate_open(l.burnin_update,
				net.burn_in,
				l.train_only_bn != 0,
				l.update != nullptr,
				l.dont_update,
				iteration_num))
			{
				continue;
			}
			const std::string layer_type = to_string(l.type);

			std::string optimizer_reason;
			if (!training_optimizer_update_layer(net, l, update_batch, rate, &optimizer_command_buffer, &optimizer_encoder, &optimizer_reason))
			{
				darknet_fatal_error(DARKNET_LOC,
					"Apple GPU optimizer update failed at layer %d (%s): %s",
					i,
					layer_type.c_str(),
					optimizer_reason.c_str());
			}
			++gpu_update_count;
		}

		if (optimizer_encoder)
		{
			metal_end_compute_encoder(optimizer_encoder);
			optimizer_encoder = nullptr;
		}
		if (gpu_update_count > 0)
		{
			// Apple-only queue ordering: GPU optimizer kernels and the next
			// MPSGraph forward use the same training queue, so the next batch
			// observes the updated weights without blocking the CPU here.
			Detail::ObjcStrong pending_update((__bridge id<MTLCommandBuffer>)optimizer_command_buffer);
			if (!commit_training_command_buffer(&optimizer_command_buffer, nullptr))
			{
				darknet_fatal_error(DARKNET_LOC,
					"Apple GPU optimizer update submission failed after %d layer update(s)",
					gpu_update_count);
			}
			record_training_update_command_buffer((id<MTLCommandBuffer>)pending_update.get());
		}

		const auto update_end = std::chrono::high_resolution_clock::now();
		const double update_encode_ms = std::chrono::duration<double, std::milli>(update_end - update_start).count();
		finalize_training_iteration(net, update_encode_ms, update_wait_ms);
	}

	TrainingGraphInfo build_training_graph_info(const Network * net)
	{
		return training_graph_info_snapshot(net);
	}

}
