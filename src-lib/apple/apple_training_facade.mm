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
#include <limits>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

namespace Darknet::Apple
{
	using namespace TrainingDetail;

	bool validate_training_data_views(const Network & net, const FloatBufferView & input,
		const FloatBufferView & truth, std::string & reason)
	{
		const auto element_count = [](const int batch, const int elements) -> size_t
		{
			if (batch <= 0 or elements <= 0 or
				static_cast<size_t>(elements) > std::numeric_limits<size_t>::max() / sizeof(float) / static_cast<size_t>(batch))
			{
				return 0;
			}
			return static_cast<size_t>(batch) * static_cast<size_t>(elements);
		};
		const size_t inputs = element_count(net.batch, net.inputs);
		if (inputs == 0 or not input.data() or input.size() < inputs)
		{
			reason = "input buffer is missing, too small, or has invalid geometry";
			return false;
		}
		if (not net.layers or net.n <= 0)
		{
			reason = "training network has no layers";
			return false;
		}
		size_t truths = 0;
		for (int i = 0; i < net.n; ++i)
		{
			const auto & layer = net.layers[i];
			if (layer.type != ELayerType::YOLO) continue;
			const size_t count = element_count(layer.batch, layer.truths);
			const size_t box_elements = element_count(layer.max_boxes, layer.truth_size);
			if (count == 0 or box_elements == 0 or box_elements > static_cast<size_t>(layer.truths))
			{
				reason = "YOLO truth geometry exceeds its declared buffer size";
				return false;
			}
			truths = std::max(truths, count);
		}
		if (truths == 0 or not truth.data() or truth.size() < truths)
		{
			reason = "truth buffer is missing or too small for the YOLO heads";
			return false;
		}
		reason = "ok";
		return true;
	}

	bool configure_training_graph(Network & net, std::string & reason)
	{
		std::string graph_reason;
		if (not configure_training_graph_engine(net, graph_reason))
		{
			invalidate_training_buffers();
			invalidate_training_graph_engine(net);
			reason = graph_reason.empty() ? "Apple training graph is unavailable" : graph_reason;
			return false;
		}

		std::string buffer_reason;
		if (not wait_for_training_runtime(buffer_reason))
		{
			invalidate_training_buffers();
			invalidate_training_graph_engine(net);
			reason = buffer_reason.empty() ? "Apple training graph runtime buffers are unavailable" : buffer_reason;
			return false;
		}
		reason = "ok";
		return true;
	}

	bool training_graph_enabled(const Network & net)
	{
		return training_graph_engine_enabled(net);
	}

	void invalidate_training_graph(Network & net)
	{
		if (not training_graph_engine_enabled(net))
		{
			return;
		}

		invalidate_training_buffers();
		invalidate_training_graph_engine(net);
	}

	double sync_training_host_state(Network & net, const bool sync_rolling_stats, const bool sync_parameters)
	{
		return sync_training_layers_to_host(net, sync_rolling_stats, sync_parameters);
	}

	bool prepare_validation_network(Network & training, Network & validation, std::string & reason)
	{
		if (&training == &validation or not training.layers or not validation.layers or
			training.n <= 0 or training.n != validation.n)
		{
			reason = "validation requires a distinct network with matching layers";
			return false;
		}
		if (not training_graph_engine_enabled(training))
		{
			reason = "Apple training graph is not configured for the source network";
			return false;
		}

		const auto channel_count = [](const Layer & layer)
		{
			if (layer.type == ELayerType::CONVOLUTIONAL) return layer.n;
			if (layer.type == ELayerType::CONNECTED) return layer.outputs;
			return 0;
		};
		const auto storage_is_dedicated = [](const float *src, const float *dst, const int count)
		{
			return not src or count <= 0 or (dst and dst != src);
		};

		// Validate the entire destination before copying so a configuration error
		// cannot leave a partially refreshed validation network.
		for (int i = 0; i < training.n; ++i)
		{
			const Layer & src = training.layers[i];
			const Layer & dst = validation.layers[i];
			const int channels = channel_count(src);
			const bool convolution_geometry_matches = src.type != ELayerType::CONVOLUTIONAL or
				(src.c == dst.c and src.n == dst.n and src.groups == dst.groups and src.size == dst.size);
			const bool connected_geometry_matches = src.type != ELayerType::CONNECTED or src.inputs == dst.inputs;
			if (src.type != dst.type or src.nweights != dst.nweights or channels != channel_count(dst) or
				src.batch_normalize != dst.batch_normalize or not convolution_geometry_matches or
				not connected_geometry_matches or
				not storage_is_dedicated(src.weights, dst.weights, src.nweights) or
				not storage_is_dedicated(src.biases, dst.biases, channels) or
				not storage_is_dedicated(src.scales, dst.scales, channels) or
				not storage_is_dedicated(src.rolling_mean, dst.rolling_mean, channels) or
				not storage_is_dedicated(src.rolling_variance, dst.rolling_variance, channels))
			{
				reason = "validation layer #" + std::to_string(i) +
					" does not have compatible dedicated parameter storage";
				return false;
			}
		}

		sync_training_layers_to_host(training, true, true);
		const auto copy = [](const float *src, float *dst, const int count)
		{
			if (src and dst and count > 0)
			{
				std::memcpy(dst, src, static_cast<size_t>(count) * sizeof(float));
			}
		};
		for (int i = 0; i < training.n; ++i)
		{
			const Layer & src = training.layers[i];
			Layer & dst = validation.layers[i];
			const int channels = channel_count(src);
			copy(src.weights, dst.weights, src.nweights);
			copy(src.biases, dst.biases, channels);
			copy(src.scales, dst.scales, channels);
			copy(src.rolling_mean, dst.rolling_mean, channels);
			copy(src.rolling_variance, dst.rolling_variance, channels);
			Darknet::Apple::mps_invalidate_layer_cache(&dst);
		}

		reason = "ok";
		return true;
	}

	float train_graph_network_datum(Network & net, const FloatBufferView & input, const FloatBufferView & truth)
	{
		if (not training_graph_engine_enabled(net))
		{
			const auto info = build_training_graph_info(net);
			const std::string reason =
				not info.reason.empty() ? info.reason : "Apple graph training is not configured";
			darknet_fatal_error(DARKNET_LOC,
				"Apple training requires the graph engine. graph_reason=%s",
				reason.c_str());
		}

		return train_training_graph_datum(net, input, truth);
	}

	void update_training_graph(Network & net)
	{
		if (not training_graph_engine_enabled(net))
		{
			const auto info = build_training_graph_info(net);
			const std::string reason =
				not info.reason.empty() ? info.reason : "Apple graph training is not configured";
			darknet_fatal_error(DARKNET_LOC,
				"Apple training updates require the graph engine. graph_reason=%s",
				reason.c_str());
		}

		const auto update_start = std::chrono::high_resolution_clock::now();
		const int iteration_num = (*net.seen) / (net.batch * net.subdivisions);
		const int update_batch = net.batch * net.subdivisions * get_sequence_value(net);
		const float rate = get_current_rate(net);
		TrainingOptimizerCommands commands;
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
			if (not training_update_gate_open(l.burnin_update,
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
			if (not training_optimizer_update_layer(net, l, update_batch, rate, commands, optimizer_reason))
			{
				darknet_fatal_error(DARKNET_LOC,
					"Apple GPU optimizer update failed at layer %d (%s): %s",
					i,
					layer_type.c_str(),
					optimizer_reason.c_str());
			}
			++gpu_update_count;
		}

		if (commands.encoder)
		{
			[commands.encoder endEncoding];
			commands.encoder = nil;
		}
		if (gpu_update_count > 0)
		{
			// Apple-only queue ordering: GPU optimizer kernels and the next
			// MPSGraph forward use the same training queue, so the next batch
			// observes the updated weights without blocking the CPU here.
			id<MTLCommandBuffer> submission = commands.command_buffer;
			commit_training_command_buffer(submission);
			record_training_update_command_buffer(commands.command_buffer);
		}

		const auto update_end = std::chrono::high_resolution_clock::now();
		const double update_encode_ms = std::chrono::duration<double, std::milli>(update_end - update_start).count();
		finalize_training_iteration(net, update_encode_ms, update_wait_ms);
	}

	TrainingGraphInfo build_training_graph_info(const Network & net)
	{
		return training_graph_info_snapshot(net);
	}

}
