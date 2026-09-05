#include "darknet_internal.hpp"
#include "apple_training_internal.hpp"
#include "metal_backend.hpp"
#include "apple_mps.hpp"

#include <chrono>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace Darknet::Apple::TrainingDetail
{
	using Detail::ObjcStrong;

	struct MpsSharedBufferView
	{
		id<MTLBuffer> buffer = nil;
		size_t offset = 0;
		size_t bytes = 0;

		explicit operator bool() const
		{
			return buffer != nil;
		}
	};

	static MpsSharedBufferView training_shared_buffer_view(const void *ptr, size_t bytes, TrainingBufferAccess access);
	static bool sync_training_shared_buffer_to_host(const void *ptr, size_t bytes);

	static MPSShape * shape_from_vector(const std::vector<int64_t> & shape)
	{
		NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:shape.size()];
		for (const auto dim : shape)
		{
			[values addObject:@(dim)];
		}
		return values;
	}

	enum class SharedBufferCoherency
	{
		kCoherent,
		kHostNewer,
		kGpuNewer,
	};

	struct SharedBufferEntry
	{
		ObjcStrong buffer;
		size_t bytes = 0;
		bool mirrored = false;
		SharedBufferCoherency coherency = SharedBufferCoherency::kCoherent;
	};

		MPSDataType data_type_for_bridge(const MPSGraphBufferBridge & bridge)
		{
		switch (bridge.data_type)
		{
			case MPSGraphBridgeDataType::kInt32: return MPSDataTypeInt32;
			case MPSGraphBridgeDataType::kFloat32:
				default: return MPSDataTypeFloat32;
			}
		}

	std::unordered_map<const void *, SharedBufferEntry> & shared_buffer_cache()
	{
		static std::unordered_map<const void *, SharedBufferEntry> cache;
		return cache;
	}

	std::mutex & shared_buffer_mutex()
	{
		static std::mutex mutex;
		return mutex;
	}

	bool access_reads(const TrainingBufferAccess access)
	{
		return access != TrainingBufferAccess::kWrite;
	}

	bool access_writes(const TrainingBufferAccess access)
	{
		return access != TrainingBufferAccess::kRead;
	}

	bool load_required_training_view(const void *ptr,
		const size_t bytes,
		const TrainingBufferAccess access,
		const std::string & label,
		TrainingBufferView & view,
		std::string & reason)
	{
		if (!ptr || bytes == 0)
		{
			reason = std::string(not label.empty() ? label : "buffer") + " is unavailable";
			return false;
		}
		if (!get_training_buffer_view(ptr, bytes, access, view))
		{
			reason = std::string(not label.empty() ? label : "buffer") + " does not have a Metal buffer view";
			return false;
		}
		return true;
	}

	bool dispatch_training_kernel_1d_with_encoder(const std::string & kernel_name,
		const size_t threads,
		const MetalDispatchBuffer *buffers,
		const size_t buffer_count,
		const MetalDispatchBytes *bytes,
		const size_t bytes_count,
		void *encoder_handle,
		std::string & reason)
	{
		if (kernel_name.empty() || threads == 0 || !encoder_handle)
		{
			reason = std::string("invalid dispatch state for ") +
				(not kernel_name.empty() ? kernel_name : "Apple training kernel");
			return false;
		}

		if (!metal_dispatch_1d_with_encoder(kernel_name,
			threads,
			std::min<size_t>(threads, 256),
			nullptr, 0,
			buffers, buffer_count,
			bytes, bytes_count,
			encoder_handle))
		{
			reason = std::string("failed to dispatch ") +
				(not kernel_name.empty() ? kernel_name : "Apple training kernel");
			return false;
		}
		return true;
	}

	bool upload_host_contents_if_needed(SharedBufferEntry & entry, const void *ptr, const TrainingBufferAccess access)
	{
		if (!access_reads(access) || entry.coherency != SharedBufferCoherency::kHostNewer)
		{
			return true;
		}
		if (entry.mirrored)
		{
			id<MTLBuffer> buffer = (id<MTLBuffer>)entry.buffer.get();
			void *contents = buffer ? [buffer contents] : nullptr;
			if (!contents || !ptr || entry.bytes == 0)
			{
				return false;
			}
			std::memcpy(contents, ptr, entry.bytes);
		}
		entry.coherency = SharedBufferCoherency::kCoherent;
		return true;
	}

	bool prepare_shared_buffer_entry_for_access(SharedBufferEntry & entry,
		const void *ptr,
		const size_t bytes,
		const TrainingBufferAccess access)
	{
		if (!entry.buffer.get() || entry.bytes != bytes ||
			!upload_host_contents_if_needed(entry, ptr, access))
		{
			return false;
		}
		if (access_writes(access))
		{
			entry.coherency = SharedBufferCoherency::kGpuNewer;
		}
		return true;
	}

	MpsSharedBufferView lookup_shared_buffer_view_locked(const void *ptr, const size_t bytes)
	{
		auto & cache = shared_buffer_cache();
		const auto it = cache.find(ptr);
		if (it == cache.end() || it->second.bytes != bytes || !it->second.buffer.get())
		{
			return {};
		}
		return {(id<MTLBuffer>)it->second.buffer.get(), 0, it->second.bytes};
	}

	bool sync_shared_buffer_entry_to_host(SharedBufferEntry & entry, const void *ptr)
	{
		if (entry.coherency != SharedBufferCoherency::kGpuNewer)
		{
			return true;
		}
		if (!entry.mirrored)
		{
			entry.coherency = SharedBufferCoherency::kCoherent;
			return true;
		}
		id<MTLBuffer> buffer = (id<MTLBuffer>)entry.buffer.get();
		if (!buffer || !ptr || entry.bytes == 0)
		{
			return false;
		}
		void *contents = [buffer contents];
		if (!contents)
		{
			return false;
		}
		std::memcpy(const_cast<void *>(ptr), contents, entry.bytes);
		entry.coherency = SharedBufferCoherency::kCoherent;
		return true;
	}

	size_t bytes_for_floats(const int count)
	{
		return count > 0 ? static_cast<size_t>(count) * sizeof(float) : 0;
	}

	bool sync_training_shared_buffer_entry(const void *ptr, const size_t bytes)
	{
		return sync_training_shared_buffer_to_host(ptr, bytes);
	}

	void mark_training_shared_buffer_host_dirty(const void *ptr)
	{
		note_training_shared_buffer_updated_from_host(ptr);
	}

	void mark_training_shared_buffer_gpu_dirty(const void *ptr)
	{
		note_training_shared_buffer_updated_from_gpu(ptr);
	}

	void erase_training_shared_buffer_entries(const void *ptr)
	{
		erase_training_shared_buffer_storage(ptr);
	}

	int layer_shared_channel_count(const Darknet::Layer & layer)
	{
		if (layer.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			return layer.n;
		}
		if (layer.type == Darknet::ELayerType::CONNECTED)
		{
			return layer.outputs;
		}
		return 0;
	}

	void sync_layer_training_cpu_inputs_to_host(const Darknet::Layer & layer,
		const bool sync_rolling_stats,
		const bool sync_parameters,
		const bool sync_optimizer_state)
	{
		if (sync_parameters && layer.weights && layer.nweights > 0)
		{
			sync_training_shared_buffer_entry(layer.weights, bytes_for_floats(layer.nweights));
		}

		if (sync_optimizer_state && layer.weight_updates && layer.nweights > 0)
		{
			sync_training_shared_buffer_entry(layer.weight_updates, bytes_for_floats(layer.nweights));
		}

		const int channel_count = layer_shared_channel_count(layer);
		if (channel_count <= 0)
		{
			return;
		}

		if (sync_parameters && layer.biases)
		{
			sync_training_shared_buffer_entry(layer.biases, bytes_for_floats(channel_count));
		}
		if (sync_parameters && layer.scales)
		{
			sync_training_shared_buffer_entry(layer.scales, bytes_for_floats(channel_count));
		}
		if (sync_optimizer_state && layer.bias_updates)
		{
			sync_training_shared_buffer_entry(layer.bias_updates, bytes_for_floats(channel_count));
		}
		if (sync_optimizer_state && layer.scale_updates)
		{
			sync_training_shared_buffer_entry(layer.scale_updates, bytes_for_floats(channel_count));
		}
		if (sync_rolling_stats && layer.batch_normalize)
		{
			if (layer.rolling_mean)
			{
				sync_training_shared_buffer_entry(layer.rolling_mean, bytes_for_floats(channel_count));
			}
			if (layer.rolling_variance)
			{
				sync_training_shared_buffer_entry(layer.rolling_variance, bytes_for_floats(channel_count));
			}
		}
	}

	using SharedBufferDirtyMarker = void (*)(const void *);

	void mark_layer_training_parameter_buffers(const Darknet::Layer & layer, const SharedBufferDirtyMarker mark_dirty)
	{
		mark_dirty(layer.weights);

		const int channel_count = layer_shared_channel_count(layer);
		if (channel_count <= 0)
		{
			return;
		}

		mark_dirty(layer.biases);
		mark_dirty(layer.scales);
	}

	void mark_layer_training_optimizer_buffers(const Darknet::Layer & layer, const SharedBufferDirtyMarker mark_dirty)
	{
		mark_dirty(layer.weight_updates);

		const int channel_count = layer_shared_channel_count(layer);
		if (channel_count <= 0)
		{
			return;
		}

		mark_dirty(layer.bias_updates);
		mark_dirty(layer.scale_updates);
	}

	bool is_detector_head_type(const Darknet::ELayerType type)
	{
		return type == Darknet::ELayerType::YOLO;
	}

	bool route_references_layer(const Darknet::Layer & l, const int layer_index)
	{
		if (l.type != Darknet::ELayerType::ROUTE || l.n <= 0 || !l.input_layers)
		{
			return false;
		}
		for (int i = 0; i < l.n; ++i)
		{
			if (l.input_layers[i] == layer_index)
			{
				return true;
			}
		}
		return false;
	}

	bool is_terminal_detector_head(const Darknet::Network & net, const int layer_index)
	{
		if (layer_index < 0 || layer_index >= net.n)
		{
			return false;
		}
		const auto & l = net.layers[layer_index];
		if (!is_detector_head_type(l.type))
		{
			return false;
		}
		if (layer_index == net.n - 1)
		{
			return true;
		}
		const auto & next = net.layers[layer_index + 1];
		return next.type == Darknet::ELayerType::ROUTE && !route_references_layer(next, layer_index);
	}

	void maybe_log_training_iteration_stats(const Darknet::Network & net)
	{
		auto & cfg_and_state = Darknet::CfgAndState::get();
		if (!cfg_and_state.is_verbose)
		{
			return;
		}

		const auto & stats = training_iteration_stats();
		*cfg_and_state.output
			<< "[Apple train] iter=" << stats.iteration
			<< " total=" << std::fixed << std::setprecision(2) << stats.total_batch_ms << "ms"
			<< " cpu_encode=" << (stats.forward_encode_ms + stats.backward_encode_ms +
				stats.backward_bridge_ms + stats.update_encode_ms) << "ms"
			<< " gpu_completion=" << (stats.forward_run_ms + stats.backward_run_ms + stats.update_wait_ms) << "ms"
			<< " forward=" << std::fixed << std::setprecision(2) << (stats.forward_encode_ms + stats.forward_run_ms) << "ms"
			<< " backward=" << (stats.backward_encode_ms + stats.backward_run_ms + stats.backward_bridge_ms) << "ms"
			<< " update=" << stats.update_encode_ms + stats.update_wait_ms << "ms"
			<< " submissions=" << stats.command_buffer_submissions
			<< " precision=" << training_precision_summary(net)
			<< " optimizer=" << training_optimizer_summary(net)
			<< " imgs/sec=" << stats.images_per_second
			<< std::endl;
	}

	AppleTrainingIterationStats & training_iteration_stats()
	{
		static AppleTrainingIterationStats stats;
		return stats;
	}

	void reset_training_buffer_state()
	{
		reset_gpu_yolo_loss_stats_buffer_cache();
		{
			std::scoped_lock lock(shared_buffer_mutex());
			shared_buffer_cache().clear();
		}
		training_iteration_stats() = AppleTrainingIterationStats{};
	}

	void invalidate_layer_training_storage(const Layer & layer)
	{
		erase_training_shared_buffer_entries(layer.weights);
		erase_training_shared_buffer_entries(layer.weight_updates);
		erase_training_shared_buffer_entries(layer.biases);
		erase_training_shared_buffer_entries(layer.bias_updates);
		erase_training_shared_buffer_entries(layer.scales);
		erase_training_shared_buffer_entries(layer.scale_updates);
		erase_training_shared_buffer_entries(layer.mean);
		erase_training_shared_buffer_entries(layer.variance);
		erase_training_shared_buffer_entries(layer.mean_delta);
		erase_training_shared_buffer_entries(layer.variance_delta);
		erase_training_shared_buffer_entries(layer.rolling_mean);
		erase_training_shared_buffer_entries(layer.rolling_variance);
		erase_training_shared_buffer_entries(layer.output);
		erase_training_shared_buffer_entries(layer.delta);
		erase_training_shared_buffer_entries(layer.activation_input);
		erase_training_shared_buffer_entries(layer.x);
		erase_training_shared_buffer_entries(layer.indexes);
	}

	void note_layer_training_parameters_updated_cpu(const Layer & layer)
	{
		mark_layer_training_parameter_buffers(layer, mark_training_shared_buffer_host_dirty);
	}

	void note_layer_training_parameters_updated_gpu(const Layer & layer)
	{
		mark_layer_training_parameter_buffers(layer, mark_training_shared_buffer_gpu_dirty);
	}

	void note_layer_training_optimizer_state_updated_cpu(const Layer & layer)
	{
		mark_layer_training_optimizer_buffers(layer, mark_training_shared_buffer_host_dirty);
	}

	void note_layer_training_optimizer_state_updated_gpu(const Layer & layer)
	{
		mark_layer_training_optimizer_buffers(layer, mark_training_shared_buffer_gpu_dirty);
	}

	void note_layer_training_running_stats_updated_cpu(const Layer & layer)
	{
		mark_training_shared_buffer_host_dirty(layer.rolling_mean);
		mark_training_shared_buffer_host_dirty(layer.rolling_variance);
	}

	void note_layer_training_running_stats_updated_gpu(const Layer & layer)
	{
		mark_training_shared_buffer_gpu_dirty(layer.rolling_mean);
		mark_training_shared_buffer_gpu_dirty(layer.rolling_variance);
	}

	static MpsSharedBufferView training_shared_buffer_view(const void *ptr,
		const size_t bytes,
		const TrainingBufferAccess access)
	{
		if (!training_mps_ready() || !ptr || bytes == 0)
		{
			return {};
		}

		{
			std::scoped_lock lock(shared_buffer_mutex());
			auto & cache = shared_buffer_cache();
			auto it = cache.find(ptr);
			if (it != cache.end())
			{
				if (!prepare_shared_buffer_entry_for_access(it->second, ptr, bytes, access))
				{
					return {};
				}
				return lookup_shared_buffer_view_locked(ptr, bytes);
			}
		}

		id<MTLDevice> device = training_mps_device();
		if (!device)
		{
			return {};
		}

		id<MTLBuffer> buffer = nil;
		if (Detail::no_copy_buffer_eligible(ptr, bytes))
		{
			buffer = [device newBufferWithBytesNoCopy:const_cast<void *>(ptr)
				length:bytes
				options:MTLResourceStorageModeShared
				deallocator:nil];
		}
		SharedBufferEntry entry;
		entry.bytes = bytes;
		if (buffer)
		{
			entry.buffer = ObjcStrong(buffer);
		}
		else
		{
			buffer = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
			if (!buffer)
			{
				return {};
			}
			void *contents = [buffer contents];
			if (!contents)
			{
				return {};
			}
			std::memcpy(contents, ptr, bytes);
			entry.buffer = ObjcStrong(buffer);
			entry.mirrored = true;

			auto & cfg_and_state = Darknet::CfgAndState::get();
			if (cfg_and_state.is_verbose)
			{
				*cfg_and_state.output
					<< "[Apple train] shared-buffer fallback"
					<< " ptr=" << ptr
					<< " bytes=" << bytes
					<< " mode=mirrored-shared"
					<< std::endl;
			}
		}

		MpsSharedBufferView view{};
		{
			std::scoped_lock lock(shared_buffer_mutex());
			auto & cache = shared_buffer_cache();
			auto [it, inserted] = cache.emplace(ptr, std::move(entry));
			(void)inserted;
			if (!prepare_shared_buffer_entry_for_access(it->second, ptr, bytes, access))
			{
				return {};
			}
			view = lookup_shared_buffer_view_locked(ptr, bytes);
		}

		return view;
	}

	static bool sync_training_shared_buffer_to_host(const void *ptr, const size_t bytes)
	{
		if (!ptr || bytes == 0)
		{
			return false;
		}

		std::scoped_lock lock(shared_buffer_mutex());
		auto & cache = shared_buffer_cache();
		const auto it = cache.find(ptr);
		if (it == cache.end() || it->second.bytes != bytes)
		{
			return false;
		}
		return sync_shared_buffer_entry_to_host(it->second, ptr);
	}

	void sync_all_training_shared_buffers_to_host()
	{
		std::scoped_lock lock(shared_buffer_mutex());
		auto & cache = shared_buffer_cache();
		for (auto & [ptr, entry] : cache)
		{
			(void)sync_shared_buffer_entry_to_host(entry, ptr);
		}
	}

	void note_training_shared_buffer_updated_from_host(const void *ptr)
	{
		if (!ptr)
		{
			return;
		}
		std::scoped_lock lock(shared_buffer_mutex());
		auto & cache = shared_buffer_cache();
		const auto it = cache.find(ptr);
		if (it != cache.end())
		{
			it->second.coherency = SharedBufferCoherency::kHostNewer;
		}
	}

	void note_training_shared_buffer_updated_from_gpu(const void *ptr)
	{
		if (!ptr)
		{
			return;
		}
		std::scoped_lock lock(shared_buffer_mutex());
		auto & cache = shared_buffer_cache();
		const auto it = cache.find(ptr);
		if (it != cache.end())
		{
			it->second.coherency = SharedBufferCoherency::kGpuNewer;
		}
	}

	void erase_training_shared_buffer_storage(const void *ptr)
	{
		if (!ptr)
		{
			return;
		}
		std::scoped_lock lock(shared_buffer_mutex());
		shared_buffer_cache().erase(ptr);
	}

	void begin_training_iteration(Network & net)
	{
		auto & stats = training_iteration_stats();
		if (net.current_subdivision == 0)
		{
			stats = {};
			stats.iteration = get_current_iteration(net);
		}
	}

	bool prepare_training_buffers(Network & net, std::string & reason)
	{
		(void)net;
		if (!training_mps_ready())
		{
			reason = "Apple MPS runtime is unavailable";
			return false;
		}
		wait_for_training_command_buffer();

		reason = "ok";
		return true;
	}

	void invalidate_training_buffers(Network * net)
	{
		if (!net)
		{
			return;
		}

		// Optimizer submissions are asynchronous and retain no-copy views into
		// Darknet arrays. They must finish before any cache or network storage is
		// released. The legacy MPS wait does not cover this command buffer.
		wait_for_training_update_command_buffer(nullptr);
		const std::string asynchronous_error = take_training_execution_error();
		if (!asynchronous_error.empty())
		{
			darknet_fatal_error(DARKNET_LOC, "Apple training synchronization failed: %s", asynchronous_error.c_str());
		}
		wait_for_training_command_buffer();
		// Mirrored shared buffers, unlike no-copy buffers, do not update their
		// Darknet host arrays automatically. Preserve GPU-newer parameters,
		// optimizer state, and running statistics while those arrays are still
		// valid, before discarding the buffer cache.
		sync_all_training_shared_buffers_to_host();
		reset_training_buffer_state();
	}

	bool get_training_buffer_view(const void *ptr,
		const size_t bytes,
		const TrainingBufferAccess access,
		TrainingBufferView & view)
	{
		if (!ptr || bytes == 0)
		{
			view = {};
			return false;
		}

		const auto shared_view = training_shared_buffer_view(ptr, bytes, access);
		if (!shared_view)
		{
			view = {};
			return false;
		}

		view = {(__bridge void *)shared_view.buffer, shared_view.offset, shared_view.bytes};
		return true;
	}

	bool make_training_graph_buffer_bridge(const TrainingBufferView & view,
		const std::vector<int64_t> & shape,
		MPSGraphBufferBridge & bridge,
		std::string & reason)
	{
		if (!view)
		{
			reason = "training buffer view is unavailable";
			return false;
		}
		bridge.buffer_handle = view.buffer_handle;
		bridge.offset = view.offset;
		bridge.bytes = view.bytes;
		bridge.shape = shape;
		reason = "ok";
		return true;
	}

	bool make_training_graph_buffer_bridge(const void *ptr,
		const size_t bytes,
		const std::vector<int64_t> & shape,
		const TrainingBufferAccess access,
		MPSGraphBufferBridge & bridge,
		std::string & reason)
	{
		TrainingBufferView view{};
		if (!get_training_buffer_view(ptr, bytes, access, view))
		{
			reason = "training tensor does not have a shared Metal buffer view";
			return false;
		}
		return make_training_graph_buffer_bridge(view, shape, bridge, reason);
	}

		bool make_training_graph_tensor_data_handle(const MPSGraphBufferBridge & bridge, void *& tensor_data_handle, std::string & reason)
		{

		id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)bridge.buffer_handle;
		if (!buffer)
		{
			reason = "buffer bridge Metal buffer handle is invalid";
			return false;
		}

		if (bridge.offset != 0)
		{
			reason = "MPSGraph tensor-data bridge requires zero buffer offsets";
			return false;
		}

		@autoreleasepool
		{
			MPSGraphTensorData *tensor_data = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
				shape:shape_from_vector(bridge.shape)
				dataType:data_type_for_bridge(bridge)];
			if (!tensor_data)
			{
				reason = "failed to create MPSGraphTensorData from Metal buffer";
				return false;
			}
			CFRetain((__bridge CFTypeRef)tensor_data);
			tensor_data_handle = (__bridge void *)tensor_data;
		}

			reason = "ok";
			return true;
		}

	void release_training_graph_tensor_data_handle(void *tensor_data_handle)
	{
		if (!tensor_data_handle)
			{
				return;
			}
			CFRelease((CFTypeRef)tensor_data_handle);
		}

	bool ensure_training_command_buffer(void *& command_buffer_handle)
	{
		if (!training_mps_ready())
		{
			return false;
		}

		id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)command_buffer_handle;
		if (!command_buffer)
		{
			command_buffer = create_training_command_buffer_handle();
			if (!command_buffer)
			{
				return false;
			}
			command_buffer.label = @"darknet.training.command_buffer";
			// The caller stores this Objective-C object as a raw void*. Retain it
			// until a commit helper releases the handle.
			command_buffer_handle = (void *)CFBridgingRetain(command_buffer);
			return true;
		}
		return true;
	}

	bool commit_training_command_buffer(void *& command_buffer_handle, uint64_t *commit_count)
	{

		id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)command_buffer_handle;
		if (!command_buffer)
		{
			return true;
		}

		if (commit_count)
		{
			++(*commit_count);
		}
		++training_iteration_stats().command_buffer_submissions;
		[command_buffer addCompletedHandler:^(id<MTLCommandBuffer> completed)
		{
			if (completed.status != MTLCommandBufferStatusCompleted)
			{
				std::string message = "Metal command buffer execution failed";
				if (completed.label.length > 0)
				{
					message += " (" + std::string(completed.label.UTF8String) + ")";
				}
				const std::string detail = error_description(completed.error);
				if (!detail.empty()) message += ": " + detail;
				record_training_execution_error(message);
			}
		}];
		[command_buffer commit];
		CFRelease((CFTypeRef)command_buffer_handle);
		command_buffer_handle = nullptr;
		return true;
	}

	bool commit_and_wait_training_command_buffer(void *& command_buffer_handle, double *wait_ms, uint64_t *commit_count)
	{

		id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)command_buffer_handle;
		if (!command_buffer)
		{
			return true;
		}

		const auto t1 = std::chrono::high_resolution_clock::now();
		if (commit_count)
		{
			++(*commit_count);
		}
		++training_iteration_stats().command_buffer_submissions;
		[command_buffer commit];
		[command_buffer waitUntilCompleted];
		const auto t2 = std::chrono::high_resolution_clock::now();
		if (wait_ms)
		{
			*wait_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
		}
		const bool completed = command_buffer.status == MTLCommandBufferStatusCompleted;
		if (!completed)
		{
			std::string message = "Metal command buffer execution failed";
			const std::string detail = error_description(command_buffer.error);
			if (!detail.empty()) message += ": " + detail;
			record_training_execution_error(message);
		}
		CFRelease((CFTypeRef)command_buffer_handle);
		command_buffer_handle = nullptr;
		return completed;
	}

	bool training_is_terminal_detector_head(const Network & net, int layer_index)
	{
		return is_terminal_detector_head(net, layer_index);
	}

	void sync_training_layers_to_host(Network & net, double *wait_ms, bool sync_rolling_stats, bool sync_parameters)
	{
		wait_for_training_update_command_buffer(wait_ms);
		const std::string asynchronous_error = take_training_execution_error();
		if (!asynchronous_error.empty())
		{
			darknet_fatal_error(DARKNET_LOC, "Apple training synchronization failed: %s", asynchronous_error.c_str());
		}
		const auto sync_start = std::chrono::high_resolution_clock::now();
		for (int i = 0; i < net.n; ++i)
		{
			const auto & l = net.layers[i];
			if (l.type == Darknet::ELayerType::CONVOLUTIONAL || l.type == Darknet::ELayerType::CONNECTED)
			{
				sync_layer_training_cpu_inputs_to_host(
					l,
					sync_rolling_stats,
					sync_parameters,
					l.train != 0);
			}
		}
		if (wait_ms)
		{
			*wait_ms += std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - sync_start).count();
		}
	}

	void finalize_training_iteration(Network & net, double update_encode_ms, double update_wait_ms)
	{
		auto & stats = training_iteration_stats();
		stats.update_encode_ms = update_encode_ms;
		stats.update_wait_ms = update_wait_ms;
		stats.total_batch_ms =
			stats.forward_encode_ms + stats.forward_run_ms +
			stats.backward_encode_ms + stats.backward_run_ms + stats.backward_bridge_ms +
			stats.update_encode_ms + stats.update_wait_ms;
		stats.images_per_second = stats.total_batch_ms > 0.0
			? (1000.0 * net.batch * net.subdivisions / stats.total_batch_ms)
			: 0.0;

		maybe_log_training_iteration_stats(net);
	}
}
