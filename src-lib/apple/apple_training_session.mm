#include "darknet_internal.hpp"
#include "apple_training_internal.hpp"

#include "apple_backend.hpp"
#include "apple_mps.hpp"
#include "darknet_layers.hpp"
#include "darknet_network.hpp"
#include "metal_backend.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace Darknet::Apple::TrainingDetail
{
	std::string activation_kind_label(const MPSGraphActivationKind activation)
	{
		switch (activation)
		{
			case MPSGraphActivationKind::kLinear: return "linear";
			case MPSGraphActivationKind::kLeaky: return "leaky";
			case MPSGraphActivationKind::kLogistic: return "logistic";
		}
		return "activation";
	}

	void append_shape_fingerprint(std::ostringstream & fingerprint, const std::vector<int64_t> & shape)
	{
		fingerprint << "{";
		for (size_t i = 0; i < shape.size(); ++i)
		{
			if (i > 0)
			{
				fingerprint << "x";
			}
			fingerprint << shape[i];
		}
		fingerprint << "}";
	}

	void append_ints_fingerprint(std::ostringstream & fingerprint, const std::vector<int> & values)
	{
		fingerprint << "{";
		for (size_t i = 0; i < values.size(); ++i)
		{
			if (i > 0)
			{
				fingerprint << ",";
			}
			fingerprint << values[i];
		}
		fingerprint << "}";
	}

	std::string training_graph_plan_fingerprint(const MPSGraphSegmentPlan & plan)
	{
		std::ostringstream fingerprint;
		fingerprint << "heads=";
		for (size_t i = 0; i < plan.terminal_output_layers.size(); ++i)
		{
			if (i > 0)
			{
				fingerprint << ",";
			}
			fingerprint << plan.terminal_output_layers[i];
		}
		fingerprint << " nodespec=";
		for (const auto & node : plan.nodes)
		{
			fingerprint
				<< node.darknet_layer_index
				<< ":" << node_kind_label(node.op_kind)
				<< ":out";
			append_shape_fingerprint(fingerprint, node.output_shape_nchw);
			fingerprint << ":in";
			append_ints_fingerprint(fingerprint, node.input_node_indices);
			fingerprint
				<< ":segin=" << node.segment_input
				<< ":segyolo=" << node.segment_output_to_yolo
				<< ":bn=" << node.batch_normalize
				<< ":act=" << activation_kind_label(node.activation)
				<< ":khw=" << node.kernel_h << "x" << node.kernel_w
				<< ":stride=" << node.stride_y << "x" << node.stride_x
				<< ":pad=" << node.pad_top << "," << node.pad_bottom << "," << node.pad_left << "," << node.pad_right
				<< ":groups=" << node.groups
				<< ":slice=" << node.slice_channel_offset << "+" << node.slice_channel_count
				<< ":concat_axis=" << node.route_concat_axis
				<< ":up=" << node.upsample_scale
				<< ":tensors=";
			for (const auto & tensor_plan : node.tensors)
			{
				fingerprint
					<< "[" << static_cast<int>(tensor_plan.binding_kind)
					<< ":owner=" << tensor_plan.owner_layer_index
					<< ":src=" << tensor_plan.source_layer_index
					<< ":" << tensor_data_type_label(tensor_plan.data_type)
					<< ":bytes=" << tensor_plan.bytes
					<< ":shape";
				append_shape_fingerprint(fingerprint, tensor_plan.shape_nchw);
				fingerprint
					<< ":fr=" << tensor_plan.forward_result
					<< ":mat=" << tensor_plan.materialize_to_darknet_buffer
					<< "]";
			}
			fingerprint << ";";
		}
		fingerprint << " boundary=";
		for (const auto & tensor_plan : plan.boundary_tensors)
		{
			fingerprint
				<< "[" << static_cast<int>(tensor_plan.binding_kind)
				<< ":owner=" << tensor_plan.owner_layer_index
				<< ":src=" << tensor_plan.source_layer_index
				<< ":" << tensor_data_type_label(tensor_plan.data_type)
				<< ":bytes=" << tensor_plan.bytes
				<< ":shape";
			append_shape_fingerprint(fingerprint, tensor_plan.shape_nchw);
			fingerprint
				<< ":fr=" << tensor_plan.forward_result
				<< ":mat=" << tensor_plan.materialize_to_darknet_buffer
				<< "]";
		}
		return fingerprint.str();
	}

	struct MPSGraphState
	{
		bool enabled = false;
		std::string reason;
	};

	namespace cache_detail
	{
		struct CacheEntry
		{
			std::shared_ptr<const MPSGraphCompiledExecutables> compiled;
		};

		void release_cache_entry(CacheEntry & entry)
		{
			entry.compiled.reset();
		}

	}

		/** All state owned by the one supported Apple training network.
		 *
		 * The Network address alone is not an identity: Darknet may reuse it after
		 * resize or destruction. The stable details/layers addresses prevent stale
		 * executables and buffers from being reused in that case.
		 */
	struct TrainingSession
	{
		std::recursive_mutex mutex;
		const Network *network = nullptr;
		const void *details_identity = nullptr;
		const Layer *layers_identity = nullptr;
		MPSGraphState graph;
		MPSGraphSegmentPlan plan;
		cache_detail::CacheEntry cache;
		Detail::ObjcStrong pending_update_command_buffer;
		std::string asynchronous_error;

		bool matches(const Network *candidate) const
		{
			return candidate and network == candidate and
				details_identity == candidate->details and
				layers_identity == candidate->layers;
		}

		void bind(const Network & candidate)
		{
			network = &candidate;
			details_identity = candidate.details;
			layers_identity = candidate.layers;
			graph = {};
			plan = {};
			asynchronous_error.clear();
		}

		void release_owned_resources()
		{
			// Pending work is drained before callers acquire mutex.  Waiting here
			// would deadlock when a failed command buffer's completion handler tries
			// to record its error through the same session mutex.
			pending_update_command_buffer = {};
			cache_detail::release_cache_entry(cache);
			graph = {};
			plan = {};
			asynchronous_error.clear();
			network = nullptr;
			details_identity = nullptr;
			layers_identity = nullptr;
		}

		~TrainingSession()
		{
			// Static destruction has no outer session lock.  Drain any final command
			// buffer before releasing no-copy storage referenced by it.
			if (id<MTLCommandBuffer> command_buffer = (id<MTLCommandBuffer>)pending_update_command_buffer.get())
			{
				[command_buffer waitUntilCompleted];
			}
			release_owned_resources();
		}
	};

	TrainingSession & training_session()
	{
		static TrainingSession session;
		return session;
	}

	std::string error_description(NSError *error)
	{
		if (not error)
		{
			return {};
		}
		NSString *description = error.localizedDescription ?: error.description;
		return description ? std::string(description.UTF8String) : std::string("unknown NSError");
	}

	void record_training_execution_error(const std::string & message)
	{
		if (message.empty())
		{
			return;
		}
		auto & session = training_session();
		std::scoped_lock lock(session.mutex);
		if (session.asynchronous_error.empty())
		{
			session.asynchronous_error = message;
		}
	}

	std::string take_training_execution_error()
	{
		auto & session = training_session();
		std::scoped_lock lock(session.mutex);
		std::string message = std::move(session.asynchronous_error);
		session.asynchronous_error.clear();
		return message;
	}

	void record_training_update_command_buffer(id<MTLCommandBuffer> command_buffer)
	{
		if (not command_buffer)
		{
			return;
		}
		auto & session = training_session();
		std::scoped_lock lock(session.mutex);
		session.pending_update_command_buffer = Detail::ObjcStrong(command_buffer);
	}

	double wait_for_training_update_command_buffer()
	{
		Detail::ObjcStrong pending_command_buffer;
		{
			auto & session = training_session();
			std::scoped_lock lock(session.mutex);
			pending_command_buffer = std::move(session.pending_update_command_buffer);
		}
		id<MTLCommandBuffer> buffer = (id<MTLCommandBuffer>)pending_command_buffer.get();
		if (not buffer)
		{
			return 0.0;
		}
		const auto start = std::chrono::high_resolution_clock::now();
		[buffer waitUntilCompleted];
		if (buffer.status != MTLCommandBufferStatusCompleted)
		{
			std::string message = "asynchronous optimizer command buffer failed";
			const std::string detail = error_description(buffer.error);
			if (not detail.empty()) message += ": " + detail;
			record_training_execution_error(message);
		}
		return std::chrono::duration<double, std::milli>(
			std::chrono::high_resolution_clock::now() - start).count();
	}

	MPSGraphExecutableExecutionDescriptor *make_execution_descriptor(
		NSString *label,
		const NSUInteger expected_result_count,
		const bool wait_until_completed)
	{
		MPSGraphExecutableExecutionDescriptor *descriptor = [[MPSGraphExecutableExecutionDescriptor alloc] init];
		descriptor.waitUntilCompleted = wait_until_completed ? YES : NO;
		NSString *captured_label = [label copy];
		descriptor.completionHandler = ^(NSArray<MPSGraphTensorData *> *completion_results, NSError *error)
			{
			if (error)
			{
				record_training_execution_error(
					std::string(captured_label.UTF8String) + ": " + error_description(error));
				return;
			}
			if (completion_results.count != expected_result_count)
			{
				record_training_execution_error(
					std::string(captured_label.UTF8String) + ": expected " +
					std::to_string(expected_result_count) + " result(s), received " +
					std::to_string(completion_results.count));
			}
		};
		return descriptor;
	}

	bool validate_execution_results(NSArray<MPSGraphTensorData *> *results,
		const NSUInteger expected_result_count,
		const std::string & label,
		std::string & reason)
	{
		if (results.count != expected_result_count)
		{
			reason = std::string(label) + " returned " + std::to_string(results.count) +
				" result(s), expected " + std::to_string(expected_result_count);
			return false;
		}
		const std::string asynchronous_error = take_training_execution_error();
		if (not asynchronous_error.empty())
		{
			reason = asynchronous_error;
			return false;
		}
		return true;
	}

	bool validate_bound_execution_results(NSArray<MPSGraphTensorData *> *results,
		NSArray<MPSGraphTensorData *> *bound_results,
		const std::string & label,
		std::string & reason)
	{
		return validate_execution_results(results, bound_results.count, label, reason);
	}

	MPSGraphBridgeDataType bridge_data_type_for_tensor_plan(const MPSGraphTensorPlan & tensor_plan)
	{
		switch (tensor_plan.data_type)
		{
			case MPSGraphTensorDataType::kInt32: return MPSGraphBridgeDataType::kInt32;
			case MPSGraphTensorDataType::kFloat32:
			default:
				return MPSGraphBridgeDataType::kFloat32;
		}
	}

	bool ensure_tensor_storage(MPSGraphCompiledExecutables & compiled, const MPSGraphTensorPlan & tensor_plan, std::string & reason)
	{
		if (tensor_plan.materialize_to_darknet_buffer)
		{
			reason = "ok";
			return true;
		}

		if (tensor_plan.storage_index == std::numeric_limits<size_t>::max())
		{
			reason = "MPSGraph tensor has no compiled storage index: " + tensor_debug_label(tensor_plan);
			return false;
		}
		if (compiled.tensor_storage.size() <= tensor_plan.storage_index)
		{
			compiled.tensor_storage.resize(tensor_plan.storage_index + 1);
		}
		auto & storage = compiled.tensor_storage[tensor_plan.storage_index];
		if (storage.buffer.get())
		{
			if (storage.bridge.bytes != std::max<size_t>(1, tensor_plan.bytes) or
				storage.bridge.data_type != bridge_data_type_for_tensor_plan(tensor_plan) or
				storage.bridge.shape != tensor_plan.shape_nchw)
			{
				reason = "MPSGraph tensor storage identity has inconsistent metadata: " + tensor_debug_label(tensor_plan);
				return false;
			}
			reason = "ok";
			return true;
		}

		id<MTLDevice> device = (__bridge id<MTLDevice>)runtime_metal_device_handle();
		if (not device)
		{
			reason = "Metal device is unavailable for MPSGraph tensor storage";
			return false;
		}

		const NSUInteger length = static_cast<NSUInteger>(std::max<size_t>(1, tensor_plan.bytes));
		id<MTLBuffer> buffer = [device newBufferWithLength:length options:MTLResourceStorageModePrivate];
		if (not buffer)
		{
			reason = "failed to allocate private MPSGraph tensor storage for " + tensor_debug_label(tensor_plan);
			return false;
		}

		storage.buffer = Detail::ObjcStrong(buffer);
		storage.bridge.buffer_handle = nil;
		storage.bridge.offset = 0;
		storage.bridge.bytes = static_cast<size_t>(length);
		storage.bridge.data_type = bridge_data_type_for_tensor_plan(tensor_plan);
		storage.bridge.shape = tensor_plan.shape_nchw;
		reason = "ok";
		return true;
	}

	bool uses_graph_owned_storage(const MPSGraphTensorPlan & tensor_plan)
	{
		if (tensor_plan.materialize_to_darknet_buffer)
		{
			return false;
		}

		switch (tensor_plan.binding_kind)
		{
			case MPSGraphTensorBindingKind::kLayerOutput:
			case MPSGraphTensorBindingKind::kNetworkInputDelta:
			case MPSGraphTensorBindingKind::kWeightGrad:
			case MPSGraphTensorBindingKind::kBiasGrad:
			case MPSGraphTensorBindingKind::kBatchNormScaleGrad:
			case MPSGraphTensorBindingKind::kConvSourceTape:
			case MPSGraphTensorBindingKind::kBatchNormInputTape:
			case MPSGraphTensorBindingKind::kBatchMeanTape:
			case MPSGraphTensorBindingKind::kBatchVarianceTape:
			case MPSGraphTensorBindingKind::kBatchNormOutputTape:
			case MPSGraphTensorBindingKind::kActivationTape:
			case MPSGraphTensorBindingKind::kMaxPoolIndicesTape:
				return true;
			default:
				return false;
		}
	}

	bool resolve_forward_feed_pointer(const Network & net,
		const Darknet::NetworkState & state,
		const MPSGraphTensorPlan & tensor_plan,
		TrainingHostBuffer & ptr,
		std::string & reason)
	{
		ptr = nullptr;

		switch (tensor_plan.binding_kind)
		{
			case MPSGraphTensorBindingKind::kNetworkInput:
				ptr = state.input;
				break;
			case MPSGraphTensorBindingKind::kWeights:
				ptr = net.layers[tensor_plan.owner_layer_index].weights;
				break;
			case MPSGraphTensorBindingKind::kBias:
				ptr = net.layers[tensor_plan.owner_layer_index].biases;
				break;
			case MPSGraphTensorBindingKind::kBatchNormScale:
				ptr = net.layers[tensor_plan.owner_layer_index].scales;
				break;
			default:
				reason = "unsupported forward feed binding " + tensor_debug_label(tensor_plan);
				return false;
		}
		if (not ptr)
		{
			reason = "forward feed buffer is unavailable for " + tensor_debug_label(tensor_plan);
			return false;
		}
		reason = "ok";
		return true;
	}

	bool resolve_forward_result_pointer(Network & net,
		const MPSGraphTensorPlan & tensor_plan,
		TrainingHostBuffer & ptr,
		std::string & reason)
	{
		ptr = nullptr;

		Layer & layer = net.layers[tensor_plan.owner_layer_index];
		switch (tensor_plan.binding_kind)
		{
			case MPSGraphTensorBindingKind::kLayerOutput:
				ptr = layer.output;
				break;
			case MPSGraphTensorBindingKind::kYoloBoundaryOutput:
				ptr = layer.output;
				break;
			case MPSGraphTensorBindingKind::kConvSourceTape:
				if (tensor_plan.source_layer_index >= 0)
				{
					ptr = net.layers[tensor_plan.source_layer_index].output;
				}
				break;
			case MPSGraphTensorBindingKind::kBatchNormInputTape:
				ptr = layer.x;
				break;
			case MPSGraphTensorBindingKind::kBatchMeanTape:
				ptr = layer.mean;
				break;
			case MPSGraphTensorBindingKind::kBatchVarianceTape:
				ptr = layer.variance;
				break;
			case MPSGraphTensorBindingKind::kBatchNormOutputTape:
				ptr = layer.x_norm;
				break;
			case MPSGraphTensorBindingKind::kMaxPoolIndicesTape:
				ptr = layer.indexes;
				break;
			default:
				reason = "unsupported forward result binding " + tensor_debug_label(tensor_plan);
				return false;
		}
		if (not ptr)
		{
			reason = "forward result buffer is unavailable for " + tensor_debug_label(tensor_plan);
			return false;
		}
		reason = "ok";
		return true;
	}

	bool bridge_for_forward_tensor(const TrainingHostBuffer ptr,
		const size_t bytes,
		const MPSGraphTensorPlan & tensor_plan,
		const TrainingBufferAccess access,
		MPSGraphBufferBridge & bridge,
		std::string & reason)
	{
		if (not make_training_graph_buffer_bridge(ptr, bytes, tensor_plan.shape_nchw, access, bridge, reason))
		{
			return false;
		}
		bridge.data_type = bridge_data_type_for_tensor_plan(tensor_plan);
		if (bridge.bytes < tensor_plan.bytes)
		{
			reason = "bridge for " + tensor_debug_label(tensor_plan) + " is smaller than the tensor plan";
			return false;
		}
		reason = "ok";
		return true;
	}

	bool resolve_backward_feed_pointer(Network & net,
		const float *network_input,
		const MPSGraphTensorPlan & tensor_plan,
		TrainingHostBuffer & ptr,
		std::string & reason)
	{
		ptr = nullptr;

		Layer & layer = net.layers[tensor_plan.owner_layer_index];
		switch (tensor_plan.binding_kind)
		{
			case MPSGraphTensorBindingKind::kNetworkInput:
				ptr = network_input;
				break;
			case MPSGraphTensorBindingKind::kLayerOutput:
			case MPSGraphTensorBindingKind::kYoloBoundaryOutput:
				ptr = layer.output;
				break;
			case MPSGraphTensorBindingKind::kYoloBoundaryDelta:
				ptr = layer.delta;
				break;
			case MPSGraphTensorBindingKind::kConvSourceTape:
				if (tensor_plan.source_layer_index >= 0)
				{
					ptr = net.layers[tensor_plan.source_layer_index].output;
				}
				break;
			case MPSGraphTensorBindingKind::kWeights:
				ptr = layer.weights;
				break;
			case MPSGraphTensorBindingKind::kWeightGrad:
				ptr = layer.weight_updates;
				break;
			case MPSGraphTensorBindingKind::kBiasGrad:
				ptr = layer.bias_updates;
				break;
			case MPSGraphTensorBindingKind::kBatchNormScale:
				ptr = layer.scales;
				break;
			case MPSGraphTensorBindingKind::kBatchNormScaleGrad:
				ptr = layer.scale_updates;
				break;
			case MPSGraphTensorBindingKind::kBatchNormInputTape:
				ptr = layer.x;
				break;
			case MPSGraphTensorBindingKind::kBatchMeanTape:
				ptr = layer.mean;
				break;
			case MPSGraphTensorBindingKind::kBatchVarianceTape:
				ptr = layer.variance;
				break;
			case MPSGraphTensorBindingKind::kBatchNormOutputTape:
				ptr = layer.x_norm;
				break;
			case MPSGraphTensorBindingKind::kMaxPoolIndicesTape:
				ptr = layer.indexes;
				break;
			default:
				reason = "unsupported backward feed binding " + tensor_debug_label(tensor_plan);
				return false;
		}
		if (not ptr)
		{
			reason = "backward feed buffer is unavailable for " + tensor_debug_label(tensor_plan);
			return false;
		}
		reason = "ok";
		return true;
	}

	bool resolve_backward_result_pointer(Network & net,
		const MPSGraphTensorPlan & tensor_plan,
		TrainingHostBuffer & ptr,
		std::string & reason)
	{
		ptr = nullptr;

		if (tensor_plan.binding_kind == MPSGraphTensorBindingKind::kNetworkInputDelta)
		{
			reason = "network-input delta must use graph-owned tensor storage";
			return false;
		}
		Layer & layer = net.layers[tensor_plan.owner_layer_index];
		switch (tensor_plan.binding_kind)
		{
			case MPSGraphTensorBindingKind::kWeightGrad:
				ptr = layer.weight_updates;
				break;
			case MPSGraphTensorBindingKind::kBiasGrad:
				ptr = layer.bias_updates;
				break;
			case MPSGraphTensorBindingKind::kBatchNormScaleGrad:
				ptr = layer.scale_updates;
				break;
			case MPSGraphTensorBindingKind::kLayerDelta:
				ptr = layer.delta;
				break;
			default:
				reason = "unsupported backward result binding " + tensor_debug_label(tensor_plan);
				return false;
		}
		if (not ptr)
		{
			reason = "backward result buffer is unavailable for " + tensor_debug_label(tensor_plan);
			return false;
		}
		reason = "ok";
		return true;
	}

	struct TensorStorageIdentity
	{
		MPSGraphTensorBindingKind binding_kind;
		int owner_layer_index;
		int source_layer_index;
		MPSGraphTensorDataType data_type;

		bool operator==(const TensorStorageIdentity & other) const
		{
			return binding_kind == other.binding_kind and
				owner_layer_index == other.owner_layer_index and
				source_layer_index == other.source_layer_index and
				data_type == other.data_type;
		}
	};

	struct TensorStorageIdentityHash
	{
		size_t operator()(const TensorStorageIdentity & identity) const
		{
			size_t value = static_cast<size_t>(identity.binding_kind);
			value = value * 1315423911u + static_cast<size_t>(identity.owner_layer_index + 1);
			value = value * 1315423911u + static_cast<size_t>(identity.source_layer_index + 1);
			return value * 31u + static_cast<size_t>(identity.data_type);
		}
	};

	struct TensorStorageSlot
	{
		size_t index;
		size_t bytes;
		std::vector<int64_t> shape;
	};

	bool assign_tensor_storage_indices(MPSGraphExecutableBuildResult & build, std::string & reason)
	{
		std::unordered_map<TensorStorageIdentity, TensorStorageSlot, TensorStorageIdentityHash> slots;
		size_t next_index = 0;
		const auto assign = [&](std::vector<MPSGraphTensorPlan> & plans)
		{
			for (auto & plan : plans)
			{
				if (plan.materialize_to_darknet_buffer)
				{
					continue;
				}
				const TensorStorageIdentity identity{
					plan.binding_kind,
					plan.owner_layer_index,
					plan.source_layer_index,
					plan.data_type,
				};
				auto [it, inserted] = slots.try_emplace(identity,
					TensorStorageSlot{next_index, plan.bytes, plan.shape_nchw});
				if (inserted)
				{
					++next_index;
				}
				else if (it->second.bytes != plan.bytes or it->second.shape != plan.shape_nchw)
				{
					reason = "inconsistent compiled tensor identity for " + tensor_debug_label(plan);
					return false;
				}
				plan.storage_index = it->second.index;
			}
			return true;
		};
		return assign(build.forward_feed_plans) and
			assign(build.forward_result_plans) and
			assign(build.backward_feed_plans) and
			assign(build.backward_result_plans);
	}

	bool ensure_training_graph_cache(Network & net, const MPSGraphSegmentPlan & plan, std::string & reason)
	{
		auto & session = training_session();
		std::scoped_lock lock(session.mutex);
		if (not session.matches(&net))
		{
			reason = "Apple training session identity changed during graph compilation";
			return false;
		}
		auto & entry = session.cache;
		const auto plan_key = training_graph_plan_fingerprint(plan);
		if (entry.compiled and entry.compiled->fingerprint == plan_key)
		{
			reason = "ok";
			return true;
		}

		cache_detail::release_cache_entry(entry);
		auto build = build_training_graph_executables(net, plan);
		const bool cache_ready = build.forward_ready and build.backward_ready;
		std::string compile_reason = build.reason.empty()
			? (cache_ready ? "ok" : "MPSGraph executable build did not produce both executables")
			: build.reason;
		if (not cache_ready or not assign_tensor_storage_indices(build, compile_reason))
		{
			reason = compile_reason;
			return false;
		}

		auto compiled = std::make_shared<MPSGraphCompiledExecutables>();
		compiled->forward.executable = std::move(build.forward_executable);
		compiled->forward.feed_plans = std::move(build.forward_feed_plans);
		compiled->forward.result_plans = std::move(build.forward_result_plans);
		compiled->backward.executable = std::move(build.backward_executable);
		compiled->backward.feed_plans = std::move(build.backward_feed_plans);
		compiled->backward.result_plans = std::move(build.backward_result_plans);
		compiled->fingerprint = plan_key;
		compiled->reason = compile_reason;
		for (const auto & tensor_plan : compiled->forward.result_plans)
		{
			std::string storage_reason;
			if (not ensure_tensor_storage(*compiled, tensor_plan, storage_reason))
			{
				reason = storage_reason;
				return false;
			}
		}
		for (const auto & tensor_plan : compiled->backward.result_plans)
		{
			std::string storage_reason;
			if (not ensure_tensor_storage(*compiled, tensor_plan, storage_reason))
			{
				reason = storage_reason;
				return false;
			}
		}
		entry.compiled = std::move(compiled);
		reason = compile_reason;
		return true;
	}

	MPSGraphExecutableBundle training_graph_executable_bundle(const Network *net, const bool backward)
	{
		if (not net)
		{
			return {};
		}
		auto & session = training_session();
		std::scoped_lock lock(session.mutex);
		return session.matches(net) ? MPSGraphExecutableBundle{session.cache.compiled, backward}
									: MPSGraphExecutableBundle{};
	}

	bool training_graph_tensor_storage_bridge(const MPSGraphExecutableBundle & bundle,
		const MPSGraphTensorPlan & tensor_plan,
		MPSGraphBufferBridge & bridge,
		std::string & reason)
	{
		const auto & compiled = bundle.compiled;
		if (not compiled or tensor_plan.storage_index >= compiled->tensor_storage.size())
		{
			reason = "MPSGraph tensor storage is unavailable for " + tensor_debug_label(tensor_plan);
			return false;
		}
		const auto & storage = compiled->tensor_storage[tensor_plan.storage_index];
		if (not storage.buffer.get())
		{
			reason = "MPSGraph tensor storage is unavailable for " + tensor_debug_label(tensor_plan);
			return false;
		}
		bridge = storage.bridge;
		bridge.buffer_handle = (id<MTLBuffer>)storage.buffer.get();
		reason = "ok";
		return true;
	}

	TrainingGraphInfo training_graph_info_snapshot(const Network & net)
	{
		const auto & runtime = initialize_runtime();
		TrainingGraphInfo info;
		info.runtime_ready = runtime.metal_available and runtime.metal_command_queue_available;
		{
			auto & session = training_session();
			std::scoped_lock lock(session.mutex);
			if (session.matches(&net))
			{
				info.graph_supported = session.graph.enabled;
				info.reason = session.graph.reason;
			}
			else
			{
				info.reason = "Apple training graph is not configured for this network";
			}
		}
		if (not info.runtime_ready and (info.reason.empty() or info.reason == "ok"))
		{
			info.reason = runtime.metal_available ? "Apple Metal command queue unavailable" :
						(runtime.reason.empty() ? "Apple runtime unavailable" : runtime.reason);
		}
		else if (info.reason.empty())
		{
			info.reason = "ok";
		}
		return info;
	}

	bool training_graph_runtime_available_impl(std::string & reason)
	{
		const auto & runtime = initialize_runtime();
		const bool ready = runtime.metal_available and runtime.metal_command_queue_available;
		reason = ready ? "ok" : (runtime.reason.empty() ? "Metal runtime unavailable for MPSGraph" : runtime.reason);
		return ready;
	}

	bool configure_training_graph_engine(Network & net, std::string & reason)
	{
		// Metal waits include completion handlers.  Drain outside session.mutex
		// because a failure handler records its error under that mutex.
		wait_for_training_update_command_buffer();
		auto & session = training_session();
		std::scoped_lock lock(session.mutex);
		if (session.network and not session.matches(&net))
		{
			// A matching Network address with different owned storage is a reused
			// object, not the active session. Drain it before binding the new one.
			if (session.network == &net)
			{
				session.release_owned_resources();
				reset_training_buffer_state();
			}
			else
			{
				reason = "Apple graph training already has an active training network";
				return false;
			}
		}
		if (not session.network)
		{
			session.bind(net);
		}
		auto & state = session.graph;
		state = {};

		session.plan = build_training_graph_plan(net);
		const auto & plan = session.plan;
		if (not plan.supported)
		{
			state.reason = plan.reason;
			reason = state.reason;
			return false;
		}
		std::string host_consumer_reason;
		if (not host_parameter_consumers_supported_by_current_contract(net, host_consumer_reason))
		{
			state.reason = host_consumer_reason;
			state.enabled = false;
			reason = state.reason;
			return false;
		}
		std::string runtime_reason;
		if (not training_graph_runtime_available_impl(runtime_reason))
		{
			state.reason = runtime_reason;
			reason = state.reason;
			return false;
		}
		std::string gpu_loss_reason;
		if (not gpu_yolo_loss_supported_by_current_contract(net, gpu_loss_reason))
		{
			state.reason = gpu_loss_reason;
			state.enabled = false;
			reason = state.reason;
			return false;
		}

		std::string cache_reason;
		if (not ensure_training_graph_cache(net, plan, cache_reason))
		{
			state.reason = cache_reason;
			state.enabled = false;
			reason = state.reason;
			return false;
		}

		state.enabled = true;
		state.reason = "ok";
		reason = state.reason;
		return true;
	}

	bool training_graph_engine_enabled(const Network & net)
	{
		auto & session = training_session();
		std::scoped_lock lock(session.mutex);
		return session.matches(&net) and session.graph.enabled;
	}

	bool run_training_graph_forward(Network & net,
		const Darknet::NetworkState & state,
		std::string & reason,
		__strong id<MTLCommandBuffer> & encoded_command_buffer_handle, const bool encode_to_command_buffer)
	{
		const auto bundle = training_graph_executable_bundle(&net, false);
		if (not bundle.available())
		{
			reason = bundle.reason().empty() ? "MPSGraph forward executable is unavailable" : bundle.reason();
			return false;
		}
		if (bundle.feed_plans().empty() or bundle.result_plans().empty())
		{
			reason = "MPSGraph forward bundle is missing feed or result metadata";
			return false;
		}

		id<MTLCommandQueue> command_queue = training_mps_command_queue();
		if (not command_queue)
		{
			reason = "Metal command queue is unavailable for MPSGraph forward execution";
			return false;
		}
		encoded_command_buffer_handle = nullptr;

		@autoreleasepool
		{
			MPSGraphExecutable *forward_executable = bundle.executable();
			NSMutableArray<MPSGraphTensorData *> *inputs = [NSMutableArray arrayWithCapacity:bundle.feed_plans().size()];
			NSMutableArray<MPSGraphTensorData *> *results = [NSMutableArray arrayWithCapacity:bundle.result_plans().size()];

			for (const auto & tensor_plan : bundle.feed_plans())
			{
				TrainingHostBuffer ptr;

				MPSGraphBufferBridge bridge;
				std::string bridge_reason;
				if (not resolve_forward_feed_pointer(net, state, tensor_plan, ptr, bridge_reason) or
					not bridge_for_forward_tensor(ptr, tensor_plan.bytes, tensor_plan, TrainingBufferAccess::kRead, bridge, bridge_reason))
				{
					reason = bridge_reason;
					return false;
				}
				MPSGraphTensorData *tensor_data_handle = nil;
				if (not make_training_graph_tensor_data(bridge, tensor_data_handle, bridge_reason))
				{
					reason = bridge_reason;
					return false;
				}

				[inputs addObject:tensor_data_handle];
			}

			for (const auto & tensor_plan : bundle.result_plans())
			{
				MPSGraphBufferBridge bridge;
				std::string bridge_reason;
				if (uses_graph_owned_storage(tensor_plan))
				{
					if (not training_graph_tensor_storage_bridge(bundle, tensor_plan, bridge, bridge_reason))
					{
						reason = bridge_reason;
						return false;
					}
				}
				else
				{
					TrainingHostBuffer ptr;

					if (not resolve_forward_result_pointer(net, tensor_plan, ptr, bridge_reason) or
						not bridge_for_forward_tensor(ptr, tensor_plan.bytes, tensor_plan, TrainingBufferAccess::kWrite, bridge, bridge_reason))
					{
						reason = bridge_reason;
						return false;
					}
				}
				MPSGraphTensorData *tensor_data_handle = nil;
				if (not make_training_graph_tensor_data(bridge, tensor_data_handle, bridge_reason))
				{
					reason = bridge_reason;
					return false;
				}

				[results addObject:tensor_data_handle];
			}

			double forward_encode_ms = 0.0;
			double forward_run_ms = 0.0;
			MPSGraphExecutableExecutionDescriptor *execution_descriptor = make_execution_descriptor(
				@"apple_training_forward",
				results.count,
				not encode_to_command_buffer);
			if (encode_to_command_buffer)
			{
				MPSCommandBuffer *mps_command_buffer = [MPSCommandBuffer commandBufferFromCommandQueue:command_queue];
				if (not mps_command_buffer)
				{
					reason = "failed to allocate MPSGraph command buffer";
					return false;
				}
				const auto start = std::chrono::high_resolution_clock::now();
				NSArray<MPSGraphTensorData *> *run_results = [forward_executable encodeToCommandBuffer:mps_command_buffer
					inputsArray:inputs
					resultsArray:results
					executionDescriptor:execution_descriptor];
				const auto end = std::chrono::high_resolution_clock::now();
				if (not validate_bound_execution_results(run_results, results, "MPSGraph forward encode", reason))
				{
					return false;
				}
				forward_encode_ms = std::chrono::duration<double, std::milli>(end - start).count();
				encoded_command_buffer_handle = mps_command_buffer;
			}
			else
			{
				const auto start = std::chrono::high_resolution_clock::now();
				NSArray<MPSGraphTensorData *> *run_results = [forward_executable runWithMTLCommandQueue:command_queue
					inputsArray:inputs
					resultsArray:results
					executionDescriptor:execution_descriptor];
				const auto end = std::chrono::high_resolution_clock::now();
				if (not validate_bound_execution_results(run_results, results, "MPSGraph forward run", reason))
				{
					return false;
				}
				forward_run_ms = std::chrono::duration<double, std::milli>(end - start).count();
			}


			auto & stats = training_iteration_stats();
			stats.forward_encode_ms += forward_encode_ms;
			stats.forward_run_ms += forward_run_ms;
		}
		reason = "ok";
		return true;
	}

	bool commit_graph_gradient_results(Network & net,
		const MPSGraphExecutableBundle & bundle,
		const std::vector<MPSGraphBufferBridge> & result_bridges,
		__strong id<MTLCommandBuffer> & command_buffer_handle,
		std::string & reason)
	{
		if (not ensure_training_command_buffer(command_buffer_handle))
		{
			reason = "failed to allocate gradient accumulation command buffer";
			return false;
		}
		id<MTLCommandBuffer> metal_command_buffer = command_buffer_handle;
		id<MTLComputeCommandEncoder> encoder = [metal_command_buffer computeCommandEncoder];
		if (not encoder)
		{
			commit_and_wait_training_command_buffer(command_buffer_handle);
			reason = "failed to create gradient accumulation encoder";
			return false;
		}

		int copied = 0;
		const auto & result_plans = bundle.result_plans();
		for (size_t result_index = 0; result_index < result_plans.size(); ++result_index)
		{
			const auto & tensor_plan = result_plans[result_index];
			if (tensor_plan.owner_layer_index < 0 or tensor_plan.owner_layer_index >= net.n)
			{
				continue;
			}
			float *destination = nullptr;
			Layer & layer = net.layers[tensor_plan.owner_layer_index];
			switch (tensor_plan.binding_kind)
			{
				case MPSGraphTensorBindingKind::kWeightGrad:
					destination = layer.weight_updates;
					break;
				case MPSGraphTensorBindingKind::kBiasGrad:
					destination = layer.bias_updates;
					break;
				case MPSGraphTensorBindingKind::kBatchNormScaleGrad:
					destination = layer.scale_updates;
					break;
				default:
					continue;
			}

			MPSGraphBufferBridge source_bridge;
			TrainingBufferView destination_view;
			if (not destination or tensor_plan.bytes == 0 or
				result_index >= result_bridges.size() or
				not load_required_training_view(destination,
					tensor_plan.bytes,
					TrainingBufferAccess::kReadWrite,
					"gradient update",
					destination_view,
					reason))
			{
				[encoder endEncoding];
				commit_and_wait_training_command_buffer(command_buffer_handle);
				return false;
			}
			source_bridge = result_bridges[result_index];
			id<MTLBuffer> source = source_bridge.buffer_handle;
			id<MTLBuffer> destination_buffer = destination_view.buffer_handle;
			if (not source or not destination_buffer or source_bridge.offset + tensor_plan.bytes > source_bridge.bytes or
				destination_view.offset + tensor_plan.bytes > destination_view.bytes)
			{
				[encoder endEncoding];
				commit_and_wait_training_command_buffer(command_buffer_handle);
				reason = "gradient copy buffer range is invalid";
				return false;
			}
			const MetalBuffer source_buffer{(__bridge void *)source_bridge.buffer_handle, source_bridge.bytes};
			const MetalBuffer wrapped_destination{(__bridge void *)destination_view.buffer_handle, destination_view.bytes};
			const MetalDispatchBuffer buffers[] = {
				{&source_buffer, source_bridge.offset},
				{&wrapped_destination, destination_view.offset},
			};
			const float alpha = 1.0f;
			const uint32_t total = static_cast<uint32_t>(tensor_plan.bytes / sizeof(float));
			const MetalDispatchBytes bytes[] = {
				{&alpha, sizeof(alpha)},
				{&total, sizeof(total)},
			};
			if (not dispatch_training_kernel_1d_with_encoder("buffer_axpy",
					total,
					buffers, 2,
					bytes, 2,
					encoder,
					reason))
			{
				[encoder endEncoding];
				commit_and_wait_training_command_buffer(command_buffer_handle);
				return false;
			}
			note_training_shared_buffer_updated_from_gpu(destination);
			++copied;
		}

		[encoder endEncoding];
		if (copied == 0)
		{
			commit_and_wait_training_command_buffer(command_buffer_handle);
			reason = "backward graph produced no trainable gradients";
			return false;
		}
		Detail::ObjcStrong pending_gradient_copy(metal_command_buffer);
		commit_training_command_buffer(command_buffer_handle);
		record_training_update_command_buffer((id<MTLCommandBuffer>)pending_gradient_copy.get());
		return true;
	}

	bool run_training_graph_terminal_backward(Network & net,
		const float *network_input,
		std::string & reason)
	{
		const auto bundle = training_graph_executable_bundle(&net, true);
		if (not bundle.available())
		{
			reason = bundle.reason().empty() ? "MPSGraph backward executable is unavailable" : bundle.reason();
			return false;
		}
		if (bundle.feed_plans().empty() or bundle.result_plans().empty())
		{
			reason = "MPSGraph backward bundle is missing feed or result metadata";
			return false;
		}

		id<MTLCommandQueue> command_queue = training_mps_command_queue();
		if (not command_queue)
		{
			reason = "Metal command queue is unavailable for MPSGraph backward execution";
			return false;
		}

		@autoreleasepool
		{
			MPSGraphExecutable *backward_executable = bundle.executable();
			NSMutableArray<MPSGraphTensorData *> *inputs = [NSMutableArray arrayWithCapacity:bundle.feed_plans().size()];
			NSMutableArray<MPSGraphTensorData *> *results = [NSMutableArray arrayWithCapacity:bundle.result_plans().size()];

			std::vector<MPSGraphBufferBridge> result_bridges;
			result_bridges.reserve(bundle.result_plans().size());

			for (const auto & tensor_plan : bundle.feed_plans())
			{
				MPSGraphBufferBridge bridge;
				std::string bridge_reason;
				const bool feed_uses_graph_storage =
					uses_graph_owned_storage(tensor_plan) and
					tensor_plan.binding_kind != MPSGraphTensorBindingKind::kNetworkInput and
					tensor_plan.binding_kind != MPSGraphTensorBindingKind::kYoloBoundaryDelta;
				if (feed_uses_graph_storage)
				{
					if (not training_graph_tensor_storage_bridge(bundle, tensor_plan, bridge, bridge_reason))
					{
						reason = bridge_reason;
						return false;
					}
				}
				else
				{
					TrainingHostBuffer ptr;

					if (not resolve_backward_feed_pointer(net, network_input, tensor_plan, ptr, bridge_reason) or
						not bridge_for_forward_tensor(ptr, tensor_plan.bytes, tensor_plan, TrainingBufferAccess::kRead, bridge, bridge_reason))
					{
						reason = bridge_reason;
						return false;
					}
				}
				MPSGraphTensorData *tensor_data_handle = nil;
				if (not make_training_graph_tensor_data(bridge, tensor_data_handle, bridge_reason))
				{
					reason = bridge_reason;
					return false;
				}

				[inputs addObject:tensor_data_handle];
			}

			for (const auto & tensor_plan : bundle.result_plans())
			{
				MPSGraphBufferBridge bridge;
				std::string bridge_reason;
				if (uses_graph_owned_storage(tensor_plan))
				{
					if (not training_graph_tensor_storage_bridge(bundle, tensor_plan, bridge, bridge_reason))
					{
						reason = bridge_reason;
						return false;
					}
				}
				else
				{
					TrainingHostBuffer ptr;

					if (not resolve_backward_result_pointer(net, tensor_plan, ptr, bridge_reason) or
						not bridge_for_forward_tensor(ptr, tensor_plan.bytes, tensor_plan, TrainingBufferAccess::kWrite, bridge, bridge_reason))
					{
						reason = bridge_reason;
						return false;
					}
				}
				result_bridges.push_back(bridge);
				MPSGraphTensorData *tensor_data_handle = nil;
				if (not make_training_graph_tensor_data(bridge, tensor_data_handle, bridge_reason))
				{
					reason = bridge_reason;
					return false;
				}

				[results addObject:tensor_data_handle];
			}

			MPSCommandBuffer *mps_command_buffer = [MPSCommandBuffer commandBufferFromCommandQueue:command_queue];
			if (not mps_command_buffer)
			{
				reason = "failed to allocate backward MPSGraph command buffer";
				return false;
			}
			id<MTLCommandBuffer> backward_command_buffer = mps_command_buffer;
			MPSGraphExecutableExecutionDescriptor *execution_descriptor = make_execution_descriptor(
				@"apple_training_backward",
				results.count,
				false);
			const auto start = std::chrono::high_resolution_clock::now();
			NSArray<MPSGraphTensorData *> *run_results = [backward_executable encodeToCommandBuffer:mps_command_buffer
				inputsArray:inputs
				resultsArray:results
				executionDescriptor:execution_descriptor];
			const auto end = std::chrono::high_resolution_clock::now();
			if (not validate_bound_execution_results(run_results, results, "MPSGraph backward encode", reason))
			{
				commit_and_wait_training_command_buffer(backward_command_buffer);

				return false;
			}
			const double backward_encode_ms = std::chrono::duration<double, std::milli>(end - start).count();
			training_iteration_stats().backward_encode_ms += backward_encode_ms;
			if (not commit_graph_gradient_results(net, bundle, result_bridges, backward_command_buffer, reason))
			{
				return false;
			}

		}

		reason = "ok";
		return true;
	}

	void invalidate_training_graph_engine(Network & net)
	{
		// See configure_training_graph_engine(): never wait for Metal callbacks
		// while holding the session mutex they use for error publication.
		wait_for_training_update_command_buffer();
		auto & session = training_session();
		std::scoped_lock lock(session.mutex);
		if (session.network == &net)
		{
			session.release_owned_resources();
		}
	}
}

namespace Darknet::Apple
{
	bool training_graph_runtime_available(std::string & reason)
	{
		return TrainingDetail::training_graph_runtime_available_impl(reason);
	}
}
