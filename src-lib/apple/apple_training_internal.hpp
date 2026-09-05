#pragma once

/** @file
 * Private contracts shared by the Apple training Objective-C++ translation units.
 */

#ifndef DARKNET_USE_MPS
#error "apple_training_internal.hpp is private to builds with Apple MPS enabled"
#endif

#ifndef __OBJC__
#error "apple_training_internal.hpp requires Objective-C++"
#endif

#include "apple_backend.hpp"
#include "apple_training_data.hpp"
#include "apple_objc.hpp"
#include "apple_training_plan.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

struct MetalDispatchBuffer;
struct MetalDispatchBytes;

namespace Darknet
{
	struct Layer;
	struct Network;
	struct NetworkState;

	namespace Apple
	{
		// Implemented by apple_mps.mm and shared with the training buffer bridge.
		bool training_mps_ready();
		id<MTLDevice> training_mps_device();
		void * training_mps_command_queue_handle();
		id<MTLCommandBuffer> create_training_command_buffer_handle();
		void wait_for_training_command_buffer();

		namespace TrainingDetail
		{
			struct AppleTrainingIterationStats
			{
				double total_batch_ms = 0.0;
				double forward_encode_ms = 0.0;
				double forward_run_ms = 0.0;
				double backward_bridge_ms = 0.0;
				double backward_encode_ms = 0.0;
				double backward_run_ms = 0.0;
				double update_encode_ms = 0.0;
				double update_wait_ms = 0.0;
				double images_per_second = 0.0;
				uint64_t command_buffer_submissions = 0;
				int64_t iteration = -1;
			};

			/** Borrowed Metal storage owned by the shared-buffer cache. Valid until
			 * that cache is invalidated; callers must finish GPU work before then.
			 */
			struct TrainingBufferView
			{
				__unsafe_unretained id<MTLBuffer> buffer_handle = nil;
				size_t offset = 0;
				size_t bytes = 0;

				explicit operator bool() const
				{
					return buffer_handle != nullptr and bytes > 0;
				}
			};

			enum class TrainingBufferAccess
			{
				kRead,
				kWrite,
				kReadWrite,
			};

			enum class MPSGraphBridgeDataType
			{
				kFloat32,
				kInt32,
			};

			/** Borrowed Metal storage owned by a buffer cache or compiled graph.
			 * The owner must remain alive while this bridge is used.
			 */
			struct MPSGraphBufferBridge
			{
				__unsafe_unretained id<MTLBuffer> buffer_handle = nil;
				size_t offset = 0;
				size_t bytes = 0;
				MPSGraphBridgeDataType data_type = MPSGraphBridgeDataType::kFloat32;
				std::vector<int64_t> shape;
			};

			struct MPSGraphExecutablePhase
			{
				Detail::ObjcStrong executable;
				std::vector<MPSGraphTensorPlan> feed_plans;
				std::vector<MPSGraphTensorPlan> result_plans;
			};

			struct MPSGraphTensorStorage
			{
				Detail::ObjcStrong buffer;
				MPSGraphBufferBridge bridge;
			};

			struct MPSGraphCompiledExecutables
			{
				MPSGraphExecutablePhase forward;
				MPSGraphExecutablePhase backward;
				std::vector<MPSGraphTensorStorage> tensor_storage;
				std::string fingerprint;
				std::string reason;
			};

			struct MPSGraphExecutableBundle
			{
				std::shared_ptr<const MPSGraphCompiledExecutables> compiled;
				bool backward = false;

				const MPSGraphExecutablePhase & phase() const
				{
					static const MPSGraphExecutablePhase empty;
					return compiled ? (backward ? compiled->backward : compiled->forward) : empty;
				}

				bool available() const { return compiled and phase().executable.get(); }
				void * executable_handle() const { return (__bridge void *)phase().executable.get(); }
				const std::vector<MPSGraphTensorPlan> & feed_plans() const { return phase().feed_plans; }
				const std::vector<MPSGraphTensorPlan> & result_plans() const { return phase().result_plans; }
				std::string reason() const
				{
					return compiled ? compiled->reason : std::string{};
				}
			};

			struct MPSGraphExecutableBuildResult
			{
				bool forward_ready = false;
				bool backward_ready = false;
				Detail::ObjcStrong forward_executable;
				Detail::ObjcStrong backward_executable;
				std::vector<MPSGraphTensorPlan> forward_feed_plans;
				std::vector<MPSGraphTensorPlan> forward_result_plans;
				std::vector<MPSGraphTensorPlan> backward_feed_plans;
				std::vector<MPSGraphTensorPlan> backward_result_plans;
				std::string reason;
			};

			inline std::string node_kind_label(const MPSGraphNodeKind kind)
			{
				switch (kind)
				{
					case MPSGraphNodeKind::kConvBnAct: return "conv_bn_act";
					case MPSGraphNodeKind::kRouteConcat: return "route_concat";
					case MPSGraphNodeKind::kRouteGroupSlice: return "route_group_slice";
					case MPSGraphNodeKind::kMaxPool: return "maxpool";
					case MPSGraphNodeKind::kUpsample: return "upsample";
					case MPSGraphNodeKind::kYoloBoundaryOutput: return "yolo_boundary";
				}
				return "node";
			}

			inline std::string tensor_binding_label(const MPSGraphTensorBindingKind kind)
			{
				switch (kind)
				{
					case MPSGraphTensorBindingKind::kNetworkInput: return "network_input";
					case MPSGraphTensorBindingKind::kNetworkInputDelta: return "network_input_delta";
					case MPSGraphTensorBindingKind::kLayerOutput: return "layer_output";
					case MPSGraphTensorBindingKind::kLayerDelta: return "layer_delta";
					case MPSGraphTensorBindingKind::kWeights: return "weights";
					case MPSGraphTensorBindingKind::kWeightGrad: return "weight_grad";
					case MPSGraphTensorBindingKind::kBias: return "bias";
					case MPSGraphTensorBindingKind::kBiasGrad: return "bias_grad";
					case MPSGraphTensorBindingKind::kBatchNormScale: return "bn_scale";
					case MPSGraphTensorBindingKind::kBatchNormScaleGrad: return "bn_scale_grad";
					case MPSGraphTensorBindingKind::kConvSourceTape: return "conv_source";
					case MPSGraphTensorBindingKind::kBatchNormInputTape: return "batchnorm_input";
					case MPSGraphTensorBindingKind::kBatchMeanTape: return "batch_mean";
					case MPSGraphTensorBindingKind::kBatchVarianceTape: return "batch_variance";
					case MPSGraphTensorBindingKind::kBatchNormOutputTape: return "batchnorm_output";
					case MPSGraphTensorBindingKind::kActivationTape: return "activation_tape";
					case MPSGraphTensorBindingKind::kMaxPoolIndicesTape: return "maxpool_indices";
					case MPSGraphTensorBindingKind::kYoloBoundaryOutput: return "yolo_boundary_output";
					case MPSGraphTensorBindingKind::kYoloBoundaryDelta: return "yolo_boundary_delta";
				}
				return "tensor";
			}

			inline std::string node_debug_label(const MPSGraphNodePlan & node)
			{
				return "layer_" + std::to_string(node.darknet_layer_index) + "_" + node_kind_label(node.op_kind);
			}

			inline std::string tensor_debug_label(const MPSGraphTensorPlan & tensor_plan)
			{
				std::string label = "layer_" + std::to_string(tensor_plan.owner_layer_index) + "_" +
					tensor_binding_label(tensor_plan.binding_kind);
				if (tensor_plan.source_layer_index >= 0 and tensor_plan.source_layer_index != tensor_plan.owner_layer_index)
				{
					label += "_src_" + std::to_string(tensor_plan.source_layer_index);
				}
				return label;
			}

			inline std::string tensor_data_type_label(const MPSGraphTensorDataType data_type)
			{
				switch (data_type)
				{
					case MPSGraphTensorDataType::kInt32: return "i32";
					case MPSGraphTensorDataType::kFloat32:
					default: return "fp32";
				}
			}

			inline std::string shape_key(const std::vector<int64_t> & shape)
			{
				std::ostringstream stream;
				for (size_t i = 0; i < shape.size(); ++i)
				{
					if (i > 0) stream << "x";
					stream << shape[i];
				}
				return stream.str();
			}

			std::string training_optimizer_summary(const Network & net);
			std::string training_precision_summary(const Network & net);
			/** Own the command buffer and encoder shared by one optimizer update. */
			struct TrainingOptimizerCommands
			{
				TrainingOptimizerCommands() = default;
				TrainingOptimizerCommands(const TrainingOptimizerCommands &) = delete;
				TrainingOptimizerCommands & operator=(const TrainingOptimizerCommands &) = delete;

				~TrainingOptimizerCommands()
				{
					if (encoder) [encoder endEncoding];
				}

				__strong id<MTLCommandBuffer> command_buffer = nil;
				__strong id<MTLComputeCommandEncoder> encoder = nil;
			};

			bool training_optimizer_update_layer(Network & net, Layer & layer, int update_batch,
				float learning_rate, TrainingOptimizerCommands & commands,
				std::string & reason);

			AppleTrainingIterationStats & training_iteration_stats();
			void begin_training_iteration(Network & net);
			bool prepare_training_buffers(Network & net, std::string & reason);
			void invalidate_training_buffers(Network & net);
			void reset_training_buffer_state();
			bool get_training_buffer_view(const void *ptr, size_t bytes, TrainingBufferAccess access, TrainingBufferView & view);
			bool load_required_training_view(const void *ptr, size_t bytes, TrainingBufferAccess access,
				const std::string & label, TrainingBufferView & view, std::string & reason);
			bool dispatch_training_kernel_1d_with_encoder(const std::string & kernel_name, size_t threads,
				const MetalDispatchBuffer *buffers, size_t buffer_count,
				const MetalDispatchBytes *bytes, size_t bytes_count,
				void *encoder_handle, std::string & reason);
			bool ensure_training_command_buffer(void *& command_buffer_handle);
			bool commit_training_command_buffer(void *& command_buffer_handle, uint64_t *commit_count = nullptr);
			bool commit_and_wait_training_command_buffer(void *& command_buffer_handle, double *wait_ms,
				uint64_t *commit_count = nullptr);
			void note_training_shared_buffer_updated_from_host(const void *ptr);
			void note_training_shared_buffer_updated_from_gpu(const void *ptr);
			void erase_training_shared_buffer_storage(const void *ptr);
			void sync_all_training_shared_buffers_to_host();
			void invalidate_layer_training_storage(const Layer & layer);
			void note_layer_training_parameters_updated_cpu(const Layer & layer);
			void note_layer_training_parameters_updated_gpu(const Layer & layer);
			void note_layer_training_optimizer_state_updated_cpu(const Layer & layer);
			void note_layer_training_optimizer_state_updated_gpu(const Layer & layer);
			void note_layer_training_running_stats_updated_cpu(const Layer & layer);
			void note_layer_training_running_stats_updated_gpu(const Layer & layer);
			bool training_is_terminal_detector_head(const Network & net, int layer_index);
			double sync_training_layers_to_host(Network & net, const bool sync_rolling_stats = true,
				const bool sync_parameters = false);
			void finalize_training_iteration(Network & net, double update_encode_ms, double update_wait_ms = 0.0);
			bool make_training_graph_buffer_bridge(const TrainingBufferView & view,
				const std::vector<int64_t> & shape, MPSGraphBufferBridge & bridge, std::string & reason);
			bool make_training_graph_buffer_bridge(const void *ptr, size_t bytes,
				const std::vector<int64_t> & shape, TrainingBufferAccess access,
				MPSGraphBufferBridge & bridge, std::string & reason);
			bool make_training_graph_tensor_data_handle(const MPSGraphBufferBridge & bridge,
				void *& tensor_data_handle, std::string & reason);
			void release_training_graph_tensor_data_handle(void *tensor_data_handle);
			std::string error_description(NSError *error);
			void record_training_execution_error(const std::string &message);
			std::string take_training_execution_error();
			void record_training_update_command_buffer(id<MTLCommandBuffer> command_buffer);
			double wait_for_training_update_command_buffer();

			bool host_parameter_consumers_supported_by_current_contract(const Network & net, std::string & reason);
			bool gpu_yolo_loss_supported_by_current_contract(const Network & net, std::string & reason);
			void reset_gpu_yolo_loss_stats_buffer_cache(const Network * net = nullptr);
			float train_training_graph_datum(Network & net, const FloatBufferView & input, const FloatBufferView & truth);

			bool configure_training_graph_engine(Network & net, std::string & reason);
			bool training_graph_engine_enabled(const Network & net);
			void invalidate_training_graph_engine(Network & net);
			bool run_training_graph_forward_probe(Network & net, const NetworkState & state, std::string & reason,
				void *& encoded_command_buffer_handle, const bool encode_to_command_buffer = true);

			inline bool run_training_graph_forward_probe(Network & net, const NetworkState & state, std::string & reason)
			{
				void *command_buffer = nullptr;
				return run_training_graph_forward_probe(net, state, reason, command_buffer, false);
			}
			bool run_training_graph_terminal_backward(Network & net, const float *network_input, std::string & reason);
			TrainingGraphInfo training_graph_info_snapshot(const Network & net);
			MPSGraphExecutableBundle training_graph_executable_bundle(const Network * net, bool backward);
			bool training_graph_tensor_storage_bridge(const MPSGraphExecutableBundle & bundle,
				const MPSGraphTensorPlan & tensor_plan,
				MPSGraphBufferBridge & bridge, std::string & reason);

			MPSGraphExecutableBuildResult build_training_graph_executables(Network & net,
				const MPSGraphSegmentPlan & plan);
		}
	}
}
