#include "darknet_internal.hpp"
#include "apple_training_internal.hpp"
#include "apple_training_plan.hpp"

#include "darknet_network.hpp"

#include <algorithm>
#include <CoreFoundation/CoreFoundation.h>
#include <chrono>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace Darknet::Apple::TrainingDetail
{
	namespace
	{
		NSString *ns_string(const std::string & value)
		{
			return [NSString stringWithUTF8String:value.c_str()];
		}

		MPSShape *shape_from_vector(const std::vector<int64_t> & shape)
		{
			NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:shape.size()];
			for (const auto dim : shape)
			{
				[values addObject:@(dim)];
			}
			return values;
		}

		MPSShape *shape_for_channel_parameter(const int channels)
		{
			return @[ @1, @(channels), @1, @1 ];
		}

		float darknet_leaky_alpha()
		{
			return 0.1f;
		}

		MPSGraphTensor *add_scalar_constant(MPSGraph *graph,
			const double value)
		{
			return [graph constantWithScalar:value
				dataType:MPSDataTypeFloat32];
		}

		MPSGraphTensor *apply_darknet_leaky_relu(MPSGraph *graph,
			MPSGraphTensor *tensor,
			const std::string & debug_name)
		{
			return [graph leakyReLUWithTensor:tensor
				alpha:darknet_leaky_alpha()
				name:ns_string(debug_name + ":leaky")];
		}

		std::string activation_name(const MPSGraphActivationKind activation)
		{
			switch (activation)
			{
				case MPSGraphActivationKind::kLeaky: return "leaky";
				case MPSGraphActivationKind::kLogistic: return "logistic";
				case MPSGraphActivationKind::kLinear:
				default:
					return "linear";
			}
		}

		MPSGraphTensor *add_int64_vector_constant(MPSGraph *graph,
			const std::vector<int64_t> & values)
		{
			NSData *data = [NSData dataWithBytes:values.data()
				length:static_cast<NSUInteger>(values.size() * sizeof(int64_t))];
			return [graph constantWithData:data
				shape:shape_from_vector({static_cast<int64_t>(values.size())})
				dataType:MPSDataTypeInt64];
		}

		NSArray<NSNumber *> *batchnorm_axes_nchw()
		{
			return @[ @0, @2, @3 ];
		}

		float batchnorm_forward_epsilon()
		{
			return 0.00001f;
		}

		float batchnorm_backward_epsilon()
		{
			return 0.00001f;
		}

		MPSDataType graph_data_type_for_tensor_data_type(const MPSGraphTensorDataType data_type)
		{
			switch (data_type)
			{
				case MPSGraphTensorDataType::kInt32: return MPSDataTypeInt32;
				case MPSGraphTensorDataType::kFloat32:
				default:
					return MPSDataTypeFloat32;
			}
		}

		MPSDataType graph_data_type_for_tensor_plan(const MPSGraphTensorPlan & tensor_plan)
		{
			return graph_data_type_for_tensor_data_type(tensor_plan.data_type);
		}

		MPSGraphTensor *cast_tensor_to_data_type(MPSGraph *graph,
			MPSGraphTensor *tensor,
			const MPSGraphTensorDataType data_type,
			const std::string & debug_name)
		{
			if (not graph or not tensor)
			{
				return tensor;
			}

			return [graph castTensor:tensor
				toType:graph_data_type_for_tensor_data_type(data_type)
				name:ns_string(debug_name)];
		}

		MPSGraphTensor *cast_tensor_to_tensor_plan(MPSGraph *graph,
			MPSGraphTensor *tensor,
			const MPSGraphTensorPlan & tensor_plan,
			const std::string & debug_name)
		{
			if (tensor_plan.data_type == MPSGraphTensorDataType::kFloat32)
			{
				return tensor;
			}
			return cast_tensor_to_data_type(graph, tensor, tensor_plan.data_type, debug_name);
		}

		std::vector<int64_t> output_shape_for_layer(const Layer & layer)
		{
			return
			{
				static_cast<int64_t>(layer.batch),
				static_cast<int64_t>(layer.out_c),
				static_cast<int64_t>(layer.out_h),
				static_cast<int64_t>(layer.out_w),
			};
		}

		std::string node_key(const int layer_index, const std::string & suffix)
		{
			return "layer_" + std::to_string(layer_index) + "_" + suffix;
		}

		struct ForwardBuildState
		{
			MPSGraph *graph = nil;
			NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feed_shapes = nil;
			NSMutableArray<MPSGraphTensor *> *forward_targets = nil;
			std::unordered_map<int, MPSGraphTensor *> output_by_layer;
			std::unordered_map<std::string, MPSGraphTensor *> placeholder_by_key;
			std::unordered_set<std::string> forward_result_keys;
			std::vector<MPSGraphTensor *> forward_feed_tensors;
			std::vector<MPSGraphTensorPlan> forward_feed_plans;
			std::vector<MPSGraphTensor *> forward_result_tensors;
			std::vector<MPSGraphTensorPlan> forward_result_plans;
		};

		struct BackwardBuildState
		{
			MPSGraph *graph = nil;
			NSMutableDictionary<MPSGraphTensor *, MPSGraphShapedType *> *feed_shapes = nil;
			NSMutableArray<MPSGraphTensor *> *backward_targets = nil;
			std::unordered_map<std::string, MPSGraphTensor *> placeholder_by_key;
			std::vector<MPSGraphTensor *> backward_feed_tensors;
			std::vector<MPSGraphTensorPlan> backward_feed_plans;
			std::vector<MPSGraphTensor *> backward_result_tensors;
			std::vector<MPSGraphTensorPlan> backward_result_plans;
		};

		bool order_tensor_plans_for_executable(NSArray<MPSGraphTensor *> *executable_tensors,
			const std::vector<MPSGraphTensor *> & planned_tensors,
			const std::vector<MPSGraphTensorPlan> & planned_plans,
			const std::string & label,
			std::vector<MPSGraphTensorPlan> & ordered_plans,
			std::string & reason)
		{
			ordered_plans.clear();
			if (planned_tensors.size() != planned_plans.size())
			{
				reason = std::string(label) + " tensor and plan counts differ before executable ordering";
				return false;
			}
			if (not executable_tensors or executable_tensors.count != planned_tensors.size())
			{
				std::ostringstream oss;
				oss << label << " executable tensor count mismatch: executable="
					<< static_cast<size_t>(executable_tensors ? executable_tensors.count : 0)
					<< " planned=" << planned_tensors.size();
				reason = oss.str();
				return false;
			}

			std::vector<unsigned char> used(planned_tensors.size(), 0);
			ordered_plans.reserve(planned_plans.size());
			for (MPSGraphTensor *tensor in executable_tensors)
			{
				bool matched = false;
				for (size_t idx = 0; idx < planned_tensors.size(); ++idx)
				{
					if (not used[idx] and planned_tensors[idx] == tensor)
					{
						used[idx] = 1;
						ordered_plans.push_back(planned_plans[idx]);
						matched = true;
						break;
					}
				}
				if (not matched)
				{
					reason = std::string("MPSGraph executable returned an unknown ") + label + " tensor";
					return false;
				}
			}
			return true;
		}

		MPSGraphTensor *add_placeholder(ForwardBuildState & state,
			const std::string & key,
			const std::vector<int64_t> & shape,
			const std::string & debug_name,
			const MPSGraphTensorPlan *tensor_plan = nullptr)
		{
			const auto it = state.placeholder_by_key.find(key);
			if (it != state.placeholder_by_key.end())
			{
				return it->second;
			}

			const MPSDataType data_type = tensor_plan
				? graph_data_type_for_tensor_plan(*tensor_plan)
				: MPSDataTypeFloat32;
			MPSGraphTensor *tensor = [state.graph placeholderWithShape:shape_from_vector(shape)
				dataType:data_type
				name:ns_string(debug_name)];
			MPSGraphShapedType *type = [[MPSGraphShapedType alloc] initWithShape:shape_from_vector(shape) dataType:data_type];
			state.feed_shapes[tensor] = type;
			state.placeholder_by_key.emplace(key, tensor);
			if (tensor_plan)
			{
				state.forward_feed_tensors.push_back(tensor);
				state.forward_feed_plans.push_back(*tensor_plan);
			}
			return tensor;
		}

		MPSGraphTensor *add_parameter_placeholder(ForwardBuildState & state,
			const MPSGraphNodePlan & node,
			const std::string & suffix,
			const MPSGraphTensorPlan & tensor_plan,
			const std::vector<int64_t> & shape)
		{
			return add_placeholder(state,
				node_key(node.darknet_layer_index, suffix),
				shape,
				node_debug_label(node) + ":" + suffix,
				&tensor_plan);
		}

		MPSGraphTensor *add_backward_placeholder(BackwardBuildState & state,
			const std::string & key,
			const std::vector<int64_t> & shape,
			const std::string & debug_name,
			const MPSGraphTensorPlan & tensor_plan)
		{
			const auto it = state.placeholder_by_key.find(key);
			if (it != state.placeholder_by_key.end())
			{
				return it->second;
			}

			const MPSDataType data_type = graph_data_type_for_tensor_plan(tensor_plan);
			MPSGraphTensor *tensor = [state.graph placeholderWithShape:shape_from_vector(shape)
				dataType:data_type
				name:ns_string(debug_name)];
			MPSGraphShapedType *type = [[MPSGraphShapedType alloc] initWithShape:shape_from_vector(shape) dataType:data_type];
			state.feed_shapes[tensor] = type;
			state.placeholder_by_key.emplace(key, tensor);
			state.backward_feed_tensors.push_back(tensor);
			state.backward_feed_plans.push_back(tensor_plan);
			return tensor;
		}

		void append_forward_result(ForwardBuildState & state, MPSGraphTensor *tensor, const MPSGraphTensorPlan & tensor_plan)
		{
			if (not tensor)
			{
				return;
			}
			const std::string key =
				std::to_string(static_cast<int>(tensor_plan.binding_kind)) + ":" +
				std::to_string(tensor_plan.owner_layer_index) + ":" +
				std::to_string(tensor_plan.source_layer_index) + ":" +
				tensor_data_type_label(tensor_plan.data_type) + ":" +
				shape_key(tensor_plan.shape_nchw) + ":" +
				tensor_debug_label(tensor_plan);
			if (not state.forward_result_keys.insert(key).second)
			{
				return;
			}
			tensor = cast_tensor_to_tensor_plan(state.graph,
				tensor,
				tensor_plan,
				tensor_debug_label(tensor_plan) + ":result_cast");
			[state.forward_targets addObject:tensor];
			state.forward_result_tensors.push_back(tensor);
			state.forward_result_plans.push_back(tensor_plan);
		}

		const MPSGraphTensorPlan *find_boundary_tensor_plan(const MPSGraphSegmentPlan & plan,
			const MPSGraphTensorBindingKind binding_kind,
			const int owner_layer_index)
		{
			for (const auto & tensor_plan : plan.boundary_tensors)
			{
				if (tensor_plan.binding_kind == binding_kind and
					tensor_plan.owner_layer_index == owner_layer_index)
				{
					return &tensor_plan;
				}
			}
			return nullptr;
		}

		const MPSGraphNodePlan *find_node_plan(const MPSGraphSegmentPlan & plan, const int layer_index)
		{
			for (const auto & node : plan.nodes)
			{
				if (node.darknet_layer_index == layer_index)
				{
					return &node;
				}
			}
			return nullptr;
		}

		const MPSGraphTensorPlan *find_tensor_plan(const MPSGraphNodePlan & node, const MPSGraphTensorBindingKind binding_kind)
		{
			for (const auto & tensor_plan : node.tensors)
			{
				if (tensor_plan.binding_kind == binding_kind)
				{
					return &tensor_plan;
				}
			}
			return nullptr;
		}

		MPSGraphTensor *source_tensor_for_node(ForwardBuildState & state,
			const Network & net,
			const MPSGraphNodePlan & node,
			std::string & reason)
		{
			if (node.segment_input)
			{
				const Layer & layer = net.layers[node.darknet_layer_index];
				for (const auto & tensor_plan : node.tensors)
				{
					if (tensor_plan.binding_kind == MPSGraphTensorBindingKind::kNetworkInput)
					{
						return add_placeholder(state,
							"network_input",
							{
								static_cast<int64_t>(layer.batch),
								static_cast<int64_t>(layer.c),
								static_cast<int64_t>(layer.h),
								static_cast<int64_t>(layer.w),
							},
							"network_input",
							&tensor_plan);
					}
				}
				reason = "network input tensor plan is unavailable";
				return nil;
			}

			if (node.op_kind == MPSGraphNodeKind::kRouteConcat or node.op_kind == MPSGraphNodeKind::kRouteGroupSlice)
			{
				return nil;
			}

			const int source_layer_index = node.darknet_layer_index - 1;
			const auto it = state.output_by_layer.find(source_layer_index);
			if (it == state.output_by_layer.end())
			{
				reason = "missing source tensor for layer #" + std::to_string(node.darknet_layer_index);
				return nil;
			}
			return it->second;
		}

		bool append_forward_tape_targets(ForwardBuildState & state,
			const MPSGraphNodePlan & node,
			MPSGraphTensor *conv_source,
			MPSGraphTensor *layer_output,
			MPSGraphTensor *batchnorm_input,
			MPSGraphTensor *batch_mean,
			MPSGraphTensor *batch_variance,
			MPSGraphTensor *batchnorm_output,
			MPSGraphTensor *activation_tape)
		{
			for (const auto & tensor_plan : node.tensors)
			{
				if (not tensor_plan.forward_result)
				{
					continue;
				}

				switch (tensor_plan.binding_kind)
				{
					case MPSGraphTensorBindingKind::kLayerOutput:
						append_forward_result(state, layer_output, tensor_plan);
						break;
					case MPSGraphTensorBindingKind::kConvSourceTape:
						append_forward_result(state, conv_source, tensor_plan);
						break;
					case MPSGraphTensorBindingKind::kBatchNormInputTape:
						append_forward_result(state, batchnorm_input, tensor_plan);
						break;
					case MPSGraphTensorBindingKind::kBatchMeanTape:
						append_forward_result(state, batch_mean, tensor_plan);
						break;
					case MPSGraphTensorBindingKind::kBatchVarianceTape:
						append_forward_result(state, batch_variance, tensor_plan);
						break;
					case MPSGraphTensorBindingKind::kBatchNormOutputTape:
						append_forward_result(state, batchnorm_output, tensor_plan);
						break;
					case MPSGraphTensorBindingKind::kActivationTape:
						append_forward_result(state, activation_tape, tensor_plan);
						break;
					default:
						break;
				}
			}

			return true;
		}

		bool build_forward_conv_node(ForwardBuildState & state,
			const Network & net,
			const MPSGraphNodePlan & node,
			std::string & reason)
		{
			MPSGraphTensor *source = source_tensor_for_node(state, net, node, reason);
			if (not source)
			{
				return false;
			}

			const Layer & layer = net.layers[node.darknet_layer_index];
			const auto tensor_plan_for = [&](const MPSGraphTensorBindingKind kind) -> const MPSGraphTensorPlan *
			{
				for (const auto & tensor_plan : node.tensors)
				{
					if (tensor_plan.binding_kind == kind)
					{
						return &tensor_plan;
					}
				}
				return nullptr;
			};
			const auto *weights_plan = tensor_plan_for(MPSGraphTensorBindingKind::kWeights);
			if (not weights_plan)
			{
				reason = "weight tensor plan is unavailable for layer #" + std::to_string(node.darknet_layer_index);
				return false;
			}
			MPSGraphTensor *weights = add_parameter_placeholder(state,
				node,
				"weights",
				*weights_plan,
				{
					static_cast<int64_t>(layer.n),
					static_cast<int64_t>(std::max(1, layer.c / std::max(1, layer.groups))),
					static_cast<int64_t>(layer.size),
					static_cast<int64_t>(layer.size),
				});
			auto *descriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:node.stride_x
				strideInY:node.stride_y
				dilationRateInX:1
				dilationRateInY:1
				groups:std::max(1, node.groups)
				paddingLeft:node.pad_left
				paddingRight:node.pad_right
				paddingTop:node.pad_top
				paddingBottom:node.pad_bottom
				paddingStyle:MPSGraphPaddingStyleExplicit
				dataLayout:MPSGraphTensorNamedDataLayoutNCHW
				weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
			if (not descriptor)
			{
				reason = "failed to create convolution descriptor for layer #" + std::to_string(node.darknet_layer_index);
				return false;
			}

			MPSGraphTensor *conv_output = [state.graph convolution2DWithSourceTensor:source
				weightsTensor:weights
				descriptor:descriptor
				name:ns_string(node_debug_label(node) + ":conv")];
			MPSGraphTensor *output = conv_output;

			MPSGraphTensor *batchnorm_input = nil;
			MPSGraphTensor *batch_mean = nil;
			MPSGraphTensor *batch_variance = nil;
			MPSGraphTensor *batchnorm_output = nil;
			MPSGraphTensor *activation_tape = nil;

			if (node.batch_normalize)
			{
				const auto *gamma_plan = tensor_plan_for(MPSGraphTensorBindingKind::kBatchNormScale);
				const auto *beta_plan = tensor_plan_for(MPSGraphTensorBindingKind::kBias);
				if (not gamma_plan or not beta_plan)
				{
					reason = "batchnorm parameter tensor plan is unavailable for layer #" + std::to_string(node.darknet_layer_index);
					return false;
				}
				MPSGraphTensor *gamma = add_parameter_placeholder(state, node, "bn_scale", *gamma_plan, {static_cast<int64_t>(layer.n)});
				MPSGraphTensor *beta = add_parameter_placeholder(state, node, "bias", *beta_plan, {static_cast<int64_t>(layer.n)});
				MPSGraphTensor *gamma_4d = [state.graph reshapeTensor:gamma
					withShape:shape_for_channel_parameter(layer.n)
					name:ns_string(node_debug_label(node) + ":gamma_reshape")];
				MPSGraphTensor *beta_4d = [state.graph reshapeTensor:beta
					withShape:shape_for_channel_parameter(layer.n)
					name:ns_string(node_debug_label(node) + ":beta_reshape")];

				batchnorm_input = conv_output;
				MPSGraphTensor *batch_mean_reduced = [state.graph meanOfTensor:conv_output
					axes:batchnorm_axes_nchw()
					name:ns_string(node_debug_label(node) + ":mean")];
				// Match cuDNN: normalization uses the population variance.  The
				// unbiased correction is applied only when updating rolling state.
				MPSGraphTensor *batch_variance_reduced = [state.graph varianceOfTensor:conv_output
					meanTensor:batch_mean_reduced
					axes:batchnorm_axes_nchw()
					name:ns_string(node_debug_label(node) + ":variance")];
				batch_mean = [state.graph reshapeTensor:batch_mean_reduced
					withShape:shape_for_channel_parameter(layer.n)
					name:ns_string(node_debug_label(node) + ":mean_reshape")];
				batch_variance = [state.graph reshapeTensor:batch_variance_reduced
					withShape:shape_for_channel_parameter(layer.n)
					name:ns_string(node_debug_label(node) + ":variance_reshape")];
				batchnorm_output = [state.graph normalizationWithTensor:conv_output
					meanTensor:batch_mean
					varianceTensor:batch_variance
					gammaTensor:nil
					betaTensor:nil
					epsilon:batchnorm_forward_epsilon()
					name:ns_string(node_debug_label(node) + ":normalize")];
				output = [state.graph multiplicationWithPrimaryTensor:batchnorm_output
					secondaryTensor:gamma_4d
					name:ns_string(node_debug_label(node) + ":scale")];
				output = [state.graph additionWithPrimaryTensor:output
					secondaryTensor:beta_4d
					name:ns_string(node_debug_label(node) + ":bias_add")];
			}
			else if (layer.biases)
			{
				const auto *bias_plan = tensor_plan_for(MPSGraphTensorBindingKind::kBias);
				if (not bias_plan)
				{
					reason = "bias tensor plan is unavailable for layer #" + std::to_string(node.darknet_layer_index);
					return false;
				}
				MPSGraphTensor *bias = add_parameter_placeholder(state, node, "bias", *bias_plan, {static_cast<int64_t>(layer.n)});
				MPSGraphTensor *bias_4d = [state.graph reshapeTensor:bias
					withShape:shape_for_channel_parameter(layer.n)
					name:ns_string(node_debug_label(node) + ":bias_reshape")];
				output = [state.graph additionWithPrimaryTensor:output
					secondaryTensor:bias_4d
					name:ns_string(node_debug_label(node) + ":bias_add")];
			}

			switch (node.activation)
			{
				case MPSGraphActivationKind::kLeaky:
					output = apply_darknet_leaky_relu(state.graph,
						output,
						node_debug_label(node));
					activation_tape = output;
					break;
				case MPSGraphActivationKind::kLogistic:
					output = [state.graph sigmoidWithTensor:output
						name:ns_string(node_debug_label(node) + ":logistic")];
					activation_tape = output;
					break;
				case MPSGraphActivationKind::kLinear:
				default:
					break;
			}

			if (not append_forward_tape_targets(state, node, nil, output, batchnorm_input, batch_mean, batch_variance, batchnorm_output, activation_tape))
			{
				return false;
			}
			state.output_by_layer[node.darknet_layer_index] = output;
			return true;
		}

		bool build_forward_route_concat_node(ForwardBuildState & state,
			const MPSGraphNodePlan & node,
			std::string & reason)
		{
			NSMutableArray<MPSGraphTensor *> *inputs = [NSMutableArray arrayWithCapacity:node.input_node_indices.size()];
			for (const int source_layer_index : node.input_node_indices)
			{
				const auto it = state.output_by_layer.find(source_layer_index);
				if (it == state.output_by_layer.end())
				{
					reason = "missing concat input from layer #" + std::to_string(source_layer_index);
					return false;
				}
				[inputs addObject:it->second];
			}

			if (inputs.count == 0)
			{
				reason = "route concat has no inputs";
				return false;
			}

			MPSGraphTensor *output = nil;
			if (inputs.count == 1)
			{
				output = inputs[0];
			}
			else
			{
				output = [state.graph concatTensors:inputs
					dimension:node.route_concat_axis
					name:ns_string(node_debug_label(node) + ":concat")];
			}
			state.output_by_layer[node.darknet_layer_index] = output;
			if (const auto *output_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kLayerOutput);
				output_plan and output_plan->forward_result)
			{
				append_forward_result(state, output, *output_plan);
			}
			return true;
		}

		bool build_forward_route_group_slice_node(ForwardBuildState & state,
			const MPSGraphNodePlan & node,
			std::string & reason)
		{
			if (node.input_node_indices.empty())
			{
				reason = "route group slice has no source input";
				return false;
			}

			const int source_layer_index = node.input_node_indices.front();
			const auto it = state.output_by_layer.find(source_layer_index);
			if (it == state.output_by_layer.end())
			{
				reason = "missing route group source from layer #" + std::to_string(source_layer_index);
				return false;
			}

			MPSGraphTensor *output = [state.graph sliceTensor:it->second
				dimension:1
				start:node.slice_channel_offset
				length:node.slice_channel_count
				name:ns_string(node_debug_label(node) + ":slice")];
			state.output_by_layer[node.darknet_layer_index] = output;
			if (const auto *output_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kLayerOutput);
				output_plan and output_plan->forward_result)
			{
				append_forward_result(state, output, *output_plan);
			}
			return true;
		}

		bool build_forward_maxpool_node(ForwardBuildState & state,
			const Network & net,
			const MPSGraphNodePlan & node,
			std::string & reason)
		{
			MPSGraphTensor *source = source_tensor_for_node(state, net, node, reason);
			if (not source)
			{
				return false;
			}

			auto *descriptor = [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:node.kernel_w
				kernelHeight:node.kernel_h
				strideInX:node.stride_x
				strideInY:node.stride_y
				dilationRateInX:1
				dilationRateInY:1
				paddingLeft:node.pad_left
				paddingRight:node.pad_right
				paddingTop:node.pad_top
				paddingBottom:node.pad_bottom
				paddingStyle:MPSGraphPaddingStyleExplicit
				dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
			descriptor.ceilMode = NO;
			descriptor.returnIndicesMode = MPSGraphPoolingReturnIndicesGlobalFlatten2D;
			descriptor.returnIndicesDataType = MPSDataTypeInt32;

			NSArray<MPSGraphTensor *> *results = [state.graph maxPooling2DReturnIndicesWithSourceTensor:source
				descriptor:descriptor
				name:ns_string(node_debug_label(node) + ":maxpool")];
			if (results.count < 2)
			{
				reason = "maxpool did not return output and indices tensors";
				return false;
			}

			if (not append_forward_tape_targets(state, node, nil, results[0], nil, nil, nil, nil, nil))
			{
				return false;
			}
			state.output_by_layer[node.darknet_layer_index] = results[0];
			for (const auto & tensor_plan : node.tensors)
			{
				if (tensor_plan.binding_kind == MPSGraphTensorBindingKind::kMaxPoolIndicesTape and tensor_plan.forward_result)
				{
					append_forward_result(state, results[1], tensor_plan);
				}
			}
			return true;
		}

		bool build_forward_upsample_node(ForwardBuildState & state,
			const Network & net,
			const MPSGraphNodePlan & node,
			std::string & reason)
		{
			MPSGraphTensor *source = source_tensor_for_node(state, net, node, reason);
			if (not source)
			{
				return false;
			}

			const Layer & layer = net.layers[node.darknet_layer_index];
			MPSGraphTensor *size_tensor = add_int64_vector_constant(state.graph,
				{static_cast<int64_t>(layer.out_h), static_cast<int64_t>(layer.out_w)});
			MPSGraphTensor *output = [state.graph resizeNearestWithTensor:source
				sizeTensor:size_tensor
				nearestRoundingMode:MPSGraphResizeNearestRoundingModeFloor
				centerResult:NO
				alignCorners:NO
				layout:MPSGraphTensorNamedDataLayoutNCHW
				name:ns_string(node_debug_label(node) + ":upsample")];
			output = [state.graph reshapeTensor:output
				withShape:shape_from_vector(node.output_shape_nchw)
				name:ns_string(node_debug_label(node) + ":upsample_shape")];
			state.output_by_layer[node.darknet_layer_index] = output;
			if (const auto *output_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kLayerOutput);
				output_plan and output_plan->forward_result)
			{
				append_forward_result(state, output, *output_plan);
			}
			return true;
		}

		MPSGraphExecutableBuildResult build_forward_executable(Network & net, const MPSGraphSegmentPlan & plan)
		{
			MPSGraphExecutableBuildResult result;

			id<MTLDevice> device = (__bridge id<MTLDevice>)runtime_metal_device_handle();
			if (not device)
			{
				result.reason = "Metal device unavailable for MPSGraph compile";
				return result;
			}

			MPSGraphDevice *graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
			if (not graph_device)
			{
				result.reason = "failed to create MPSGraph device";
				return result;
			}

			@autoreleasepool
			{
				ForwardBuildState state;
				state.graph = [[MPSGraph alloc] init];
				state.graph.options = MPSGraphOptionsNone;
				state.feed_shapes = [NSMutableDictionary dictionary];
				state.forward_targets = [NSMutableArray array];
				for (const auto & node : plan.nodes)
				{
					std::string reason;
					bool ok = false;
					switch (node.op_kind)
					{
						case MPSGraphNodeKind::kConvBnAct:
							ok = build_forward_conv_node(state, net, node, reason);
							break;
						case MPSGraphNodeKind::kRouteConcat:
							ok = build_forward_route_concat_node(state, node, reason);
							break;
						case MPSGraphNodeKind::kRouteGroupSlice:
							ok = build_forward_route_group_slice_node(state, node, reason);
							break;
						case MPSGraphNodeKind::kMaxPool:
							ok = build_forward_maxpool_node(state, net, node, reason);
							break;
						case MPSGraphNodeKind::kUpsample:
							ok = build_forward_upsample_node(state, net, node, reason);
							break;
						case MPSGraphNodeKind::kYoloBoundaryOutput:
							ok = true;
							break;
					}
					if (not ok)
					{
						result.reason = reason.empty() ? ("failed to lower " + node_debug_label(node)) : reason;
						return result;
					}
				}

				for (const int layer_index : plan.terminal_output_layers)
				{
					const auto it = state.output_by_layer.find(layer_index);
					if (it == state.output_by_layer.end())
					{
						result.reason = "missing terminal output tensor for layer #" + std::to_string(layer_index);
						return result;
					}
					const auto *boundary_plan = find_boundary_tensor_plan(plan,
						MPSGraphTensorBindingKind::kYoloBoundaryOutput,
						layer_index);
					if (not boundary_plan)
					{
						result.reason = "missing boundary tensor plan for terminal output layer #" + std::to_string(layer_index);
						return result;
					}
					append_forward_result(state,
						it->second,
						*boundary_plan);
				}

				if (state.forward_targets.count == 0 or state.feed_shapes.count == 0)
				{
					result.reason = "forward graph did not produce targets or feeds";
					return result;
				}

				MPSGraphCompilationDescriptor *compilation_descriptor = [[MPSGraphCompilationDescriptor alloc] init];
				compilation_descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
				compilation_descriptor.waitForCompilationCompletion = YES;
				compilation_descriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathNone;
				MPSGraphExecutable *forward_executable = [state.graph compileWithDevice:graph_device
					feeds:state.feed_shapes
					targetTensors:state.forward_targets
					targetOperations:nil
					compilationDescriptor:compilation_descriptor];
				if (not forward_executable)
				{
					result.reason = "MPSGraph forward executable compilation failed";
					return result;
				}
				forward_executable.options = MPSGraphOptionsNone;

				std::string order_reason;
				std::vector<MPSGraphTensorPlan> ordered_forward_feed_plans;
				std::vector<MPSGraphTensorPlan> ordered_forward_result_plans;
				if (not order_tensor_plans_for_executable(forward_executable.feedTensors,
						state.forward_feed_tensors,
						state.forward_feed_plans,
						"forward feed",
						ordered_forward_feed_plans,
						order_reason) or
					not order_tensor_plans_for_executable(forward_executable.targetTensors,
						state.forward_result_tensors,
						state.forward_result_plans,
						"forward target",
						ordered_forward_result_plans,
						order_reason))
				{
					result.reason = order_reason;
					return result;
				}

				result.forward_ready = true;
				result.forward_executable = Detail::ObjcStrong(forward_executable);
				result.forward_feed_plans = std::move(ordered_forward_feed_plans);
				result.forward_result_plans = std::move(ordered_forward_result_plans);
				result.reason = "forward executable compiled; backward executable unavailable";
				return result;
			}
		}

		MPSGraphExecutableBuildResult build_backward_terminal_heads_executable(Network & net, const MPSGraphSegmentPlan & plan)
		{
			MPSGraphExecutableBuildResult result;

			id<MTLDevice> device = (__bridge id<MTLDevice>)runtime_metal_device_handle();
			if (not device)
			{
				result.reason = "Metal device unavailable for MPSGraph backward compile";
				return result;
			}

			MPSGraphDevice *graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
			if (not graph_device)
			{
				result.reason = "failed to create MPSGraph device for backward compile";
				return result;
			}

			@autoreleasepool
			{
				BackwardBuildState state;
				state.graph = [[MPSGraph alloc] init];
				state.graph.options = MPSGraphOptionsNone;
				state.feed_shapes = [NSMutableDictionary dictionary];
				state.backward_targets = [NSMutableArray array];
				std::unordered_map<int, MPSGraphTensor *> output_grad_by_layer;
				const auto bytes_for_float_shape = [](const std::vector<int64_t> & shape) -> size_t
				{
					size_t elements = 1;
					for (const auto dim : shape)
					{
						elements *= static_cast<size_t>(std::max<int64_t>(1, dim));
					}
					return elements * sizeof(float);
				};
				const auto reshape_to_channels = [&](MPSGraphTensor *tensor, const int channels, const std::string & name) -> MPSGraphTensor *
				{
					return [state.graph reshapeTensor:tensor
						withShape:shape_for_channel_parameter(channels)
						name:ns_string(name)];
				};
				const auto reduce_channels = [&](MPSGraphTensor *tensor, const int channels, const std::string & name) -> MPSGraphTensor *
				{
					MPSGraphTensor *reduced = [state.graph reductionSumWithTensor:tensor
						axes:batchnorm_axes_nchw()
						name:ns_string(name + ":reduce")];
					return [state.graph reshapeTensor:reduced
						withShape:@[ @(channels) ]
						name:ns_string(name)];
				};
				const auto append_backward_result = [&](MPSGraphTensor *tensor, const MPSGraphTensorPlan & plan_tensor)
				{
					tensor = cast_tensor_to_tensor_plan(state.graph,
						tensor,
						plan_tensor,
						tensor_debug_label(plan_tensor) + ":backward_result_cast");
					[state.backward_targets addObject:tensor];
					state.backward_result_tensors.push_back(tensor);
					state.backward_result_plans.push_back(plan_tensor);
				};
				const auto accumulate_output_grad = [&](const int layer_index,
														MPSGraphTensor *gradient,
														const std::string & name)
				{
					if (not gradient or layer_index < 0)
					{
						return;
					}
					const auto it = output_grad_by_layer.find(layer_index);
					if (it == output_grad_by_layer.end())
					{
						output_grad_by_layer.emplace(layer_index, gradient);
						return;
					}
					it->second = [state.graph additionWithPrimaryTensor:it->second
						secondaryTensor:gradient
						name:ns_string(name)];
				};
				const auto append_or_accumulate_input_grad = [&](const int source_layer_index,
																 const int consumer_layer_index,
																 const std::vector<int64_t> & shape_nchw,
																 const size_t bytes,
																 MPSGraphTensor *input_grad)
				{
					if (not input_grad)
					{
						return;
					}
					if (source_layer_index < 0)
					{
						MPSGraphTensorPlan input_grad_plan;
						input_grad_plan.binding_kind = MPSGraphTensorBindingKind::kNetworkInputDelta;
						input_grad_plan.owner_layer_index = -1;
						input_grad_plan.source_layer_index = consumer_layer_index;
						input_grad_plan.shape_nchw = shape_nchw;
						input_grad_plan.bytes = bytes;
						input_grad_plan.materialize_to_darknet_buffer = false;
						append_backward_result(input_grad, input_grad_plan);
						return;
					}
					if (find_node_plan(plan, source_layer_index))
					{
						accumulate_output_grad(source_layer_index,
							input_grad,
							node_key(consumer_layer_index, "input_grad_accum"));
						return;
					}

					MPSGraphTensorPlan input_grad_plan;
					input_grad_plan.binding_kind = MPSGraphTensorBindingKind::kLayerDelta;
					input_grad_plan.owner_layer_index = source_layer_index;
					input_grad_plan.source_layer_index = consumer_layer_index;
					input_grad_plan.shape_nchw = shape_nchw;
					input_grad_plan.bytes = bytes;
					input_grad_plan.materialize_to_darknet_buffer = true;
					append_backward_result(input_grad, input_grad_plan);
				};
				const auto build_conv_backward = [&](const MPSGraphNodePlan & node,
													 MPSGraphTensor *incoming_delta,
													 const bool seed_from_yolo_boundary,
													 MPSGraphTensor *__strong & input_grad_out,
													 std::string & reason) -> bool
				{
					const int layer_index = node.darknet_layer_index;
					const Layer & layer = net.layers[layer_index];
					if (node.op_kind != MPSGraphNodeKind::kConvBnAct or
						(node.activation != MPSGraphActivationKind::kLinear and
							node.activation != MPSGraphActivationKind::kLeaky and
							node.activation != MPSGraphActivationKind::kLogistic))
					{
						reason =
							"conv backward expects supported conv activation at layer #" + std::to_string(layer_index) +
							" op=" + std::to_string(static_cast<int>(node.op_kind == MPSGraphNodeKind::kConvBnAct)) +
							" act=" + activation_name(node.activation) +
							" incoming=" + std::string(incoming_delta ? "yes" : "no");
						return false;
					}

					if (not incoming_delta and not seed_from_yolo_boundary)
					{
						reason = "conv backward has no incoming gradient for layer #" + std::to_string(layer_index);
						return false;
					}

					const int source_layer_index = not node.input_node_indices.empty()
						? node.input_node_indices.front()
						: layer_index - 1;
					const MPSGraphNodePlan *source_node = source_layer_index >= 0
						? find_node_plan(plan, source_layer_index)
						: nullptr;
					const MPSGraphTensorPlan *source_plan = source_node
						? find_tensor_plan(*source_node, MPSGraphTensorBindingKind::kLayerOutput)
						: nullptr;
					const MPSGraphTensorPlan *network_input_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kNetworkInput);
					const auto *weights_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kWeights);
					const auto *weight_grad_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kWeightGrad);
					const auto *bias_grad_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kBiasGrad);
					if (not source_plan and network_input_plan)
					{
						source_plan = network_input_plan;
					}
					if (not source_plan or not weights_plan or not weight_grad_plan or not bias_grad_plan)
					{
						reason = "conv backward plans are incomplete for layer #" + std::to_string(layer_index);
						return false;
					}

					if (seed_from_yolo_boundary)
					{
						const auto *delta_plan = find_boundary_tensor_plan(plan, MPSGraphTensorBindingKind::kYoloBoundaryDelta, layer_index);
						if (not delta_plan)
						{
							reason = "missing YOLO boundary delta plan for layer #" + std::to_string(layer_index);
							return false;
						}
						incoming_delta = add_backward_placeholder(state,
							node_key(layer_index, "yolo_delta"),
							delta_plan->shape_nchw,
							node_debug_label(node) + ":yolo_delta",
							*delta_plan);
					}
					MPSGraphTensor *source = add_backward_placeholder(state,
						source_layer_index >= 0 ? node_key(source_layer_index, "producer_output") : "network_input",
						source_plan->shape_nchw,
						node_debug_label(node) + ":source_tape",
						*source_plan);
					MPSGraphTensor *weights = add_backward_placeholder(state,
						node_key(layer_index, "weights"),
						weights_plan->shape_nchw,
						node_debug_label(node) + ":weights",
						*weights_plan);
					MPSGraphTensor *delta_after_activation = incoming_delta;
					MPSGraphTensor *scale_grad = nil;
					if (node.activation == MPSGraphActivationKind::kLeaky or
						node.activation == MPSGraphActivationKind::kLogistic)
					{
						const MPSGraphTensorPlan *activation_tape_plan =
							find_tensor_plan(node, MPSGraphTensorBindingKind::kLayerOutput);
						if (not activation_tape_plan and seed_from_yolo_boundary)
						{
							activation_tape_plan = find_boundary_tensor_plan(
								plan, MPSGraphTensorBindingKind::kYoloBoundaryOutput, layer_index);
						}
						if (not activation_tape_plan)
						{
							reason = "missing activation tape plan for layer #" + std::to_string(layer_index);
							return false;
						}
						MPSGraphTensor *activation_tape = add_backward_placeholder(state,
							node_key(layer_index, "activated_output"),
							activation_tape_plan->shape_nchw,
							node_debug_label(node) + ":activation_tape",
							*activation_tape_plan);
						if (activation_tape_plan->data_type != MPSGraphTensorDataType::kFloat32)
						{
							activation_tape = cast_tensor_to_data_type(state.graph,
								activation_tape,
								MPSGraphTensorDataType::kFloat32,
								node_debug_label(node) + ":activation_tape_fp32");
						}
						if (node.activation == MPSGraphActivationKind::kLeaky)
						{
							MPSGraphTensor *alpha = add_scalar_constant(
								state.graph,
								darknet_leaky_alpha());
							delta_after_activation = [state.graph
								leakyReLUGradientWithIncomingGradient:incoming_delta
								sourceTensor:activation_tape
								alphaTensor:alpha
								name:ns_string(node_debug_label(node) + ":delta_after_leaky")];
						}
						else
						{
							MPSGraphTensor *activation_grad = nil;
							if (node.activation == MPSGraphActivationKind::kLogistic)
							{
								MPSGraphTensor *one = add_scalar_constant(state.graph, 1.0);
								MPSGraphTensor *one_minus_sigmoid = [state.graph subtractionWithPrimaryTensor:one
									secondaryTensor:activation_tape
									name:ns_string(node_debug_label(node) + ":logistic_one_minus")];
								activation_grad = [state.graph multiplicationWithPrimaryTensor:activation_tape
									secondaryTensor:one_minus_sigmoid
									name:ns_string(node_debug_label(node) + ":logistic_grad")];
							}
							delta_after_activation = [state.graph multiplicationWithPrimaryTensor:incoming_delta
								secondaryTensor:activation_grad
								name:ns_string(node_debug_label(node) + ":delta_after_activation")];
						}
					}

					MPSGraphTensor *normalized_delta = delta_after_activation;
					MPSGraphTensor *bias_grad = nil;
					if (node.batch_normalize)
					{
						const auto *scale_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kBatchNormScale);
						const auto *scale_grad_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kBatchNormScaleGrad);
						const auto *batch_input_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kBatchNormInputTape);
						const auto *batch_mean_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kBatchMeanTape);
						const auto *batch_variance_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kBatchVarianceTape);
						if (not scale_plan or not scale_grad_plan or not batch_input_plan or not batch_mean_plan or
							not batch_variance_plan)
						{
							reason = "batchnorm backward plans are incomplete for layer #" + std::to_string(layer_index);
							return false;
						}

						MPSGraphTensor *gamma = add_backward_placeholder(state,
							node_key(layer_index, "bn_scale"),
							scale_plan->shape_nchw,
							node_debug_label(node) + ":bn_scale",
							*scale_plan);
						MPSGraphTensor *batch_input = add_backward_placeholder(state,
							node_key(layer_index, "batch_input"),
							batch_input_plan->shape_nchw,
							node_debug_label(node) + ":batch_input",
							*batch_input_plan);
						MPSGraphTensor *batch_mean = add_backward_placeholder(state,
							node_key(layer_index, "batch_mean"),
							batch_mean_plan->shape_nchw,
							node_debug_label(node) + ":batch_mean",
							*batch_mean_plan);
						MPSGraphTensor *batch_variance = add_backward_placeholder(state,
							node_key(layer_index, "batch_variance"),
							batch_variance_plan->shape_nchw,
							node_debug_label(node) + ":batch_variance",
							*batch_variance_plan);
						if (batch_input_plan->data_type != MPSGraphTensorDataType::kFloat32)
						{
							batch_input = cast_tensor_to_data_type(state.graph,
								batch_input,
								MPSGraphTensorDataType::kFloat32,
								node_debug_label(node) + ":batch_input_fp32");
						}
						const int channels = layer.n;
						MPSGraphTensor *gamma_4d = reshape_to_channels(gamma, channels, node_debug_label(node) + ":gamma_reshape");
						MPSGraphTensor *native_scale_grad = [state.graph
							normalizationGammaGradientWithIncomingGradientTensor:delta_after_activation
							sourceTensor:batch_input
							meanTensor:batch_mean
							varianceTensor:batch_variance
							reductionAxes:batchnorm_axes_nchw()
							epsilon:batchnorm_backward_epsilon()
							name:ns_string(node_debug_label(node) + ":scale_grad")];
						MPSGraphTensor *native_bias_grad = [state.graph
							normalizationBetaGradientWithIncomingGradientTensor:delta_after_activation
							sourceTensor:batch_input
							reductionAxes:batchnorm_axes_nchw()
							name:ns_string(node_debug_label(node) + ":bias_grad")];
						normalized_delta = [state.graph
							normalizationGradientWithIncomingGradientTensor:delta_after_activation
							sourceTensor:batch_input
							meanTensor:batch_mean
							varianceTensor:batch_variance
							gammaTensor:gamma_4d
							gammaGradientTensor:native_scale_grad
							betaGradientTensor:native_bias_grad
							reductionAxes:batchnorm_axes_nchw()
							epsilon:batchnorm_backward_epsilon()
							name:ns_string(node_debug_label(node) + ":normalized_delta")];
						scale_grad = [state.graph reshapeTensor:native_scale_grad
							withShape:@[ @(channels) ]
							name:ns_string(node_debug_label(node) + ":scale_grad_reshape")];
						bias_grad = [state.graph reshapeTensor:native_bias_grad
							withShape:@[ @(channels) ]
							name:ns_string(node_debug_label(node) + ":bias_grad_reshape")];

						append_backward_result(scale_grad, *scale_grad_plan);
					}
					else
					{
						bias_grad = reduce_channels(delta_after_activation, layer.n, node_debug_label(node) + ":bias_grad");
					}

					auto *descriptor = [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:node.stride_x
						strideInY:node.stride_y
						dilationRateInX:1
						dilationRateInY:1
						groups:std::max(1, node.groups)
						paddingLeft:node.pad_left
						paddingRight:node.pad_right
						paddingTop:node.pad_top
						paddingBottom:node.pad_bottom
						paddingStyle:MPSGraphPaddingStyleExplicit
						dataLayout:MPSGraphTensorNamedDataLayoutNCHW
						weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
					if (not descriptor)
					{
						reason = "failed to create backward convolution descriptor for layer #" + std::to_string(layer_index);
						return false;
					}

					MPSGraphTensor *weight_grad = [state.graph convolution2DWeightsGradientWithIncomingGradientTensor:normalized_delta
						sourceTensor:source
						outputShape:shape_from_vector(weight_grad_plan->shape_nchw)
						forwardConvolutionDescriptor:descriptor
						name:ns_string(node_debug_label(node) + ":weight_grad")];
					MPSGraphTensor *input_grad = [state.graph convolution2DDataGradientWithIncomingGradientTensor:normalized_delta
						weightsTensor:weights
						outputShape:shape_from_vector(source_plan->shape_nchw)
						forwardConvolutionDescriptor:descriptor
						name:ns_string(node_debug_label(node) + ":input_grad")];
					append_backward_result(weight_grad, *weight_grad_plan);
					append_backward_result(bias_grad, *bias_grad_plan);
					input_grad_out = input_grad;
					return true;
				};
				const auto build_maxpool_backward = [&](const MPSGraphNodePlan & node,
														MPSGraphTensor *incoming_delta,
														std::string & reason) -> bool
				{
					const int layer_index = node.darknet_layer_index;
					if (not incoming_delta or node.op_kind != MPSGraphNodeKind::kMaxPool)
					{
						reason = "maxpool backward expects maxpool node";
						return false;
					}
					const int source_layer_index = not node.input_node_indices.empty()
						? node.input_node_indices.front()
						: layer_index - 1;
					if (source_layer_index < 0)
					{
						reason = "maxpool source layer is unavailable for layer #" + std::to_string(layer_index);
						return false;
					}
					const auto *indices_plan = find_tensor_plan(node, MPSGraphTensorBindingKind::kMaxPoolIndicesTape);
					if (not indices_plan)
					{
						reason = "missing maxpool indices plan for layer #" + std::to_string(layer_index);
						return false;
					}
					MPSGraphTensor *indices = add_backward_placeholder(state,
						node_key(layer_index, "maxpool_indices"),
						indices_plan->shape_nchw,
						node_debug_label(node) + ":maxpool_indices",
						*indices_plan);
					auto *descriptor = [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:node.kernel_w
						kernelHeight:node.kernel_h
						strideInX:node.stride_x
						strideInY:node.stride_y
						dilationRateInX:1
						dilationRateInY:1
						paddingLeft:node.pad_left
						paddingRight:node.pad_right
						paddingTop:node.pad_top
						paddingBottom:node.pad_bottom
						paddingStyle:MPSGraphPaddingStyleExplicit
						dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
					if (not descriptor)
					{
						reason = "failed to create maxpool backward descriptor for layer #" + std::to_string(layer_index);
						return false;
					}
					descriptor.ceilMode = NO;
					descriptor.returnIndicesMode = MPSGraphPoolingReturnIndicesGlobalFlatten2D;
					descriptor.returnIndicesDataType = MPSDataTypeInt32;
					const auto *source_node = find_node_plan(plan, source_layer_index);
					const std::vector<int64_t> source_shape = source_node
						? source_node->output_shape_nchw
						: output_shape_for_layer(net.layers[source_layer_index]);
					MPSGraphTensor *source_grad = [state.graph maxPooling2DGradientWithGradientTensor:incoming_delta
						indicesTensor:indices
						outputShape:shape_from_vector(source_shape)
						descriptor:descriptor
						name:ns_string(node_debug_label(node) + ":input_grad")];
					append_or_accumulate_input_grad(source_layer_index,
						layer_index,
						source_shape,
						bytes_for_float_shape(source_shape),
						source_grad);
					return true;
				};
				const auto build_upsample_backward = [&](const MPSGraphNodePlan & node,
														 MPSGraphTensor *incoming_delta,
														 std::string & reason) -> bool
				{
					const int layer_index = node.darknet_layer_index;
					if (not incoming_delta or node.op_kind != MPSGraphNodeKind::kUpsample)
					{
						reason = "upsample backward expects upsample node";
						return false;
					}
					const int source_layer_index = not node.input_node_indices.empty()
						? node.input_node_indices.front()
						: layer_index - 1;
					if (source_layer_index < 0)
					{
						reason = "upsample source layer is unavailable for layer #" + std::to_string(layer_index);
						return false;
					}
					const auto *source_node = find_node_plan(plan, source_layer_index);
					const std::vector<int64_t> source_shape = source_node
						? source_node->output_shape_nchw
						: output_shape_for_layer(net.layers[source_layer_index]);
					if (source_shape.size() != 4 or node.upsample_scale <= 0)
					{
						reason = "upsample backward requires a rank-4 source and positive scale";
						return false;
					}
					const int64_t scale = node.upsample_scale;
					const std::vector<int64_t> expanded_shape = {
						source_shape[0], source_shape[1], source_shape[2], scale, source_shape[3], scale
					};
					MPSGraphTensor *expanded_delta = [state.graph reshapeTensor:incoming_delta
						withShape:shape_from_vector(expanded_shape)
						name:ns_string(node_debug_label(node) + ":delta_blocks")];
					MPSGraphTensor *source_grad = [state.graph reductionSumWithTensor:expanded_delta
						axes:@[ @3, @5 ]
						name:ns_string(node_debug_label(node) + ":input_grad_reduce")];
					source_grad = [state.graph reshapeTensor:source_grad
						withShape:shape_from_vector(source_shape)
						name:ns_string(node_debug_label(node) + ":input_grad")];
					append_or_accumulate_input_grad(source_layer_index,
						layer_index,
						source_shape,
						bytes_for_float_shape(source_shape),
						source_grad);
					return true;
				};
				const auto build_route_group_slice_backward = [&](const MPSGraphNodePlan & node,
																  MPSGraphTensor *incoming_delta,
																  std::string & reason) -> bool
				{
					const int layer_index = node.darknet_layer_index;
					if (not incoming_delta or node.op_kind != MPSGraphNodeKind::kRouteGroupSlice or node.input_node_indices.empty())
					{
						reason = "route-group backward expects a sliced route node";
						return false;
					}
					const int source_layer_index = node.input_node_indices.front();
					const auto *source_node = find_node_plan(plan, source_layer_index);
					const std::vector<int64_t> source_shape = source_node
						? source_node->output_shape_nchw
						: output_shape_for_layer(net.layers[source_layer_index]);
					NSMutableArray<MPSGraphTensor *> *parts = [NSMutableArray arrayWithCapacity:3];
					const int leading_channels = node.slice_channel_offset;
					const int trailing_channels = static_cast<int>(source_shape[1]) - leading_channels - node.slice_channel_count;
					if (leading_channels > 0)
					{
						[parts addObject:[state.graph constantWithScalar:0.0
							shape:@[ @(source_shape[0]), @(leading_channels), @(source_shape[2]), @(source_shape[3]) ]
							dataType:MPSDataTypeFloat32]];
					}
					[parts addObject:incoming_delta];
					if (trailing_channels > 0)
					{
						[parts addObject:[state.graph constantWithScalar:0.0
							shape:@[ @(source_shape[0]), @(trailing_channels), @(source_shape[2]), @(source_shape[3]) ]
							dataType:MPSDataTypeFloat32]];
					}
					MPSGraphTensor *source_grad = parts.count == 1 ? parts[0] :
						[state.graph concatTensors:parts
							dimension:1
							name:ns_string(node_debug_label(node) + ":input_grad")];
					append_or_accumulate_input_grad(source_layer_index,
						layer_index,
						source_shape,
						bytes_for_float_shape(source_shape),
						source_grad);
					return true;
				};
				const auto build_route_concat_backward = [&](const MPSGraphNodePlan & node,
															 MPSGraphTensor *incoming_delta,
															 std::string & reason) -> bool
				{
					const int layer_index = node.darknet_layer_index;
					if (not incoming_delta or node.op_kind != MPSGraphNodeKind::kRouteConcat or node.input_node_indices.empty())
					{
						reason = "route-concat backward expects concat route node";
						return false;
					}
					int channel_offset = 0;
					for (size_t input_position = 0; input_position < node.input_node_indices.size(); ++input_position)
					{
						const int source_layer_index = node.input_node_indices[input_position];
						const auto *source_node = find_node_plan(plan, source_layer_index);
						const std::vector<int64_t> source_shape = source_node
							? source_node->output_shape_nchw
							: output_shape_for_layer(net.layers[source_layer_index]);
						const int channel_count = static_cast<int>(source_shape.size() > 1 ? source_shape[1] : 0);
						MPSGraphTensor *slice = [state.graph sliceTensor:incoming_delta
							dimension:1
							start:channel_offset
							length:channel_count
							name:ns_string(node_debug_label(node) + ":slice_grad_" + std::to_string(input_position) +
																			 "_from_" + std::to_string(source_layer_index))];
						append_or_accumulate_input_grad(source_layer_index,
							layer_index,
							source_shape,
							bytes_for_float_shape(source_shape),
							slice);
						channel_offset += channel_count;
					}
					return true;
				};
				for (auto node_it = plan.nodes.rbegin(); node_it != plan.nodes.rend(); ++node_it)
				{
					const auto & node = *node_it;
					MPSGraphTensor *incoming_delta = nil;
					const auto grad_it = output_grad_by_layer.find(node.darknet_layer_index);
					if (grad_it != output_grad_by_layer.end())
					{
						incoming_delta = grad_it->second;
					}
					const bool seed_from_yolo_boundary =
						incoming_delta == nil and
						std::find(plan.terminal_output_layers.begin(), plan.terminal_output_layers.end(), node.darknet_layer_index) != plan.terminal_output_layers.end();
					if (not incoming_delta and not seed_from_yolo_boundary)
					{
						continue;
					}

					std::string build_reason;
					switch (node.op_kind)
					{
						case MPSGraphNodeKind::kConvBnAct:
						{
							MPSGraphTensor *input_grad = nil;
							if (not build_conv_backward(node, incoming_delta, seed_from_yolo_boundary, input_grad, build_reason))
							{
								result.reason = build_reason;
								return result;
							}
							const int source_layer_index = not node.input_node_indices.empty()
								? node.input_node_indices.front()
								: node.darknet_layer_index - 1;
							if (source_layer_index >= 0)
							{
								const auto *source_node = find_node_plan(plan, source_layer_index);
								const std::vector<int64_t> source_shape = source_node
									? source_node->output_shape_nchw
									: output_shape_for_layer(net.layers[source_layer_index]);
								append_or_accumulate_input_grad(source_layer_index,
									node.darknet_layer_index,
									source_shape,
									bytes_for_float_shape(source_shape),
									input_grad);
							}
							break;
						}
						case MPSGraphNodeKind::kRouteConcat:
							if (not build_route_concat_backward(node, incoming_delta, build_reason))
							{
								result.reason = build_reason;
								return result;
							}
							break;
						case MPSGraphNodeKind::kRouteGroupSlice:
							if (not build_route_group_slice_backward(node, incoming_delta, build_reason))
							{
								result.reason = build_reason;
								return result;
							}
							break;
						case MPSGraphNodeKind::kMaxPool:
							if (not build_maxpool_backward(node, incoming_delta, build_reason))
							{
								result.reason = build_reason;
								return result;
							}
							break;
						case MPSGraphNodeKind::kUpsample:
							if (not build_upsample_backward(node, incoming_delta, build_reason))
							{
								result.reason = build_reason;
								return result;
							}
							break;
						case MPSGraphNodeKind::kYoloBoundaryOutput:
							break;
					}
				}
				if (state.backward_targets.count == 0 or state.feed_shapes.count == 0)
				{
					result.reason = "backward graph did not produce targets or feeds";
					return result;
				}

				MPSGraphCompilationDescriptor *compilation_descriptor = [[MPSGraphCompilationDescriptor alloc] init];
				compilation_descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
				compilation_descriptor.waitForCompilationCompletion = YES;
				compilation_descriptor.reducedPrecisionFastMath = MPSGraphReducedPrecisionFastMathNone;
				MPSGraphExecutable *backward_executable = [state.graph compileWithDevice:graph_device
					feeds:state.feed_shapes
					targetTensors:state.backward_targets
					targetOperations:nil
					compilationDescriptor:compilation_descriptor];
				if (not backward_executable)
				{
					result.reason = "MPSGraph backward executable compilation failed";
					return result;
				}
				backward_executable.options = MPSGraphOptionsNone;

				std::string order_reason;
				std::vector<MPSGraphTensorPlan> ordered_backward_feed_plans;
				std::vector<MPSGraphTensorPlan> ordered_backward_result_plans;
				if (not order_tensor_plans_for_executable(backward_executable.feedTensors,
						state.backward_feed_tensors,
						state.backward_feed_plans,
						"backward feed",
						ordered_backward_feed_plans,
						order_reason) or
					not order_tensor_plans_for_executable(backward_executable.targetTensors,
						state.backward_result_tensors,
						state.backward_result_plans,
						"backward target",
						ordered_backward_result_plans,
						order_reason))
				{
					result.reason = order_reason;
					return result;
				}

				result.backward_ready = true;
				result.backward_executable = Detail::ObjcStrong(backward_executable);
				result.backward_feed_plans = std::move(ordered_backward_feed_plans);
				result.backward_result_plans = std::move(ordered_backward_result_plans);
				result.reason = "MPSGraph backward executable compiled for the supported YOLO graph segment";
				return result;
			}
		}
	}

		MPSGraphExecutableBuildResult build_training_graph_executables(Network & net, const MPSGraphSegmentPlan & plan)
	{
		if (not plan.supported)
		{
			MPSGraphExecutableBuildResult result;
			result.reason = plan.reason.empty() ? "MPSGraph plan is unsupported" : plan.reason;
			return result;
		}
		auto result = build_forward_executable(net, plan);
		if (not result.forward_ready)
		{
			return result;
		}
		auto backward = build_backward_terminal_heads_executable(net, plan);
		result.backward_ready = backward.backward_ready;
		result.backward_executable = std::move(backward.backward_executable);
		result.backward_feed_plans = std::move(backward.backward_feed_plans);
		result.backward_result_plans = std::move(backward.backward_result_plans);
		if (backward.backward_ready)
		{
			result.reason = "forward and full-trunk backward executables compiled";
		}
		else
		{
			result.reason = backward.reason.empty()
				? "terminal-head backward executable lowering did not compile yet"
				: backward.reason;
		}
		return result;
	}
}
