#include "apple_training_plan.hpp"

#include "darknet_internal.hpp"
#include "darknet_enums.hpp"
#include "darknet_layers.hpp"
#include "darknet_network.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>

namespace Darknet::Apple
{
	namespace
	{
		bool is_detector_head_layer_type(const Darknet::ELayerType type)
		{
			return type == Darknet::ELayerType::YOLO;
		}

		size_t element_count(const std::vector<int64_t> & shape)
		{
			size_t total = 1;
			for (const auto dim : shape)
			{
				total *= static_cast<size_t>(std::max<int64_t>(dim, 0));
			}
			return total;
		}

		size_t bytes_per_element(const MPSGraphTensorDataType data_type)
		{
			switch (data_type)
			{
				case MPSGraphTensorDataType::kInt32: return sizeof(std::int32_t);
				case MPSGraphTensorDataType::kFloat32:
				default: return sizeof(float);
			}
		}

		std::vector<int64_t> nchw_shape_for_layer_output(const Layer & l)
		{
			return {
				static_cast<int64_t>(l.batch),
				static_cast<int64_t>(l.out_c),
				static_cast<int64_t>(l.out_h),
				static_cast<int64_t>(l.out_w),
			};
		}

		std::vector<int64_t> nchw_shape_for_layer_input(const Layer & l)
		{
			return {
				static_cast<int64_t>(l.batch),
				static_cast<int64_t>(l.c),
				static_cast<int64_t>(l.h),
				static_cast<int64_t>(l.w),
			};
		}

		void split_total_padding(const int total_padding, int *leading, int *trailing)
		{
			const int clamped_padding = std::max(0, total_padding);
			const int leading_padding = clamped_padding / 2;
			if (leading)
			{
				*leading = leading_padding;
			}
			if (trailing)
			{
				*trailing = clamped_padding - leading_padding;
			}
		}

		MPSGraphActivationKind activation_kind_for_layer(const Layer & l)
		{
			switch (l.activation)
			{
				case LEAKY: return MPSGraphActivationKind::kLeaky;
				case LOGISTIC: return MPSGraphActivationKind::kLogistic;
				case LINEAR:
				default:
					return MPSGraphActivationKind::kLinear;
			}
		}

		bool activation_requires_tape(const MPSGraphActivationKind activation)
		{
			return activation == MPSGraphActivationKind::kLeaky or
				activation == MPSGraphActivationKind::kLogistic;
		}

		MPSGraphTensorPlan make_tensor_plan(const MPSGraphTensorBindingKind kind,
			const int owner_layer_index,
			const int source_layer_index,
			const std::vector<int64_t> & shape_nchw,
			const bool forward_result,
			const bool materialize_to_darknet_buffer,
			const size_t explicit_bytes = 0,
			const MPSGraphTensorDataType data_type = MPSGraphTensorDataType::kFloat32)
		{
			MPSGraphTensorPlan plan;
			plan.binding_kind = kind;
			plan.owner_layer_index = owner_layer_index;
			plan.source_layer_index = source_layer_index;
			plan.shape_nchw = shape_nchw;
			plan.data_type = data_type;
			plan.bytes = explicit_bytes ? explicit_bytes : element_count(shape_nchw) * bytes_per_element(data_type);
			plan.forward_result = forward_result;
			plan.materialize_to_darknet_buffer = materialize_to_darknet_buffer;
			return plan;
		}

		bool conv_supported(const Layer & l, std::string & reason)
		{
			if (l.type != Darknet::ELayerType::CONVOLUTIONAL)
			{
				reason = "layer is not convolutional";
				return false;
			}
			if (l.binary or l.xnor or l.deform or l.antialiasing or l.assisted_excitation)
			{
				reason = "unsupported convolution variant";
				return false;
			}
			if (l.share_layer)
			{
				reason = "share_index/shared convolution parameters are not supported";
				return false;
			}
			if (l.coordconv)
			{
				reason = "coordconv is not supported";
				return false;
			}
			if (l.grad_centr)
			{
				reason = "grad_centr is not supported";
				return false;
			}
			if (l.reverse != 0.0f)
			{
				reason = "reverse convolution updates are not supported";
				return false;
			}
			if (l.dilation != 1)
			{
				reason = "dilated convolution is not supported by Apple MPSGraph training";
				return false;
			}
			if (l.groups < 1 or l.c % l.groups != 0 or l.n % l.groups != 0)
			{
				reason = "invalid grouped convolution geometry";
				return false;
			}
			if (not (l.size == 1 or l.size == 3))
			{
				reason = "only 1x1 and 3x3 conv are supported";
				return false;
			}
			if (not (l.stride_x == l.stride_y and (l.stride_x == 1 or l.stride_x == 2)))
			{
				reason = "only stride 1 and stride 2 conv are supported";
				return false;
			}
			if (not (l.activation == LEAKY or l.activation == LOGISTIC or l.activation == LINEAR))
			{
				reason = "only leaky, logistic, and linear activations are supported";
				return false;
			}
			if (not (l.batch_normalize == 0 or l.batch_normalize == 1))
			{
				reason = "only batch_normalize=0/1 are supported";
				return false;
			}
			reason = "ok";
			return true;
		}

		bool route_supported(const Layer & l, std::string & reason)
		{
			if (l.n <= 0 or not l.input_layers)
			{
				reason = "route input layers are unavailable";
				return false;
			}
			if (l.groups > 1 and (l.group_id < 0 or l.group_id >= l.groups))
			{
				reason = "invalid route group selection";
				return false;
			}
			if (l.groups > 1 and l.n != 1)
			{
				reason = "grouped routes with multiple inputs are not supported by Apple MPSGraph training";
				return false;
			}
			reason = "ok";
			return true;
		}

		bool maxpool_supported(const Layer & l, std::string & reason)
		{
			if (l.maxpool_depth or l.antialiasing or l.avgpool)
			{
				reason = "only plain 2D maxpool is supported";
				return false;
			}
			if (l.maxpool_zero_nonmax)
			{
				reason = "maxpool_zero_nonmax is not supported";
				return false;
			}
			if (l.size <= 0 or l.stride_x <= 0 or l.stride_y <= 0)
			{
				reason = "maxpool requires positive size and stride";
				return false;
			}
			const int expected_out_w = (l.w + l.pad - l.size) / l.stride_x + 1;
			const int expected_out_h = (l.h + l.pad - l.size) / l.stride_y + 1;
			if (expected_out_w != l.out_w or expected_out_h != l.out_h)
			{
				reason = "maxpool output shape does not match Darknet floor semantics";
				return false;
			}
			const bool tiny_downsample =
				l.size == 2 and l.stride_x == 2 and l.stride_y == 2;
			const bool spp_pool =
				(l.size == 5 or l.size == 9 or l.size == 13) and
				l.stride_x == 1 and
				l.stride_y == 1 and
				l.pad == (l.size - 1);
			if (not (tiny_downsample or spp_pool))
			{
				reason = "only 2x2 stride-2 maxpool and YOLOv4 SPP maxpool are supported";
				return false;
			}
			reason = "ok";
			return true;
		}

		bool upsample_supported(const Layer & l, std::string & reason)
		{
			if (l.reverse or l.stride != 2)
			{
				reason = "only nearest upsample stride=2 is supported";
				return false;
			}
			if (std::fabs(l.scale - 1.0f) > 1e-6f)
			{
				reason = "upsample scale values other than 1 are not supported";
				return false;
			}
			reason = "ok";
			return true;
		}

		bool layer_supported(const Network & net, const int layer_index, const Layer & l, std::string & reason)
		{
			if (l.onlyforward)
			{
				reason = "onlyforward is not supported";
				return false;
			}
			if (l.stopbackward)
			{
				reason = "stopbackward is not supported";
				return false;
			}
			if (l.clip != 0.0f)
			{
				reason = "clip is not supported";
				return false;
			}
			switch (l.type)
			{
				case Darknet::ELayerType::CONVOLUTIONAL: return conv_supported(l, reason);
				case Darknet::ELayerType::SHORTCUT:
					(void)net;
					(void)layer_index;
					(void)l;
					reason = "shortcut layers are not supported by Apple MPSGraph training";
					return false;
				case Darknet::ELayerType::ROUTE: return route_supported(l, reason);
				case Darknet::ELayerType::MAXPOOL: return maxpool_supported(l, reason);
				case Darknet::ELayerType::UPSAMPLE: return upsample_supported(l, reason);
				case Darknet::ELayerType::YOLO:
					if (l.embedding_output)
					{
						reason = "embedding_layer is not supported";
						return false;
					}
					if (l.focal_loss)
					{
						reason = "focal_loss is not supported";
						return false;
					}
					if (l.label_smooth_eps != 0.0f)
					{
						reason = "label_smooth_eps is not supported";
						return false;
					}
					if (l.classes_multipliers or l.map)
					{
						reason = "YOLO class multipliers and remapping are not supported";
						return false;
					}
					if (l.iou_loss != CIOU)
					{
						reason = "only ciou YOLO loss is supported";
						return false;
					}
					if (std::fabs(l.truth_thresh - 1.0f) > 1e-6f)
					{
						reason = "truth_thresh values other than 1 are not supported";
						return false;
					}
					if (l.iou_thresh < 1.0f and l.iou_thresh_kind != IOU)
					{
						reason = "iou_thresh<1 requires iou_thresh_kind=iou";
						return false;
					}
					reason = "ok";
					return true;
				default:
					reason = "layer type is not supported by Apple MPSGraph training";
					return false;
			}
		}

		MPSGraphNodePlan build_node_plan(const Network & net, const int layer_index)
		{
			const Layer & l = net.layers[layer_index];
			MPSGraphNodePlan node;
			node.darknet_layer_index = layer_index;
			node.output_shape_nchw = nchw_shape_for_layer_output(l);

			switch (l.type)
			{
				case Darknet::ELayerType::CONVOLUTIONAL:
				{
					node.op_kind = MPSGraphNodeKind::kConvBnAct;
					node.kernel_h = l.size;
					node.kernel_w = l.size;
					node.stride_y = l.stride_y;
					node.stride_x = l.stride_x;
					node.pad_top = l.pad;
					node.pad_bottom = l.pad;
					node.pad_left = l.pad;
					node.pad_right = l.pad;
					node.groups = std::max(1, l.groups);
					node.batch_normalize = l.batch_normalize == 1;
					node.activation = activation_kind_for_layer(l);
					node.tensors.push_back(make_tensor_plan(
						MPSGraphTensorBindingKind::kWeights,
						layer_index,
						layer_index,
						{
							static_cast<int64_t>(l.n),
							static_cast<int64_t>(std::max(1, l.c / std::max(1, l.groups))),
							static_cast<int64_t>(l.size),
							static_cast<int64_t>(l.size),
						},
						false, false));
					node.tensors.push_back(make_tensor_plan(
						MPSGraphTensorBindingKind::kWeightGrad,
						layer_index,
						layer_index,
						{
							static_cast<int64_t>(l.n),
							static_cast<int64_t>(std::max(1, l.c / std::max(1, l.groups))),
							static_cast<int64_t>(l.size),
							static_cast<int64_t>(l.size),
						},
						false, false));
					node.tensors.push_back(make_tensor_plan(
						MPSGraphTensorBindingKind::kBias,
						layer_index,
						layer_index,
						{static_cast<int64_t>(l.n)},
						false, false));
					node.tensors.push_back(make_tensor_plan(
						MPSGraphTensorBindingKind::kBiasGrad,
						layer_index,
						layer_index,
						{static_cast<int64_t>(l.n)},
						false, false));
					if (node.batch_normalize)
					{
						node.tensors.push_back(make_tensor_plan(MPSGraphTensorBindingKind::kBatchNormScale, layer_index, layer_index, {static_cast<int64_t>(l.n)}, false, false));
						node.tensors.push_back(make_tensor_plan(MPSGraphTensorBindingKind::kBatchNormScaleGrad, layer_index, layer_index, {static_cast<int64_t>(l.n)}, false, false));
						node.tensors.push_back(make_tensor_plan(MPSGraphTensorBindingKind::kBatchNormInputTape, layer_index, layer_index, node.output_shape_nchw, true, false, 0, MPSGraphTensorDataType::kFloat32));
						node.tensors.push_back(make_tensor_plan(MPSGraphTensorBindingKind::kBatchMeanTape, layer_index, layer_index, {1, static_cast<int64_t>(l.n), 1, 1}, true, false));
						node.tensors.push_back(make_tensor_plan(MPSGraphTensorBindingKind::kBatchVarianceTape, layer_index, layer_index, {1, static_cast<int64_t>(l.n), 1, 1}, true, false));
					}
					break;
				}

				case Darknet::ELayerType::ROUTE:
				{
					node.op_kind = (l.groups > 1) ? MPSGraphNodeKind::kRouteGroupSlice : MPSGraphNodeKind::kRouteConcat;
					node.groups = std::max(1, l.groups);
					node.slice_channel_count = l.out_c;
					if (l.groups > 1 and l.n > 0)
					{
						const Layer & src = net.layers[l.input_layers[0]];
						node.slice_channel_count = src.out_c / node.groups;
						node.slice_channel_offset = node.slice_channel_count * std::max(0, l.group_id);
					}
					break;
				}

				case Darknet::ELayerType::MAXPOOL:
				{
					node.op_kind = MPSGraphNodeKind::kMaxPool;
					node.kernel_h = l.size;
					node.kernel_w = l.size;
					node.stride_y = l.stride_y;
					node.stride_x = l.stride_x;
					split_total_padding(l.pad, &node.pad_top, &node.pad_bottom);
					split_total_padding(l.pad, &node.pad_left, &node.pad_right);
					node.tensors.push_back(make_tensor_plan(
						MPSGraphTensorBindingKind::kMaxPoolIndicesTape,
						layer_index,
						layer_index,
						node.output_shape_nchw,
						true,
						false,
						0,
						MPSGraphTensorDataType::kInt32));
					break;
				}

				case Darknet::ELayerType::UPSAMPLE:
				{
					node.op_kind = MPSGraphNodeKind::kUpsample;
					node.upsample_scale = l.stride;
					break;
				}

				case Darknet::ELayerType::YOLO:
				{
					node.op_kind = MPSGraphNodeKind::kYoloBoundaryOutput;
					node.segment_output_to_yolo = true;
					break;
				}

				default:
					break;
			}

			if (layer_index == 0)
			{
				node.segment_input = true;
				node.tensors.push_back(make_tensor_plan(
					MPSGraphTensorBindingKind::kNetworkInput,
					layer_index,
					-1,
					nchw_shape_for_layer_input(l),
					false,
					false));
			}

			return node;
		}

		bool node_output_needed_for_backward(const MPSGraphSegmentPlan & plan, const MPSGraphNodePlan & node)
		{
			if (activation_requires_tape(node.activation))
			{
				return true;
			}
			for (const int consumer_index : node.consumer_node_indices)
			{
				if (consumer_index < 0 or consumer_index >= static_cast<int>(plan.nodes.size()))
				{
					continue;
				}
				if (plan.nodes[static_cast<size_t>(consumer_index)].op_kind == MPSGraphNodeKind::kConvBnAct)
				{
					return true;
				}
			}
			return false;
		}

	}

	MPSGraphSegmentPlan build_training_graph_plan(const Network & net)
	{
		MPSGraphSegmentPlan plan;
		if (net.track)
		{
			plan.reason = "track=1 sequence training is not supported";
			return plan;
		}
		if (net.time_steps != 1)
		{
			plan.reason = "time_steps values other than 1 are not supported";
			return plan;
		}
		if (net.sequential_subdivisions != net.subdivisions)
		{
			plan.reason = "sequential_subdivisions must equal subdivisions";
			return plan;
		}
		if (net.try_fix_nan)
		{
			plan.reason = "try_fix_nan is not supported";
			return plan;
		}
		if (net.adversarial or net.adversarial_lr != 0.0f)
		{
			plan.reason = "adversarial training is not supported";
			return plan;
		}
		if (net.dynamic_minibatch)
		{
			plan.reason = "dynamic_minibatch is not supported by a static MPSGraph training executable";
			return plan;
		}
		if (net.badlabels_rejection_percentage != 0.0f or
			net.num_sigmas_reject_badlabels != 0.0f or net.equidistant_point != 0)
		{
			plan.reason = "bad-label rejection and equidistant-point filtering are not supported";
			return plan;
		}
		if (net.ema_alpha != 0.0f)
		{
			plan.reason = "ema_alpha is not supported";
			return plan;
		}
		if (net.weights_reject_freq != 0)
		{
			plan.reason = "weights_reject_freq is not supported";
			return plan;
		}

		for (int i = 0; i < net.n; ++i)
		{
			const Layer & l = net.layers[i];
			std::string reason;
			if (not layer_supported(net, i, l, reason))
			{
				plan.reason =
					"layer #" + std::to_string(i) +
					" (" + Darknet::to_string(l.type) + "): " + reason;
				return plan;
			}

			if (is_detector_head_layer_type(l.type))
			{
				plan.yolo_layers.push_back(i);
				if (i <= 0 or net.layers[i - 1].type != Darknet::ELayerType::CONVOLUTIONAL)
				{
					plan.reason = "Apple MPSGraph training requires YOLO heads to be fed directly by convolutional layers";
					return plan;
				}
				for (auto & node : plan.nodes)
				{
					if (node.darknet_layer_index == i - 1 and
						node.op_kind == MPSGraphNodeKind::kConvBnAct)
					{
						node.segment_output_to_yolo = true;
						break;
					}
				}
				plan.terminal_output_layers.push_back(i - 1);
				MPSGraphNodePlan boundary = build_node_plan(net, i - 1);
				boundary.segment_output_to_yolo = true;
				boundary.tensors.push_back(make_tensor_plan(
					MPSGraphTensorBindingKind::kYoloBoundaryOutput,
					i - 1,
					i - 1,
					nchw_shape_for_layer_output(net.layers[i - 1]),
					true,
					true));
				boundary.tensors.push_back(make_tensor_plan(
					MPSGraphTensorBindingKind::kYoloBoundaryDelta,
					i - 1,
					i,
					nchw_shape_for_layer_output(net.layers[i - 1]),
					false,
					true));
				plan.boundary_tensors.push_back(boundary.tensors[boundary.tensors.size() - 2]);
				plan.boundary_tensors.push_back(boundary.tensors.back());
				continue;
			}

			MPSGraphNodePlan node = build_node_plan(net, i);
			const auto append_input_occurrence = [&](const int source_layer)
			{
				if (source_layer < 0 or source_layer >= i)
				{
					return;
				}
				node.input_node_indices.push_back(source_layer);
			};
			for (int input_idx = 0; input_idx < l.n; ++input_idx)
			{
				if (not l.input_layers)
				{
					break;
				}
				append_input_occurrence(l.input_layers[input_idx]);
			}
			if (node.input_node_indices.empty() and
				i > 0 and
				node.op_kind != MPSGraphNodeKind::kYoloBoundaryOutput)
			{
				node.input_node_indices.push_back(i - 1);
			}
			plan.nodes.push_back(std::move(node));
		}

		if (plan.yolo_layers.empty())
		{
			plan.reason = "MPSGraph path requires at least one YOLO-family head";
			return plan;
		}
		if (plan.yolo_layers.size() < 2 or plan.yolo_layers.size() > 3)
		{
			plan.reason = "Only cfgs with two or three YOLO heads are supported.";
			return plan;
		}
		if (net.adam)
		{
			plan.reason = "adam=1 optimizer is currently not supported, please remove this from the cfg file.";
			return plan;
		}
		for (const int yolo_layer : plan.yolo_layers)
		{
			if (net.layers[yolo_layer].random != 0.0f)
			{
				plan.reason = "random multi-scale training is currently not supported, please change this to random=0 in the cfg file.";
				return plan;
			}
		}
		if (plan.terminal_output_layers.size() != plan.yolo_layers.size())
		{
			plan.reason = "failed to infer terminal outputs before YOLO heads";
			return plan;
		}
		plan.supported = true;
		for (size_t node_index = 0; node_index < plan.nodes.size(); ++node_index)
		{
			auto & node = plan.nodes[node_index];
			const int consumer_node_index = static_cast<int>(node_index);
			for (const int input_layer_index : node.input_node_indices)
			{
				for (auto & source_node : plan.nodes)
				{
					if (source_node.darknet_layer_index == input_layer_index)
					{
						const auto consumer = std::find(
							source_node.consumer_node_indices.begin(),
							source_node.consumer_node_indices.end(),
							consumer_node_index);
						if (consumer == source_node.consumer_node_indices.end())
						{
							source_node.consumer_node_indices.push_back(consumer_node_index);
						}
						break;
					}
				}
			}
		}
		for (auto & node : plan.nodes)
		{
			if (node.segment_output_to_yolo)
			{
				continue;
			}
			const bool save_for_backward = node_output_needed_for_backward(plan, node);
			if (not save_for_backward)
			{
				continue;
			}
			node.tensors.push_back(make_tensor_plan(
				MPSGraphTensorBindingKind::kLayerOutput,
				node.darknet_layer_index,
				node.darknet_layer_index,
				node.output_shape_nchw,
				true,
				false));
		}
		plan.reason = "ok";
		return plan;
	}

}
