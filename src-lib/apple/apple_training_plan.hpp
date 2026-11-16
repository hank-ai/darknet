#pragma once

/** @file
 * Apple graph-training planner types and support checks.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace Darknet
{
	struct Network;
}

namespace Darknet::Apple
{
	/** Graph operator kinds emitted by the Apple training planner. */
	enum class MPSGraphNodeKind
	{
		kConvBnAct,			///< Convolution, optional batch normalization, and activation.
		kRouteConcat,		///< Route layer concatenation.
		kRouteGroupSlice,	///< Route group slice.
		kMaxPool,			///< Maxpool operation.
		kUpsample,			///< Nearest-neighbour upsample operation.
		kYoloBoundaryOutput,	///< Boundary tensor passed to a YOLO loss layer.
	};

	/** Activation kinds the planner can materialize inside a graph segment. */
	enum class MPSGraphActivationKind
	{
		kLinear,	///< Linear activation.
		kLeaky,		///< Leaky ReLU activation.
		kLogistic,	///< Logistic activation.
	};

	/** Stable tensor binding roles used by the planner/runtime contract. */
	enum class MPSGraphTensorBindingKind
	{
		kNetworkInput,			///< Network input tensor.
		kNetworkInputDelta,		///< Gradient tensor for the network input.
		kLayerOutput,			///< Layer output tensor.
		kLayerDelta,			///< Layer delta tensor.
		kWeights,				///< Layer weight tensor.
		kWeightGrad,			///< Layer weight-gradient tensor.
		kBias,					///< Layer bias tensor.
		kBiasGrad,				///< Layer bias-gradient tensor.
		kBatchNormScale,		///< Batch-normalization scale tensor.
		kBatchNormScaleGrad,	///< Batch-normalization scale-gradient tensor.
		kConvSourceTape,		///< Saved convolution input tensor.
		kBatchNormInputTape,	///< Saved batch-normalization input tensor.
		kBatchMeanTape,			///< Saved batch mean tensor.
		kBatchVarianceTape,		///< Saved batch variance tensor.
		kBatchNormOutputTape,	///< Saved batch-normalization output tensor.
		kActivationTape,		///< Saved activation output tensor.
		kMaxPoolIndicesTape,	///< Saved maxpool index tensor.
		kYoloBoundaryOutput,	///< Output tensor handed to YOLO loss.
		kYoloBoundaryDelta,		///< Delta tensor received from YOLO loss.
	};

	/** Scalar data type used by planned MPSGraph tensors. */
	enum class MPSGraphTensorDataType
	{
		kFloat32,	///< 32-bit floating-point values.
		kInt32,		///< 32-bit signed integer values.
	};

	/** Planned tensor storage for one graph segment edge or parameter.
	 *
	 * Stable fields:
	 * - binding_kind
	 * - owner_layer_index
	 * - source_layer_index
	 * - bytes
	 * - shape_nchw
	 *
	 * Internal execution-detail fields describe forward materialization and
	 * the compiled graph-owned storage slot.
	 */
	struct MPSGraphTensorPlan
	{
		MPSGraphTensorBindingKind binding_kind = MPSGraphTensorBindingKind::kLayerOutput;	///< Runtime binding role.
		int owner_layer_index = -1;							///< Layer that owns this tensor, or @p -1.
		int source_layer_index = -1;							///< Source layer for boundary tensors, or @p -1.
		size_t bytes = 0;									///< Tensor allocation size in bytes.
		MPSGraphTensorDataType data_type = MPSGraphTensorDataType::kFloat32;	///< Scalar tensor type.
		bool forward_result = false;							///< Tensor is returned by forward graph execution.
		bool materialize_to_darknet_buffer = false;			///< Copy this tensor back to the matching Darknet buffer.
		size_t storage_index = std::numeric_limits<size_t>::max();	///< Compiled graph-owned storage slot, if any.
		std::vector<int64_t> shape_nchw;						///< Tensor dimensions in NCHW order.
	};

	/** Planned graph node metadata for one supported Darknet layer.
	 *
	 * Stable fields describe the node kind, owning layer, input/output shape,
	 * and whether the node participates in segment boundaries. The remaining
	 * flags are internal execution details used by the graph runtime.
	 */
	struct MPSGraphNodePlan
	{
		MPSGraphNodeKind op_kind = MPSGraphNodeKind::kConvBnAct;	///< Planned graph operation.
		int darknet_layer_index = -1;					///< Index of the matching Darknet layer.
		std::vector<int> input_node_indices;				///< Ordered Darknet layer inputs; route duplicates are preserved.
		std::vector<int> consumer_node_indices;			///< Segment node indexes that consume this node.
		std::vector<int64_t> output_shape_nchw;			///< Output tensor dimensions in NCHW order.
		bool segment_input = false;						///< Node consumes a segment input boundary.
		bool segment_output_to_yolo = false;				///< Node output feeds a YOLO boundary.
		int kernel_h = 0;								///< Kernel height.
		int kernel_w = 0;								///< Kernel width.
		int stride_y = 0;								///< Vertical stride.
		int stride_x = 0;								///< Horizontal stride.
		int pad_top = 0;								///< Top padding.
		int pad_bottom = 0;								///< Bottom padding.
		int pad_left = 0;								///< Left padding.
		int pad_right = 0;								///< Right padding.
		int groups = 1;									///< Number of convolution or route groups.
		int slice_channel_offset = 0;					///< First channel for route group slicing.
		int slice_channel_count = 0;						///< Number of channels for route group slicing.
		int route_concat_axis = 1;						///< Tensor axis used by route concatenation.
		int upsample_scale = 0;							///< Nearest-neighbour upsample scale.
		bool batch_normalize = false;					///< Apply batch normalization inside this node.
		MPSGraphActivationKind activation = MPSGraphActivationKind::kLinear;	///< Activation emitted for this node.
		std::vector<MPSGraphTensorPlan> tensors;			///< Planned tensor bindings for this node.
	};

	/** Planner output for one Apple graph-training candidate segment.
	 *
	 * A supported segment means the current network can run its non-YOLO body
	 * on the Apple graph training path while detector heads use the GPU loss bridge.
	 */
	struct MPSGraphSegmentPlan
	{
		bool supported = false;								///< The network can use this graph-training segment.
		std::vector<int> terminal_output_layers;				///< Segment output layers that must be materialized.
		std::vector<int> yolo_layers;						///< YOLO layers attached to the segment boundary.
		std::vector<MPSGraphNodePlan> nodes;					///< Planned graph nodes.
		std::vector<MPSGraphTensorPlan> boundary_tensors;	///< Planned segment boundary tensors.
		std::string reason;									///< Short planner status or fallback reason.
	};

	/// Build the graph-training plan for a network.
	MPSGraphSegmentPlan build_training_graph_plan(const Network & net);

}
