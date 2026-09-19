#include "darknet_internal.hpp"

#ifdef DARKNET_GPU_CUDA

namespace
{
	constexpr int kMaxAnchors = 32;	///< needed for the fixed-size mask/biases arrays in YoloKernelArgs (kernel arguments must have a fixed size)

	/// Per-image loss sums written by yolo_loss_kernel.  layer.yolo_training_gpu_context points to batch of these.
	/// Why: the host only copies these few bytes back instead of the whole output/delta tensors.
	struct YoloImageMetrics
	{
		float total_iou, total_iou_loss;						///< avg IoU print and the iou part of layer.cost
		float delta_squared, classification_delta_squared;		///< total_loss / class_loss print and layer.cost
		int count, total_bbox, rewritten_bbox;					///< matched truths; net.total_bbox / net.rewritten_bbox counters
	};

	/// Why: Darknet::Layer is too large (and holds host pointers) to pass to a kernel, so copy only what the loss reads.
	struct YoloKernelArgs
	{
		int width, height, anchors, total_anchors, classes, max_boxes, truth_size, truths, outputs, net_width, net_height;	///< layer and network shape
		float ignore_thresh, iou_normalizer, obj_normalizer, max_delta;	///< [yolo] loss options
		int mask[kMaxAnchors];				///< l.mask by value: avoids a GPU copy of the host array
		float biases[2 * kMaxAnchors];		///< l.biases (anchor w,h pairs) by value, same reason
	};

	/// Why: yolo_entry_index() in yolo_layer.cpp is host-only; same anchor-major layout.
	__device__ __forceinline__ int entry_index(const YoloKernelArgs & args, const int batch, const int anchor, const int location, const int entry)
	{
		const int spatial = args.width * args.height;
		return batch * args.outputs + anchor * spatial * (args.classes + 5) + entry * spatial + location;
	}

	/// Why: reads one truth box (x,y,w,h) from global or shared memory; used by every truth loop in the kernel.
	__device__ __forceinline__ Darknet::Box load_truth(const float * truth, const int base)
	{
		return { truth[base], truth[base + 1], truth[base + 2], truth[base + 3] };
	}

	/// Why: get_yolo_box() in yolo_layer.cpp is host-only; this is the same formula (classic coordinates).
	__device__ __forceinline__ Darknet::Box decode_box(const float * output, const int bias_anchor, const int box_index,
		const int column, const int row, const YoloKernelArgs & args)
	{
		const int spatial = args.width * args.height;
		return {
			(column + output[box_index]) / args.width,
			(row + output[box_index + spatial]) / args.height,
			expf(output[box_index + 2 * spatial]) * args.biases[2 * bias_anchor] / args.net_width,
			expf(output[box_index + 3 * spatial]) * args.biases[2 * bias_anchor + 1] / args.net_height,
		};
	}

	/// Why: box.cpp to_tblr() is host-only; needed by ciou_derivative().
	__device__ __forceinline__ boxabs to_tblr(const Darknet::Box box)
	{
		return { box.x - box.w * 0.5f, box.x + box.w * 0.5f, box.y - box.h * 0.5f, box.y + box.h * 0.5f };
	}

	/// Why: box_iou() in box.cpp is host-only (CUDA separable compilation is off, so no cross-file device calls).
	__device__ __forceinline__ float box_iou_device(const Darknet::Box a, const Darknet::Box b)
	{
		const float width = fminf(a.x + a.w * 0.5f, b.x + b.w * 0.5f) - fmaxf(a.x - a.w * 0.5f, b.x - b.w * 0.5f);
		const float height = fminf(a.y + a.h * 0.5f, b.y + b.h * 0.5f) - fmaxf(a.y - a.h * 0.5f, b.y - b.h * 0.5f);
		if (width <= 0.0f || height <= 0.0f)
		{
			return 0.0f;
		}
		const float overlap = width * height;
		return overlap / (a.w * a.h + b.w * b.h - overlap);
	}

	/// Why: dx_box_iou(..., CIOU) in box.cpp is host-only.  Same operation order, so the gradients
	/// match the CPU loss within the unit-test tolerance.
	__device__ __forceinline__ Darknet::Box ciou_derivative(const Darknet::Box pred, const Darknet::Box truth)
	{
		const boxabs pred_tblr = to_tblr(pred);
		const float pred_t = fminf(pred_tblr.top, pred_tblr.bot);
		const float pred_b = fmaxf(pred_tblr.top, pred_tblr.bot);
		const float pred_l = fminf(pred_tblr.left, pred_tblr.right);
		const float pred_r = fmaxf(pred_tblr.left, pred_tblr.right);
		const boxabs truth_tblr = to_tblr(truth);

		const float pred_area = (pred_b - pred_t) * (pred_r - pred_l);
		const float truth_area = (truth_tblr.bot - truth_tblr.top) * (truth_tblr.right - truth_tblr.left);
		const float intersection_h = fminf(pred_b, truth_tblr.bot) - fmaxf(pred_t, truth_tblr.top);
		const float intersection_w = fminf(pred_r, truth_tblr.right) - fmaxf(pred_l, truth_tblr.left);
		const float intersection_area = intersection_w * intersection_h;
		const float union_area = pred_area + truth_area - intersection_area;
		const float center_distance = (pred.x - truth.x) * (pred.x - truth.x) + (pred.y - truth.y) * (pred.y - truth.y);

		const float d_intersection_t = pred_t > truth_tblr.top ? -intersection_w : 0.0f;
		const float d_intersection_b = pred_b < truth_tblr.bot ? intersection_w : 0.0f;
		const float d_intersection_l = pred_l > truth_tblr.left ? -intersection_h : 0.0f;
		const float d_intersection_r = pred_r < truth_tblr.right ? intersection_h : 0.0f;
		const float d_union_t = -(pred_r - pred_l) - d_intersection_t;
		const float d_union_b = (pred_r - pred_l) - d_intersection_b;
		const float d_union_l = -(pred_b - pred_t) - d_intersection_l;
		const float d_union_r = (pred_b - pred_t) - d_intersection_r;

		float p_t = 0.0f, p_b = 0.0f, p_l = 0.0f, p_r = 0.0f;
		if (union_area > 0.0f)
		{
			const float union_squared = union_area * union_area;
			p_t = (union_area * d_intersection_t - intersection_area * d_union_t) / union_squared;
			p_b = (union_area * d_intersection_b - intersection_area * d_union_b) / union_squared;
			p_l = (union_area * d_intersection_l - intersection_area * d_union_l) / union_squared;
			p_r = (union_area * d_intersection_r - intersection_area * d_union_r) / union_squared;
		}
		// Mirrors the original sequential assignments.
		p_t = pred_tblr.top < pred_tblr.bot ? p_t : p_b;
		p_b = pred_tblr.top < pred_tblr.bot ? p_b : p_t;
		p_l = pred_tblr.left < pred_tblr.right ? p_l : p_r;
		p_r = pred_tblr.left < pred_tblr.right ? p_r : p_l;

		const float enclosing_w = fmaxf(pred.x + pred.w * 0.5f, truth.x + truth.w * 0.5f) - fminf(pred.x - pred.w * 0.5f, truth.x - truth.w * 0.5f);
		const float enclosing_h = fmaxf(pred.y + pred.h * 0.5f, truth.y + truth.h * 0.5f) - fminf(pred.y - pred.h * 0.5f, truth.y - truth.h * 0.5f);
		const float diagonal_squared = enclosing_w * enclosing_w + enclosing_h * enclosing_h;
		const float d_enclosing_w_x = (pred_r > truth_tblr.right ? 1.0f : 0.0f) - (pred_l < truth_tblr.left ? 1.0f : 0.0f);
		const float d_enclosing_w_w = (pred_r > truth_tblr.right ? 0.5f : 0.0f) - (pred_l < truth_tblr.left ? -0.5f : 0.0f);
		const float d_enclosing_h_y = (pred_b > truth_tblr.bot ? 1.0f : 0.0f) - (pred_t < truth_tblr.top ? 1.0f : 0.0f);
		const float d_enclosing_h_h = (pred_b > truth_tblr.bot ? 0.5f : 0.0f) - (pred_t < truth_tblr.top ? -0.5f : 0.0f);

		const float aspect_angle = atanf(truth.w / truth.h) - atanf(pred.w / pred.h);
		const float aspect_loss = 4.0f / (M_PI * M_PI) * aspect_angle * aspect_angle;
		const float alpha = aspect_loss / (1.0f - intersection_area / union_area + aspect_loss + 0.000001f);

		// Without overlap only the distance/aspect terms remain (0 + term is exact).
		const bool no_overlap = intersection_w <= 0.0f || intersection_h <= 0.0f;
		Darknet::Box d = no_overlap ? Darknet::Box{ 0.0f, 0.0f, 0.0f, 0.0f } : Darknet::Box{ p_l + p_r, p_t + p_b, p_r - p_l, p_b - p_t };
		if (diagonal_squared > 0.0f || no_overlap)
		{
			const float diagonal_fourth = diagonal_squared * diagonal_squared;
			d.x += (2.0f * (truth.x - pred.x) * diagonal_squared - 2.0f * enclosing_w * d_enclosing_w_x * center_distance) / diagonal_fourth;
			d.y += (2.0f * (truth.y - pred.y) * diagonal_squared - 2.0f * enclosing_h * d_enclosing_h_y * center_distance) / diagonal_fourth;
			d.w += (2.0f * enclosing_w * d_enclosing_w_w * center_distance) / diagonal_fourth + alpha * (8.0f / (M_PI * M_PI) * aspect_angle * pred.h);
			d.h += (2.0f * enclosing_h * d_enclosing_h_h * center_distance) / diagonal_fourth + alpha * (-8.0f / (M_PI * M_PI) * aspect_angle * pred.w);
		}
		return d;
	}

	/// Why: fix_nan_inf() + clip_value() in yolo_layer.cpp are host-only (clipping to FLT_MAX is a no-op for finite values).
	__device__ __forceinline__ float sanitize_and_clip(const float value, const float max_delta)
	{
		return isfinite(value) ? fminf(max_delta, fmaxf(-max_delta, value)) : 0.0f;
	}

	/// Why: device version of delta_yolo_box() for CIoU: box delta of the cell/anchor matched to a truth.
	__device__ __forceinline__ void assign_ciou_box(const Darknet::Box truth, const float * output, float * delta, const int bias_anchor,
		const int box_index, const int column, const int row, const YoloKernelArgs & args, YoloImageMetrics & metrics)
	{
		const int spatial = args.width * args.height;
		if (delta[box_index] != 0.0f || delta[box_index + spatial] != 0.0f || delta[box_index + 2 * spatial] != 0.0f || delta[box_index + 3 * spatial] != 0.0f)
		{
			++metrics.rewritten_bbox;
		}

		Darknet::Box pred = decode_box(output, bias_anchor, box_index, column, row, args);
		const float iou = box_iou_device(pred, truth);
		metrics.total_iou += iou;
		metrics.total_iou_loss += 1.0f - iou;
		if (pred.w == 0.0f) pred.w = 1.0f;	// avoid nan in the derivative, as the CPU path does
		if (pred.h == 0.0f) pred.h = 1.0f;
		const Darknet::Box d = ciou_derivative(pred, truth);
		delta[box_index] += sanitize_and_clip(d.x * args.iou_normalizer, args.max_delta);
		delta[box_index + spatial] += sanitize_and_clip(d.y * args.iou_normalizer, args.max_delta);
		delta[box_index + 2 * spatial] += sanitize_and_clip(d.w * expf(output[box_index + 2 * spatial]) * args.iou_normalizer, args.max_delta);
		delta[box_index + 3 * spatial] += sanitize_and_clip(d.h * expf(output[box_index + 3 * spatial]) * args.iou_normalizer, args.max_delta);
		++metrics.total_bbox;
	}

	/// Why: device version of delta_yolo_class() (no focal loss / label smoothing, which yolo_gpu rejects).
	__device__ __forceinline__ void assign_class(const float * output, float * delta, const int class_index, const int class_id, const YoloKernelArgs & args)
	{
		const int spatial = args.width * args.height;
		const int target_index = class_index + class_id * spatial;
		if (delta[target_index] != 0.0f)
		{
			const float value = 1.0f - output[target_index];
			if (isfinite(value)) delta[target_index] = value;
			return;
		}
		for (int cls = 0; cls < args.classes; ++cls)
		{
			const int index = class_index + cls * spatial;
			const float value = (cls == class_id ? 1.0f : 0.0f) - output[index];
			if (isfinite(value)) delta[index] = value;
		}
	}

	/// Why: sums the squared deltas of one image across the block (warp shuffles); darknet has no reusable block reduction.
	__device__ __forceinline__ void block_sum_pair(float & first, float & second)
	{
		__shared__ float first_warp_sums[32];
		__shared__ float second_warp_sums[32];
		for (int offset = 16; offset > 0; offset >>= 1)
		{
			first += __shfl_down_sync(0xffffffffu, first, offset);
			second += __shfl_down_sync(0xffffffffu, second, offset);
		}
		const int lane = threadIdx.x & 31;
		const int warp = threadIdx.x >> 5;
		if (lane == 0)
		{
			first_warp_sums[warp] = first;
			second_warp_sums[warp] = second;
		}
		__syncthreads();
		first = threadIdx.x < (blockDim.x + 31) / 32 ? first_warp_sums[lane] : 0.0f;
		second = threadIdx.x < (blockDim.x + 31) / 32 ? second_warp_sums[lane] : 0.0f;
		if (warp == 0)
		{
			for (int offset = 16; offset > 0; offset >>= 1)
			{
				first += __shfl_down_sync(0xffffffffu, first, offset);
				second += __shfl_down_sync(0xffffffffu, second, offset);
			}
		}
	}

	/// Why: replaces the per-iteration GPU->CPU->GPU copy of forward_yolo_layer_gpu().  One block per image:
	/// background (objectness) deltas, then the matched truths, then the squared-delta sums.
	__global__ void yolo_loss_kernel(float * output, const float * truth, float * delta, YoloImageMetrics * image_metrics, const YoloKernelArgs args)
	{
		extern __shared__ float shared_truth[];
		__shared__ int truth_count;
		const int batch = blockIdx.x;
		const int spatial = args.width * args.height;
		for (int index = threadIdx.x; index < args.max_boxes * 5; index += blockDim.x)
		{
			shared_truth[index] = truth[batch * args.truths + (index / 5) * args.truth_size + index % 5];
		}
		__syncthreads();
		if (threadIdx.x == 0)
		{
			// The label loader already rejects bad class ids and coordinates; the class check only guards memory.
			int t = 0;
			for (; t < args.max_boxes && shared_truth[t * 5] != 0.0f; ++t)
			{
				const int class_id = static_cast<int>(shared_truth[t * 5 + 4]);
				if (class_id < 0 || class_id >= args.classes) break;
			}
			truth_count = t;
		}
		__syncthreads();

		// Background: every prediction gets the "no object" objectness delta unless it already overlaps a truth.
		for (int prediction = threadIdx.x; prediction < args.anchors * spatial; prediction += blockDim.x)
		{
			const int anchor = prediction / spatial;
			const int location = prediction % spatial;
			const int box_index = entry_index(args, batch, anchor, location, 0);
			const int obj_index = entry_index(args, batch, anchor, location, 4);
			const int class_index = entry_index(args, batch, anchor, location, 5);
			const Darknet::Box pred = decode_box(output, args.mask[anchor], box_index, location % args.width, location / args.width, args);
			bool class_match = false;
			for (int cls = 0; cls < args.classes; ++cls)
			{
				class_match = class_match || output[class_index + cls * spatial] > 0.25f;
			}
			float best_match_iou = 0.0f;
			for (int t = 0; class_match && t < truth_count; ++t)
			{
				best_match_iou = fmaxf(best_match_iou, box_iou_device(pred, load_truth(shared_truth, t * 5)));
			}
			// The CPU path zeroes a non-finite objectness (in the output too) while scanning truths.
			if (truth_count > 0 && !isfinite(output[obj_index]))
			{
				output[obj_index] = 0.0f;
			}
			delta[obj_index] = best_match_iou > args.ignore_thresh ? 0.0f : -args.obj_normalizer * output[obj_index];
		}
		__syncthreads();

		// Positives: sequential per image (like the CPU loop) because later truths may rewrite earlier cells.
		YoloImageMetrics metrics = {};
		for (int t = 0; threadIdx.x == 0 && t < truth_count; ++t)
		{
			const Darknet::Box target = load_truth(shared_truth, t * 5);
			const int class_id = static_cast<int>(shared_truth[t * 5 + 4]);
			float best_iou = 0.0f;
			int best_anchor = 0;
			for (int anchor = 0; anchor < args.total_anchors; ++anchor)
			{
				const Darknet::Box candidate = { 0.0f, 0.0f, args.biases[2 * anchor] / args.net_width, args.biases[2 * anchor + 1] / args.net_height };
				const float iou = box_iou_device(candidate, { 0.0f, 0.0f, target.w, target.h });
				if (iou > best_iou)
				{
					best_iou = iou;
					best_anchor = anchor;
				}
			}
			int masked_anchor = 0;
			while (masked_anchor < args.anchors && args.mask[masked_anchor] != best_anchor) ++masked_anchor;
			if (masked_anchor == args.anchors) continue;

			const int column = min(max(static_cast<int>(target.x * args.width), 0), args.width - 1);
			const int row = min(max(static_cast<int>(target.y * args.height), 0), args.height - 1);
			const int location = row * args.width + column;
			assign_ciou_box(target, output, delta, best_anchor, entry_index(args, batch, masked_anchor, location, 0), column, row, args, metrics);
			const int obj_index = entry_index(args, batch, masked_anchor, location, 4);
			delta[obj_index] = args.obj_normalizer * (1.0f - output[obj_index]);
			assign_class(output, delta, entry_index(args, batch, masked_anchor, location, 5), class_id, args);
			++metrics.count;
		}
		__syncthreads();

		float total = 0.0f;
		float classification = 0.0f;
		for (int index = threadIdx.x; index < args.outputs; index += blockDim.x)
		{
			const float squared = delta[batch * args.outputs + index] * delta[batch * args.outputs + index];
			total += squared;
			if ((index / spatial) % (args.classes + 5) >= 4) classification += squared;
		}
		block_sum_pair(total, classification);
		if (threadIdx.x == 0)
		{
			metrics.delta_squared = total;
			metrics.classification_delta_squared = classification;
			image_metrics[batch] = metrics;
		}
	}
}

namespace Darknet
{
	/// Why: entry point from forward_yolo_layer_gpu() when [yolo] yolo_gpu=1; checks the supported subset and launches the loss.
	void forward_yolo_training_cuda(Darknet::Layer & layer, Darknet::NetworkState state)
	{
		if (layer.iou_loss != CIOU || layer.new_coords || layer.focal_loss || layer.label_smooth_eps != 0.0f || layer.objectness_smooth
			|| layer.classes_multipliers || layer.map || layer.iou_thresh < 1.0f || layer.truth_thresh < 1.0f || layer.embedding_output
			|| state.net.adversarial || state.net.badlabels_rejection_percentage != 0.0f || state.net.num_sigmas_reject_badlabels != 0.0f
			|| state.net.equidistant_point != 0 || state.net.contrastive || state.net.track
			|| layer.total > kMaxAnchors || layer.max_boxes > 1024)
		{
			darknet_fatal_error(DARKNET_LOC, "yolo_gpu=1 only supports the classic CIoU [yolo] loss (no new_coords, focal_loss, label_smooth_eps, "
				"objectness_smooth, counters_per_class, map, iou_thresh, truth_thresh < 1, contrastive, tracking or bad-label rejection, "
				"at most %d anchors and 1024 boxes)", kMaxAnchors);
		}

		if (!layer.yolo_training_gpu_context)
		{
			CHECK_CUDA(cudaMalloc(&layer.yolo_training_gpu_context, layer.batch * sizeof(YoloImageMetrics)));
		}

		YoloKernelArgs args = {
			layer.w, layer.h, layer.n, layer.total, layer.classes, layer.max_boxes, layer.truth_size, layer.truths, layer.outputs,
			state.net.w, state.net.h, layer.ignore_thresh, layer.iou_normalizer, layer.obj_normalizer, layer.max_delta,
		};
		std::copy(layer.mask, layer.mask + layer.n, args.mask);
		std::copy(layer.biases, layer.biases + 2 * layer.total, args.biases);

		cudaStream_t stream = get_cuda_stream();
		CHECK_CUDA(cudaMemsetAsync(layer.delta_gpu, 0, layer.batch * layer.outputs * sizeof(float), stream));
		yolo_loss_kernel<<<layer.batch, BLOCK, layer.max_boxes * 5 * sizeof(float), stream>>>(layer.output_gpu, state.truth,
			layer.delta_gpu, static_cast<YoloImageMetrics *>(layer.yolo_training_gpu_context), args);
		CHECK_CUDA(cudaPeekAtLastError());
	}

	/// Why: called once per iteration after backward (network_kernels.cu); one sync publishes layer.cost for every yolo_gpu head.
	void finalize_yolo_training_cuda(Darknet::Network & net)
	{
		for (int index = 0; index < net.n; ++index)
		{
			Darknet::Layer & layer = net.layers[index];
			if (!layer.yolo_training_gpu_context) continue;	// only exists once the CUDA loss ran

			// the first sync drains backward; later heads only wait for their tiny copy
			std::vector<YoloImageMetrics> images(layer.batch);
			CHECK_CUDA(cudaMemcpyAsync(images.data(), layer.yolo_training_gpu_context, layer.batch * sizeof(YoloImageMetrics), cudaMemcpyDeviceToHost, get_cuda_stream()));
			CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()));

			YoloImageMetrics metrics = {};
			for (const YoloImageMetrics & image : images)
			{
				metrics.total_iou += image.total_iou;
				metrics.total_iou_loss += image.total_iou_loss;
				metrics.delta_squared += image.delta_squared;
				metrics.classification_delta_squared += image.classification_delta_squared;
				metrics.count += image.count;
				metrics.total_bbox += image.total_bbox;
				metrics.rewritten_bbox += image.rewritten_bbox;
			}
			const int count = metrics.count > 0 ? metrics.count : 1;
			const float classification_loss = metrics.classification_delta_squared * layer.obj_normalizer;
			*layer.cost = layer.iou_normalizer * (metrics.total_iou_loss / count) + classification_loss;
			if (net.total_bbox) *net.total_bbox += metrics.total_bbox;
			if (net.rewritten_bbox) *net.rewritten_bbox += metrics.rewritten_bbox;
			if (Darknet::CfgAndState::get().is_verbose)
			{
				std::printf("v3 ciou loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %.6f), count: %d, "
					"class_loss: %.6f, iou_loss: %.6f, total_loss: %.6f\n",
					layer.iou_normalizer, layer.obj_normalizer, layer.cls_normalizer, index, metrics.total_iou / count, metrics.count,
					classification_loss / layer.batch, (metrics.delta_squared - classification_loss) / layer.batch, metrics.delta_squared / layer.batch);
			}
		}
	}

	/// Why: frees the per-image metrics buffer; called from free_layer_custom() and resize_yolo_layer() (batch-sized).
	void release_yolo_training_cuda(Darknet::Layer & layer)
	{
		cudaFree(layer.yolo_training_gpu_context);
		layer.yolo_training_gpu_context = nullptr;
	}
}

#endif // DARKNET_GPU_CUDA
