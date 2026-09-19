#include <gtest/gtest.h>

#include <cfloat>
#include <cmath>
#include <vector>

#include "darknet_internal.hpp"
#include "yolo_layer.hpp"

#if defined(DARKNET_GPU_CUDA)

namespace
{
	bool cuda_tests_enabled()
	{
		return Darknet::CfgAndState::get().gpu_index >= 0;
	}

	void configure_classic_yolo(Darknet::Layer & layer)
	{
		layer.scale_x_y = 1.05f;
		layer.ignore_thresh = 0.7f;
		layer.truth_thresh = 1.0f;
		layer.iou_thresh = 1.0f;
		layer.iou_loss = CIOU;
		layer.iou_thresh_kind = IOU;
		layer.iou_normalizer = 1.0f;
		layer.obj_normalizer = 1.0f;
		layer.cls_normalizer = 1.0f;
		layer.delta_normalizer = 1.0f;
		layer.max_delta = FLT_MAX;
		for (int anchor = 0; anchor < layer.total; ++anchor)
		{
			layer.biases[2 * anchor] = 12.0f + 8.0f * anchor;
			layer.biases[2 * anchor + 1] = 10.0f + 6.0f * anchor;
		}
	}

	std::vector<float> activate_classic_yolo(
		const std::vector<float> & input,
		const Darknet::Layer & layer)
	{
		std::vector<float> output = input;
		const int spatial = layer.w * layer.h;
		const int entries = layer.classes + 5;
		for (size_t index = 0; index < output.size(); ++index)
		{
			const int entry = (index / spatial) % entries;
			if (entry < 2 || entry >= 4)
			{
				output[index] = 1.0f / (1.0f + std::exp(-output[index]));
			}
			if (entry < 2)
			{
				output[index] = output[index] * layer.scale_x_y
					- 0.5f * (layer.scale_x_y - 1.0f);
			}
		}
		return output;
	}
}

TEST(YoloLayerCuda, FusedActivationMatchesClassicYoloLayout)
{
	if (!cuda_tests_enabled()) GTEST_SKIP() << "CUDA tests were disabled with DARKNET_TEST_GPU=0";
	constexpr int batch = 1;
	constexpr int anchors = 1;
	constexpr int width = 2;
	constexpr int height = 1;
	constexpr int classes = 2;
	constexpr int spatial = width * height;
	constexpr int entries = 5 + classes;
	std::vector<float> input(entries * spatial, 0.0f);
	input[0] = -1.0f;
	input[1] = 1.0f;
	input[2] = 0.5f;
	input[3] = -0.5f;
	input[4] = 2.0f;
	input[5] = -2.0f;
	input[6] = 3.0f;
	input[7] = -3.0f;
	input[8] = 0.25f;
	input[9] = -0.25f;
	input[10] = 0.75f;
	input[11] = -0.75f;
	input[12] = 1.25f;
	input[13] = -1.25f;

	float * input_gpu = cuda_make_array(input.data(), input.size());
	float * output_gpu = cuda_make_array(nullptr, input.size());
	activate_yolo_output_ongpu(
		input_gpu, output_gpu, batch, anchors, width, height, classes, 1.05f, false);
	std::vector<float> output(input.size());
	cuda_pull_array(output_gpu, output.data(), output.size());

	auto logistic = [](const float x) { return 1.0f / (1.0f + std::exp(-x)); };
	for (int i = 0; i < 2 * spatial; ++i)
	{
		EXPECT_NEAR(output[i], logistic(input[i]) * 1.05f - 0.025f, 1.0e-6f);
	}
	for (int i = 2 * spatial; i < 4 * spatial; ++i)
	{
		EXPECT_FLOAT_EQ(output[i], input[i]);
	}
	for (int i = 4 * spatial; i < entries * spatial; ++i)
	{
		EXPECT_NEAR(output[i], logistic(input[i]), 1.0e-6f);
	}

	cuda_free(output_gpu);
	cuda_free(input_gpu);
}

TEST(YoloLayerCuda, ClassicTrainingMatchesCpuReferenceForPositiveAndBackgroundDeltas)
{
	if (!cuda_tests_enabled()) GTEST_SKIP() << "CUDA tests were disabled with DARKNET_TEST_GPU=0";
	constexpr int batch = 1;
	constexpr int width = 3;
	constexpr int height = 2;
	constexpr int anchors = 3;
	constexpr int total_anchors = 6;
	constexpr int classes = 2;
	constexpr int max_boxes = 8;

	Darknet::Network net = make_network(1);
	net.batch = batch;
	net.w = 224;
	net.h = 160;
	net.max_batches = 100;
	net.layers[0] = make_yolo_layer(
		batch, width, height, anchors, total_anchors, nullptr, classes, max_boxes);
	Darknet::Layer & gpu_layer = net.layers[0];
	configure_classic_yolo(gpu_layer);
	gpu_layer.mask[0] = 3;
	gpu_layer.mask[1] = 4;
	gpu_layer.mask[2] = 5;

	Darknet::Layer cpu_layer = make_yolo_layer(
		batch, width, height, anchors, total_anchors, nullptr, classes, max_boxes);
	configure_classic_yolo(cpu_layer);
	for (int index = 0; index < anchors; ++index) cpu_layer.mask[index] = gpu_layer.mask[index];
	for (int index = 0; index < total_anchors * 2; ++index) cpu_layer.biases[index] = gpu_layer.biases[index];

	std::vector<float> raw(static_cast<size_t>(gpu_layer.outputs) * batch);
	for (size_t index = 0; index < raw.size(); ++index)
	{
		raw[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.075f;
	}
	std::vector<float> truth(static_cast<size_t>(gpu_layer.truths) * batch, 0.0f);
	truth[0] = 0.52f;
	truth[1] = 0.48f;
	truth[2] = 0.24f;
	truth[3] = 0.20f;
	truth[4] = 1.0f;
	truth[5] = 7.0f;
	// A second truth deliberately collides with the same anchor/cell.  The CPU
	// implementation accumulates box deltas and applies class updates in truth
	// order; the one-thread-per-image positive kernel must preserve that rule.
	truth[6] = 0.53f;
	truth[7] = 0.49f;
	truth[8] = 0.24f;
	truth[9] = 0.20f;
	truth[10] = 0.0f;
	truth[11] = 8.0f;

	Darknet::NetworkState cpu_state = {};
	cpu_state.net = net;
	cpu_state.index = 0;
	// GPU builds skip the activation in forward_yolo_layer(), so feed it pre-activated input.
	std::vector<float> activated = activate_classic_yolo(raw, cpu_layer);
	cpu_state.input = activated.data();
	cpu_state.truth = truth.data();
	cpu_state.train = 1;
	forward_yolo_layer(cpu_layer, cpu_state);
	const std::vector<float> expected_delta(
		cpu_layer.delta, cpu_layer.delta + static_cast<size_t>(cpu_layer.batch) * cpu_layer.outputs);
	const float expected_cost = *cpu_layer.cost;

	float * raw_gpu = cuda_make_array(raw.data(), raw.size());
	float * truth_gpu = cuda_make_array(truth.data(), truth.size());
	activate_yolo_output_ongpu(raw_gpu, gpu_layer.output_gpu,
		batch, anchors, width, height, classes, gpu_layer.scale_x_y, false);
	Darknet::NetworkState gpu_state = cpu_state;
	gpu_state.input = raw_gpu;
	gpu_state.truth = truth_gpu;
	Darknet::forward_yolo_training_cuda(gpu_layer, gpu_state);
	Darknet::finalize_yolo_training_cuda(net);

	std::vector<float> actual_delta(expected_delta.size());
	cuda_pull_array(gpu_layer.delta_gpu, actual_delta.data(), actual_delta.size());
	for (size_t index = 0; index < actual_delta.size(); ++index)
	{
		EXPECT_NEAR(actual_delta[index], expected_delta[index], 2.0e-5f) << "delta index " << index;
	}
	EXPECT_NEAR(*gpu_layer.cost, expected_cost, 1.0e-4f);
	EXPECT_NE(gpu_layer.yolo_training_gpu_context, nullptr);

	activate_yolo_output_ongpu(raw_gpu, gpu_layer.output_gpu,
		batch, anchors, width, height, classes, gpu_layer.scale_x_y, false);
	Darknet::forward_yolo_training_cuda(gpu_layer, gpu_state);
	Darknet::finalize_yolo_training_cuda(net);
	std::vector<float> repeated_delta(actual_delta.size());
	cuda_pull_array(gpu_layer.delta_gpu, repeated_delta.data(), repeated_delta.size());
	EXPECT_EQ(repeated_delta, actual_delta) << "CUDA YOLO delta must be deterministic";

	cuda_free(truth_gpu);
	cuda_free(raw_gpu);
	free_layer(cpu_layer);
	free_network(net);
}

TEST(YoloLayerCuda, CfgOptionYoloGpuSelectsCpuOrCuda)
{
	if (!cuda_tests_enabled()) GTEST_SKIP() << "CUDA tests were disabled with DARKNET_TEST_GPU=0";
	Darknet::Network net = make_network(1);
	net.batch = 1;
	net.w = 224;
	net.h = 160;
	net.max_batches = 100;
	net.layers[0] = make_yolo_layer(1, 2, 2, 3, 6, nullptr, 2, 8);
	Darknet::Layer & layer = net.layers[0];
	configure_classic_yolo(layer);
	layer.mask[0] = 3;
	layer.mask[1] = 4;
	layer.mask[2] = 5;
	std::vector<float> raw(layer.outputs, 0.0f);
	std::vector<float> truth(layer.truths, 0.0f);
	float * raw_gpu = cuda_make_array(raw.data(), raw.size());
	float * truth_gpu = cuda_make_array(truth.data(), truth.size());
	Darknet::NetworkState state = {};
	state.net = net;
	state.input = raw_gpu;
	state.truth = truth_gpu;
	state.train = 1;

	// yolo_gpu=0 in the .cfg forces the CPU loss even though this configuration
	// would be accepted by the CUDA path.
	layer.yolo_gpu = 0;
	forward_yolo_layer_gpu(layer, state);
	EXPECT_EQ(layer.yolo_training_gpu_context, nullptr);
	EXPECT_FLOAT_EQ(*layer.cost, layer.n * layer.w * layer.h * 0.25f);

	// yolo_gpu=1 requires the CUDA path for a supported configuration.
	layer.yolo_gpu = 1;
	forward_yolo_layer_gpu(layer, state);
	EXPECT_NE(layer.yolo_training_gpu_context, nullptr);
	Darknet::finalize_yolo_training_cuda(net);

	// the CUDA scratch is shape dependent: resizing the layer releases it
	resize_yolo_layer(&layer, 3, 3);
	EXPECT_EQ(layer.yolo_training_gpu_context, nullptr);

	cuda_free(truth_gpu);
	cuda_free(raw_gpu);
	free_network(net);
}

#endif // DARKNET_GPU_CUDA
