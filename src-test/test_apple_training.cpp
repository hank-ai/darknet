#include <gtest/gtest.h>
#include "apple/apple_training_data.hpp"

TEST(AppleTraining, FloatViewBorrowsItsStorage)
{
	std::vector<float> values{1.0f, 2.0f, 3.0f};
	const Darknet::Apple::FloatBufferView view(values);
	EXPECT_EQ(view.data(), values.data());
	EXPECT_EQ(view.size(), values.size());
	values[1] = 4.0f;
	EXPECT_EQ(view.data()[1], 4.0f);
}

#ifdef DARKNET_USE_MPS
#include "darknet_internal.hpp"
#include "apple/apple_training.hpp"

TEST(AppleTraining, DataExtentsCoverEveryHead)
{
	Darknet::Network net{};
	Darknet::Layer heads[2]{};
	net.layers = heads;
	net.n = 2;
	net.batch = 2;
	net.inputs = 3;
	for (auto & head : heads)
	{
		head.type = Darknet::ELayerType::YOLO;
		head.batch = 2;
		head.truth_size = 6;
		head.max_boxes = 2;
		head.truths = 12;
	}
	heads[1].max_boxes = 3;
	heads[1].truths = 18;
	std::vector<float> input(6);
	std::vector<float> truth(36);
	std::string reason;
	const auto valid = [&](const Darknet::Apple::FloatBufferView & x, const Darknet::Apple::FloatBufferView & y)
	{
		return Darknet::Apple::validate_training_data_views(net, x, y, reason);
	};
	EXPECT_TRUE(valid({input.data(), input.size()}, {truth.data(), truth.size()}));
	EXPECT_EQ(reason, "ok");
	EXPECT_FALSE(valid({input.data(), 5}, {truth.data(), truth.size()}));
	EXPECT_FALSE(valid({nullptr, input.size()}, {truth.data(), truth.size()}));
	EXPECT_FALSE(valid({input.data(), input.size()}, {truth.data(), 24}));
	EXPECT_FALSE(valid({input.data(), input.size()}, {nullptr, truth.size()}));
	heads[1].truths = 12;
	EXPECT_FALSE(valid({input.data(), input.size()}, {truth.data(), truth.size()}));
	EXPECT_EQ(reason, "YOLO truth geometry exceeds its declared buffer size");
}

TEST(AppleTraining, UnconfiguredNetworkInfoUsesAReference)
{
	const Darknet::Network net{};
	const auto info = Darknet::Apple::build_training_graph_info(net);
	EXPECT_FALSE(info.graph_supported);
	EXPECT_FALSE(info.reason.empty());
}
#endif
