#include <gtest/gtest.h>

#ifdef DARKNET_USE_MPS
#include "darknet_internal.hpp"
#include "apple/apple_mps.hpp"
#include "apple/apple_training.hpp"
#include "apple/metal_backend.hpp"

TEST(AppleDiagnostics, DeviceCountReplacesPreviousReason)
{
	std::string reason = "previous call";
	EXPECT_FALSE(Darknet::Apple::mps_training_device_count_supported(2, reason));
	const std::string failure = reason;
	EXPECT_NE(failure.find("one"), std::string::npos);
	EXPECT_TRUE(Darknet::Apple::mps_training_device_count_supported(1, reason));
	EXPECT_EQ(reason, "ok");
	EXPECT_NE(failure, reason);
}

TEST(AppleDiagnostics, InvalidValidationNetworkReportsReason)
{
	Darknet::Network net{};
	std::string reason;
	EXPECT_FALSE(Darknet::Apple::mps_prepare_validation_network(net, net, reason));
	EXPECT_EQ(reason, "validation requires a distinct network with matching layers");
}

TEST(AppleDiagnostics, RuntimeReportsOwnedStatus)
{
	std::string reason;
	const bool available = Darknet::Apple::mps_runtime_available(reason);
	EXPECT_FALSE(reason.empty());
	EXPECT_EQ(available, Darknet::Apple::mps_runtime_available());
	if (available)
	{
		EXPECT_EQ(reason, "ok");
	}
	std::string training_reason;
	const bool training_available = Darknet::Apple::training_graph_runtime_available(training_reason);
	EXPECT_FALSE(training_reason.empty());
	if (training_available)
	{
		EXPECT_EQ(training_reason, "ok");
	}
}

TEST(AppleDiagnostics, LayerFailureDoesNotBorrowDiagnosticStorage)
{
	if (not Darknet::Apple::mps_runtime_available())
	{
		GTEST_SKIP() << "Apple GPU unavailable";
	}
	Darknet::Layer layer{};
	layer.groups = 0;
	std::string reason;
	EXPECT_FALSE(Darknet::Apple::mps_convolution_forward(layer, nullptr, nullptr, reason));
	EXPECT_EQ(reason, "invalid groups");
	const std::string saved = reason;
	EXPECT_TRUE(Darknet::Apple::mps_runtime_available(reason));
	EXPECT_EQ(saved, "invalid groups");
	EXPECT_EQ(reason, "ok");
}

TEST(AppleDiagnostics, KernelNamesAcceptOwnedAndTemporaryStrings)
{
	if (not Darknet::Apple::metal_is_available())
	{
		GTEST_SKIP() << "Apple GPU unavailable";
	}
	const std::string missing = "darknet_test_missing_kernel";
	EXPECT_FALSE(Darknet::Apple::metal_kernel_available(missing));
	const std::string error = Darknet::Apple::metal_last_error();
	EXPECT_NE(error.find(missing), std::string::npos);
	EXPECT_TRUE(Darknet::Apple::metal_kernel_available(std::string("relu_kernel")));
	EXPECT_TRUE(Darknet::Apple::metal_last_error().empty());
	EXPECT_NE(error.find(missing), std::string::npos);
	EXPECT_FALSE(Darknet::Apple::metal_kernel_available(std::string{}));
}
#endif
