#include <gtest/gtest.h>
#include "darknet.hpp"


TEST(NN, FreeNullPtr)
{
	Darknet::NetworkPtr ptr = nullptr;

	ASSERT_TRUE(ptr == nullptr);
	Darknet::free_neural_network(ptr);
	ASSERT_TRUE(ptr == nullptr);
}


TEST(Darknet, FormatDuration)
{
	// reminder:  1000 microseconds makes 1 millisecond

	ASSERT_STREQ("  0.123 milliseconds",	Darknet::format_duration_string(std::chrono::microseconds(123) 									).c_str());
	ASSERT_STREQ("  0.999 milliseconds",	Darknet::format_duration_string(std::chrono::milliseconds(1)	- std::chrono::microseconds(1) 	).c_str());
	ASSERT_STREQ(" 10.000 milliseconds",	Darknet::format_duration_string(std::chrono::milliseconds(10)									).c_str());
	ASSERT_STREQ("999.000 milliseconds",	Darknet::format_duration_string(std::chrono::milliseconds(999)									).c_str());
	ASSERT_STREQ("  1.999 seconds",			Darknet::format_duration_string(std::chrono::seconds(2)			- std::chrono::milliseconds(1)	).c_str());
	ASSERT_STREQ("  2.001 seconds",			Darknet::format_duration_string(std::chrono::seconds(2)			+ std::chrono::milliseconds(1)	).c_str());
	ASSERT_STREQ("119.999 seconds",			Darknet::format_duration_string(std::chrono::seconds(120)		- std::chrono::milliseconds(1)	).c_str());
	ASSERT_STREQ("120.000 seconds",			Darknet::format_duration_string(std::chrono::seconds(120)										).c_str());
	ASSERT_STREQ("  2.000 minutes",			Darknet::format_duration_string(std::chrono::seconds(120)		+ std::chrono::milliseconds(1)	).c_str());
	ASSERT_STREQ("  2.001 minutes",			Darknet::format_duration_string(std::chrono::seconds(120)		+ std::chrono::milliseconds(40)	).c_str());
	ASSERT_STREQ("  2.017 minutes",			Darknet::format_duration_string(std::chrono::seconds(120)		+ std::chrono::seconds(1)		).c_str());
	ASSERT_STREQ("  5.500 hours",			Darknet::format_duration_string(std::chrono::hours(5)			+ std::chrono::minutes(30)		).c_str());
	ASSERT_STREQ("  2.458 days",			Darknet::format_duration_string(std::chrono::hours(59)											).c_str());
	ASSERT_STREQ("  2.46 days",				Darknet::format_duration_string(std::chrono::hours(59), 2										).c_str());
	ASSERT_STREQ(" 50.000 days",			Darknet::format_duration_string(std::chrono::hours(24 * 50)										).c_str());
}


TEST(Darknet, Trim)
{
	ASSERT_STREQ("testing"		, Darknet::trim(" testing "			).c_str());
	ASSERT_STREQ("whitespace"	, Darknet::trim("\n whitespace\r\n"	).c_str());
	ASSERT_STREQ("whitespace"	, Darknet::trim(" \twhitespace\n"	).c_str());
	ASSERT_STREQ("whitespace"	, Darknet::trim("\t whitespace\r"	).c_str());
	ASSERT_STREQ(""				, Darknet::trim("   "				).c_str());
}


TEST(Darknet, Lowercase)
{
	ASSERT_STREQ("012345"		, Darknet::lowercase("012345"		).c_str());
	ASSERT_STREQ("  testing  "	, Darknet::lowercase("  TeStInG  "	).c_str());
	ASSERT_STREQ(""				, Darknet::lowercase(""				).c_str());
}
