#include <gtest/gtest.h>
#include "darknet.hpp"


TEST(FormatDuration, Normal)
{
	// reminder:  1000 microseconds makes 1 millisecond

	// anything smaller than "seconds" should be padded
	ASSERT_STREQ("  0.010 milliseconds",	Darknet::format_duration_string(std::chrono::nanoseconds(9999)									).c_str());
	ASSERT_STREQ("  0.010 milliseconds",	Darknet::format_duration_string(std::chrono::nanoseconds(10000)									).c_str());
	ASSERT_STREQ("  0.011 milliseconds",	Darknet::format_duration_string(std::chrono::nanoseconds(10555)									).c_str());
	ASSERT_STREQ("  0.123 milliseconds",	Darknet::format_duration_string(std::chrono::microseconds(123) 									).c_str());
	ASSERT_STREQ("  0.999 milliseconds",	Darknet::format_duration_string(std::chrono::milliseconds(1)	- std::chrono::microseconds(1) 	).c_str());
	ASSERT_STREQ(" 10.000 milliseconds",	Darknet::format_duration_string(std::chrono::milliseconds(10)									).c_str());
	ASSERT_STREQ("999.000 milliseconds",	Darknet::format_duration_string(std::chrono::milliseconds(999)									).c_str());

	// anything greater than "milliseconds" should have the trailing ".000" removed if it is all zeros
	ASSERT_STREQ("1.999 seconds",			Darknet::format_duration_string(std::chrono::seconds(2)			- std::chrono::milliseconds(1)	).c_str());
	ASSERT_STREQ("2.001 seconds",			Darknet::format_duration_string(std::chrono::seconds(2)			+ std::chrono::milliseconds(1)	).c_str());
	ASSERT_STREQ("119.999 seconds",			Darknet::format_duration_string(std::chrono::seconds(120)		- std::chrono::milliseconds(1)	).c_str());
	ASSERT_STREQ("120 seconds",				Darknet::format_duration_string(std::chrono::seconds(120)										).c_str());
	ASSERT_STREQ("2 minutes",				Darknet::format_duration_string(std::chrono::seconds(120)		+ std::chrono::milliseconds(1)	).c_str());
	ASSERT_STREQ("2.001 minutes",			Darknet::format_duration_string(std::chrono::seconds(120)		+ std::chrono::milliseconds(40)	).c_str());
	ASSERT_STREQ("2.017 minutes",			Darknet::format_duration_string(std::chrono::seconds(120)		+ std::chrono::seconds(1)		).c_str());
	ASSERT_STREQ("5.500 hours",				Darknet::format_duration_string(std::chrono::hours(5)			+ std::chrono::minutes(30)		).c_str());
	ASSERT_STREQ("2.458 days",				Darknet::format_duration_string(std::chrono::hours(59)											).c_str());
	ASSERT_STREQ("2.46 days",				Darknet::format_duration_string(std::chrono::hours(59), 2										).c_str());
	ASSERT_STREQ("7.143 weeks",				Darknet::format_duration_string(std::chrono::hours(24 * 50)										).c_str());
}


TEST(FormatDuration, Decimals)
{
	ASSERT_STREQ("1.500 seconds", Darknet::format_duration_string(std::chrono::milliseconds(1500)).c_str());
	ASSERT_STREQ("1.500 seconds", Darknet::format_duration_string(std::chrono::milliseconds(1500), 3).c_str());
	ASSERT_STREQ("1.50 seconds", Darknet::format_duration_string(std::chrono::milliseconds(1500), 2).c_str());
	ASSERT_STREQ("1.5 seconds", Darknet::format_duration_string(std::chrono::milliseconds(1500), 1).c_str());

	ASSERT_STREQ("2 seconds", Darknet::format_duration_string(std::chrono::milliseconds(2000)).c_str());
	ASSERT_STREQ("2 seconds", Darknet::format_duration_string(std::chrono::milliseconds(2000), 3).c_str());
	ASSERT_STREQ("2 seconds", Darknet::format_duration_string(std::chrono::milliseconds(2000), 10).c_str());

	ASSERT_STREQ("2.1 minutes", Darknet::format_duration_string(std::chrono::milliseconds(123456), 1).c_str());
	ASSERT_STREQ("2.058 minutes", Darknet::format_duration_string(std::chrono::milliseconds(123456)).c_str());
	ASSERT_STREQ("2.058 minutes", Darknet::format_duration_string(std::chrono::milliseconds(123456), 3).c_str());
	ASSERT_STREQ("2.057600 minutes", Darknet::format_duration_string(std::chrono::milliseconds(123456), 6).c_str());
}


TEST(FormatDuration, Zero)
{
	ASSERT_STREQ("unknown", Darknet::format_duration_string(std::chrono::high_resolution_clock::duration::zero()).c_str());
	ASSERT_STREQ("unknown", Darknet::format_duration_string(std::chrono::nanoseconds(0)).c_str());
	ASSERT_STREQ("unknown", Darknet::format_duration_string(std::chrono::seconds(0)).c_str());
	ASSERT_STREQ("unknown", Darknet::format_duration_string(std::chrono::seconds(-1)).c_str());
	ASSERT_STREQ("unknown", Darknet::format_duration_string(std::chrono::hours(-99)).c_str());

	ASSERT_STREQ("unknown", Darknet::format_duration_string(std::chrono::minutes(0)).c_str());
	ASSERT_STREQ("unknown", Darknet::format_duration_string(std::chrono::minutes(0), 5).c_str());
	ASSERT_STREQ("unknown", Darknet::format_duration_string(std::chrono::minutes(0), 25).c_str());
}


TEST(FormatDuration, SingularAndPlural)
{
	ASSERT_STREQ("  1.000 millisecond"	, Darknet::format_duration_string(std::chrono::milliseconds(1)).c_str());

	ASSERT_STREQ("999.000 milliseconds"	, Darknet::format_duration_string(std::chrono::seconds(1) - std::chrono::milliseconds(1)).c_str());
	ASSERT_STREQ("999.999 milliseconds"	, Darknet::format_duration_string(std::chrono::seconds(1) - std::chrono::microseconds(1)).c_str());
	ASSERT_STREQ("1000.000 milliseconds", Darknet::format_duration_string(std::chrono::seconds(1) - std::chrono::nanoseconds(1)).c_str());

	// this next one is **EXACTLY** 1 second
	ASSERT_STREQ("1 second"				, Darknet::format_duration_string(std::chrono::seconds(1) + std::chrono::milliseconds(0)).c_str());

	// slightly more than 1 second, but gets rounded to "1"
	ASSERT_STREQ("1 second"				, Darknet::format_duration_string(std::chrono::seconds(1) + std::chrono::nanoseconds(1)).c_str());
	ASSERT_STREQ("1 second"				, Darknet::format_duration_string(std::chrono::seconds(1) + std::chrono::microseconds(499)).c_str());

	ASSERT_STREQ("1.001 seconds"		, Darknet::format_duration_string(std::chrono::seconds(1) + std::chrono::microseconds(501)).c_str());
	ASSERT_STREQ("1.001 seconds"		, Darknet::format_duration_string(std::chrono::seconds(1) + std::chrono::milliseconds(1)).c_str());
	ASSERT_STREQ("2 seconds"			, Darknet::format_duration_string(std::chrono::seconds(2) + std::chrono::milliseconds(0)).c_str());
	ASSERT_STREQ("2.001 seconds"		, Darknet::format_duration_string(std::chrono::seconds(2) + std::chrono::milliseconds(1)).c_str());
}


TEST(FormatDuration, HighRange)
{
	ASSERT_STREQ("24 hours"		, Darknet::format_duration_string(std::chrono::hours(24)		).c_str());
	ASSERT_STREQ("2.292 days"	, Darknet::format_duration_string(std::chrono::hours(55)		).c_str());
	ASSERT_STREQ("2 weeks"		, Darknet::format_duration_string(std::chrono::hours(336)		).c_str());
	ASSERT_STREQ("5.952 weeks"	, Darknet::format_duration_string(std::chrono::hours(1000)		).c_str());

	ASSERT_STREQ("9.994 weeks"	, Darknet::format_duration_string(std::chrono::hours(1679)		).c_str());
	ASSERT_STREQ("10 weeks"		, Darknet::format_duration_string(std::chrono::hours(1680)		).c_str());
	ASSERT_STREQ("10.006 weeks"	, Darknet::format_duration_string(std::chrono::hours(1681)		).c_str());

	ASSERT_STREQ("9.994 weeks"	, Darknet::format_duration_string(std::chrono::hours(1679), 3	).c_str());
	ASSERT_STREQ("10 weeks"		, Darknet::format_duration_string(std::chrono::hours(1680), 3	).c_str());
	ASSERT_STREQ("10.006 weeks"	, Darknet::format_duration_string(std::chrono::hours(1681), 3	).c_str());

	ASSERT_STREQ("10 weeks"		, Darknet::format_duration_string(std::chrono::hours(1679), 1	).c_str());
	ASSERT_STREQ("10 weeks"		, Darknet::format_duration_string(std::chrono::hours(1680), 1	).c_str());
	ASSERT_STREQ("10 weeks"		, Darknet::format_duration_string(std::chrono::hours(1681), 1	).c_str());

	ASSERT_STREQ("52 weeks"		, Darknet::format_duration_string(std::chrono::hours(24 * 365), 0).c_str());
	ASSERT_STREQ("52 weeks"		, Darknet::format_duration_string(std::chrono::hours(24 * 7 * 52), 10).c_str());
}


TEST(FormatDuration, MidRange)
{
	ASSERT_STREQ("7.006 minutes"	, Darknet::format_duration_string(std::chrono::minutes(7) + std::chrono::milliseconds(335)		).c_str());
	ASSERT_STREQ("7.006 minutes"	, Darknet::format_duration_string(std::chrono::minutes(7) + std::chrono::milliseconds(335), 3	).c_str());
	ASSERT_STREQ("7.01 minutes"		, Darknet::format_duration_string(std::chrono::minutes(7) + std::chrono::milliseconds(335), 2	).c_str());
	ASSERT_STREQ("7 minutes"		, Darknet::format_duration_string(std::chrono::minutes(7) + std::chrono::milliseconds(335), 1	).c_str());
	ASSERT_STREQ("7 minutes"		, Darknet::format_duration_string(std::chrono::minutes(7) + std::chrono::milliseconds(335), 0	).c_str());

	ASSERT_STREQ("3.250 seconds"	, Darknet::format_duration_string(std::chrono::milliseconds(3250)	).c_str());
	ASSERT_STREQ("3.250 seconds"	, Darknet::format_duration_string(std::chrono::milliseconds(3250), 3).c_str());
	ASSERT_STREQ("3.250000 seconds"	, Darknet::format_duration_string(std::chrono::milliseconds(3250), 6).c_str());
}


TEST(FormatDuration, LowRange)
{
	// with small numbers like milliseconds, there are 2 additional rules:
	//		1) the integer portion is padded out to 3 digits
	//		2) trailing ".000" digits are not removed from the value

	ASSERT_STREQ("  1.275 milliseconds", Darknet::format_duration_string(std::chrono::milliseconds(1) + std::chrono::microseconds(275)).c_str());
	ASSERT_STREQ("  1.275 milliseconds", Darknet::format_duration_string(std::chrono::milliseconds(1) + std::chrono::microseconds(275), 3).c_str());
	ASSERT_STREQ("  1.275 milliseconds", Darknet::format_duration_string(std::chrono::microseconds(1275)).c_str());
	ASSERT_STREQ("  1.275 milliseconds", Darknet::format_duration_string(std::chrono::microseconds(1275), 3).c_str());
	ASSERT_STREQ("  1.275000 milliseconds", Darknet::format_duration_string(std::chrono::microseconds(1275), 6).c_str());

	ASSERT_STREQ("  1.000 millisecond", Darknet::format_duration_string(std::chrono::microseconds(1000)).c_str());
	ASSERT_STREQ("  1.000 millisecond", Darknet::format_duration_string(std::chrono::milliseconds(1)).c_str());

	ASSERT_STREQ("999.000 milliseconds", Darknet::format_duration_string(std::chrono::milliseconds(999)).c_str());
	ASSERT_STREQ("999.0 milliseconds", Darknet::format_duration_string(std::chrono::milliseconds(999), 1).c_str());
	ASSERT_STREQ("999.999 milliseconds", Darknet::format_duration_string(std::chrono::milliseconds(999) + std::chrono::microseconds(999), 3).c_str());
	ASSERT_STREQ("1000.0 milliseconds", Darknet::format_duration_string(std::chrono::milliseconds(999) + std::chrono::microseconds(999), 1).c_str());
	ASSERT_STREQ("1 second", Darknet::format_duration_string(std::chrono::seconds(1)).c_str());
	ASSERT_STREQ("1 second", Darknet::format_duration_string(std::chrono::milliseconds(1000)).c_str());
	ASSERT_STREQ("1.001 seconds", Darknet::format_duration_string(std::chrono::milliseconds(1001)).c_str());
}
