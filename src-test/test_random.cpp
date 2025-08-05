#include <gtest/gtest.h>
#include "darknet_utils.hpp"


const size_t ITERATIONS = 1000;


TEST(Random, rand_normal)
{
	const float min = -5.0f;
	const float max = 5.0f;

	float lo = max;
	float hi = min;

	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float f = rand_normal();
//		std::cout << "rand_normal() i=" << i << " f=" << f << std::endl;

		ASSERT_GE(f, min);
		ASSERT_LT(f, max);
		ASSERT_FALSE(std::isinf(f));
		ASSERT_FALSE(std::isnan(f));
		ASSERT_TRUE(std::isnormal(f));
		ASSERT_TRUE(f == f);

		if (f < lo)	lo = f;
		if (f > hi) hi = f;
	}

	// see if the generated floats cover the full range of values between "min" and "max"
	// (if this fails...did you decrease the number of iterations to something so low that we're not getting enough samples?)
	const float maximum_range	= max - min;
	const float allowed_range	= 0.95f * maximum_range; // within 95% of the maximum possible range
	const float actual_range	= hi - lo;
	std::cout << "rand_normal() lo=" << lo << " hi=" << hi << " maximum=" << maximum_range << " allowed=" << allowed_range << " actual=" << actual_range << std::endl;
	ASSERT_GT(actual_range, allowed_range);
}


TEST(Random, rand_uniform)
{
	const float min = -5.0f;
	const float max = 10.0f;

	float lo = max;
	float hi = min;

	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float f = rand_uniform(min, max);
//		std::cout << "rand_uniform() i=" << i << " f=" << f << std::endl;

		ASSERT_GE(f, min);
		ASSERT_LT(f, max);
		ASSERT_FALSE(std::isinf(f));
		ASSERT_FALSE(std::isnan(f));
		ASSERT_TRUE(std::isnormal(f));
		ASSERT_TRUE(f == f);

		if (f < lo)	lo = f;
		if (f > hi) hi = f;
	}

	// see if the generated floats cover the full range of values between "min" and "max"
	// (if this fails...did you decrease the number of iterations to something so low that we're not getting enough samples?)
	const float maximum_range	= max - min;
	const float allowed_range	= 0.95f * maximum_range; // within 95% of the maximum possible range
	const float actual_range	= hi - lo;
	std::cout << "rand_uniform() lo=" << lo << " hi=" << hi << " maximum=" << maximum_range << " allowed=" << allowed_range << " actual=" << actual_range << std::endl;
	ASSERT_GT(actual_range, allowed_range);
}


TEST(Random, rand_uniform_many)
{
	float array[ITERATIONS];
	for (size_t i = 0; i < ITERATIONS; i++)
	{
		array[i] = -1.0f;
	}

	const float min = 0.0f;
	const float max = 100.0f;

	float lo = max;
	float hi = min;

	rand_uniform_many(array, ITERATIONS, min, max);
	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float & f = array[i];

		ASSERT_GE(f, min);
		ASSERT_LT(f, max);
		ASSERT_FALSE(std::isinf(f));
		ASSERT_FALSE(std::isnan(f));
		ASSERT_TRUE(std::isnormal(f));
		ASSERT_TRUE(f == f);

		if (f < lo)	lo = f;
		if (f > hi) hi = f;
	}

	// see if the generated floats cover the full range of values between "min" and "max"
	// (if this fails...did you decrease the number of iterations to something so low that we're not getting enough samples?)
	const float maximum_range	= max - min;
	const float allowed_range	= 0.95f * maximum_range; // within 95% of the maximum possible range
	const float actual_range	= hi - lo;
	std::cout << "rand_uniform_many() lo=" << lo << " hi=" << hi << " maximum=" << maximum_range << " allowed=" << allowed_range << " actual=" << actual_range << std::endl;
	ASSERT_GT(actual_range, allowed_range);

	return;
}


TEST(Random, rand_float)
{
	const float min = 0.0f;
	const float max = 1.0f;

	float lo = max;
	float hi = min;

	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float f = rand_float();
//		std::cout << "rand_float() i=" << i << " f=" << f << std::endl;

		ASSERT_GE(f, min);
		ASSERT_LT(f, max);
		ASSERT_FALSE(std::isinf(f));
		ASSERT_FALSE(std::isnan(f));
		ASSERT_TRUE(std::isnormal(f));
		ASSERT_TRUE(f == f);

		if (f < lo)	lo = f;
		if (f > hi) hi = f;
	}

	// see if the generated floats cover the full range of values between "min" and "max"
	// (if this fails...did you decrease the number of iterations to something so low that we're not getting enough samples?)
	const float maximum_range	= max - min;
	const float allowed_range	= 0.95f * maximum_range; // within 95% of the maximum possible range
	const float actual_range	= hi - lo;
	std::cout << "rand_float() lo=" << lo << " hi=" << hi << " maximum=" << maximum_range << " allowed=" << allowed_range << " actual=" << actual_range << std::endl;
	ASSERT_GT(actual_range, allowed_range);
}


TEST(Random, rand_scale)
{
	const float min = 0.5f;
	const float max = 2.0f;

	float lo = max;
	float hi = min;

	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float f = rand_scale(0.5f);
//		std::cout << "rand_scale() i=" << i << " f=" << f << std::endl;

		ASSERT_GE(f, min);
		ASSERT_LT(f, max);
		ASSERT_FALSE(std::isinf(f));
		ASSERT_FALSE(std::isnan(f));
		ASSERT_TRUE(std::isnormal(f));
		ASSERT_TRUE(f == f);

		if (f < lo)	lo = f;
		if (f > hi) hi = f;
	}

	// see if the generated floats cover the full range of values between "min" and "max"
	// (if this fails...did you decrease the number of iterations to something so low that we're not getting enough samples?)
	const float maximum_range	= max - min;
	const float allowed_range	= 0.95f * maximum_range; // within 95% of the maximum possible range
	const float actual_range	= hi - lo;
	std::cout << "rand_scale() lo=" << lo << " hi=" << hi << " maximum=" << maximum_range << " allowed=" << allowed_range << " actual=" << actual_range << std::endl;
	ASSERT_GT(actual_range, allowed_range);
}


TEST(Random, rand_precalc_random)
{
	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float f = rand_precalc_random(0.5f, 1.75f, 3.0f);
//		std::cout << "rand_precalc_random() i=" << i << " f=" << f << std::endl;
		ASSERT_FLOAT_EQ(f, 4.25f);
	}
}


TEST(Random, rand_int)
{
	const int min = -50.0f;
	const int max = 50.0f;

	int lo = max;
	int hi = min;

	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const int r = rand_int(min, max);
//		std::cout << "rand_int() i=" << i << " r=" << r << std::endl;

		ASSERT_GE(r, min);
		ASSERT_LE(r, max);
		ASSERT_FALSE(std::isinf(r));
		ASSERT_FALSE(std::isnan(r));
		ASSERT_TRUE(r == r);

		if (r < lo)	lo = r;
		if (r > hi) hi = r;
	}

	// see if the generated floats cover the full range of values between "min" and "max"
	// (if this fails...did you decrease the number of iterations to something so low that we're not getting enough samples?)
	const int maximum_range	= max - min;
	const int allowed_range	= 0.95f * maximum_range; // within 95% of the maximum possible range
	const int actual_range	= hi - lo;
	std::cout << "rand_int() lo=" << lo << " hi=" << hi << " maximum=" << maximum_range << " allowed=" << allowed_range << " actual=" << actual_range << std::endl;
	ASSERT_GT(actual_range, allowed_range);
}


TEST(Random, rand_uint)
{
	const unsigned int min = 0.0f;
	const unsigned int max = 100.0f;

	unsigned int lo = max;
	unsigned int hi = min;

	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const unsigned int r = rand_uint(min, max);
//		std::cout << "rand_uint() i=" << i << " r=" << r << std::endl;

		ASSERT_GE(r, min);
		ASSERT_LE(r, max);
		ASSERT_FALSE(std::isinf(r));
		ASSERT_FALSE(std::isnan(r));
		ASSERT_TRUE(r == r);

		if (r < lo)	lo = r;
		if (r > hi) hi = r;
	}

	// see if the generated floats cover the full range of values between "min" and "max"
	// (if this fails...did you decrease the number of iterations to something so low that we're not getting enough samples?)
	const int maximum_range	= max - min;
	const int allowed_range	= 0.95f * maximum_range; // within 95% of the maximum possible range
	const int actual_range	= hi - lo;
	std::cout << "rand_uint() lo=" << lo << " hi=" << hi << " maximum=" << maximum_range << " allowed=" << allowed_range << " actual=" << actual_range << std::endl;
	ASSERT_GT(actual_range, allowed_range);
}


TEST(Random, custom_hash)
{
	std::set<unsigned long> hashes;
	std::map<std::string, unsigned long> strings =
	{
		{"Darknet/YOLO"	, 0},	{"Darknet YOLO"	, 0},	{"darknet/yolo"	, 0},	{"darknet yolo"	, 0},
		{"Hello, world!", 0},	{"hello, world!", 0},	{""				, 0},	{" "			, 0},
		{"0"			, 0},	{"01"			, 0},	{"012"			, 0},	{"0123"			, 0},
		{"01234"		, 0},	{"012345"		, 0},	{"0123456"		, 0},	{"01234567"		, 0},
		{"012345678"	, 0},	{"0123456789"	, 0},	{"9876543210"	, 0},	{"0.jpg"		, 0},
		{"0.txt"		, 0},	{"1.jpg"		, 0},	{"1.txt"		, 0},	{"a.jpg"		, 0},
		{"a.txt"		, 0},	{"b.jpg"		, 0},	{"b.txt"		, 0},	{"testing"		, 0},
		{"z testing"	, 0},	{"zz testing"	, 0},	{"zzz testing"	, 0},	{"zzzz testing"	, 0},
		{"zzzztesting"	, 0},	{"zzztesting"	, 0},	{"zztesting"	, 0},	{"ztesting"		, 0},
		{"testing"		, 0},	{"gnitset"		, 0},	{"\t"			, 0},	{"\r"			, 0},
		{"\n"			, 0},
	};

	for (auto & [key, val] : strings)
	{
		const auto hash = custom_hash(key.c_str());
		std::cout << "key=" << key << " hash=" << hash << std::endl;
		ASSERT_EQ(0, hashes.count(hash));
		hashes.insert(hash);
		strings[key] = hash;
	}

	// we should have exactly the same number of hashes as strings
	ASSERT_EQ(hashes.size(), strings.size());
}
