#include <gtest/gtest.h>
#include "darknet_utils.hpp"


const size_t ITERATIONS = 500;


TEST(Random, rand_normal)
{
	const float min = -5.0f;
	const float max = 5.0f;

	float lo = max;
	float hi = min;

	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float f = rand_normal();
		std::cout << "rand_normal() i=" << i << " f=" << f << std::endl;

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
		std::cout << "rand_uniform() i=" << i << " f=" << f << std::endl;

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


TEST(Random, random_float)
{
	const float min = 0.0f;
	const float max = 1.0f;

	float lo = max;
	float hi = min;

	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float f = random_float();
		std::cout << "random_float() i=" << i << " f=" << f << std::endl;

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
	std::cout << "random_float() lo=" << lo << " hi=" << hi << " maximum=" << maximum_range << " allowed=" << allowed_range << " actual=" << actual_range << std::endl;
	ASSERT_GT(actual_range, allowed_range);
}


TEST(Random, rand_scale)
{
	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float f = rand_scale(1.0f);
		std::cout << "rand_scale() i=" << i << " f=" << f << std::endl;
	}
}


TEST(Random, rand_precalc_random)
{
	for (size_t i = 0; i < ITERATIONS; i++)
	{
		const float f = rand_precalc_random(0.0f, 1.0f, 5.0f);
		std::cout << "rand_precalc_random() i=" << i << " f=" << f << std::endl;
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
		std::cout << "rand_int() i=" << i << " r=" << r << std::endl;

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
		std::cout << "rand_uint() i=" << i << " r=" << r << std::endl;

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
