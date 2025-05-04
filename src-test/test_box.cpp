#include <gtest/gtest.h>
#include "darknet_internal.hpp"

TEST(IoUCVRect, Zero)
{
	// no overlap between these rectangles, IoU should be exactly zero

	ASSERT_FLOAT_EQ(0.0f, Darknet::iou(cv::Rect(0, 0, 10, 10)	, cv::Rect(20, 20, 10, 10)	));
	ASSERT_FLOAT_EQ(0.0f, Darknet::iou(cv::Rect(0, 0, 0, 0)		, cv::Rect(20, 20, 10, 10)	));
	ASSERT_FLOAT_EQ(0.0f, Darknet::iou(cv::Rect(0, 0, 0, 0)		, cv::Rect(0, 0, 0, 0)		));
}


TEST(IoUCVRect, One)
{
	// perfect overlap between these rectangles, IoU should be exactly one

	ASSERT_FLOAT_EQ(1.0f, Darknet::iou(cv::Rect(0, 0, 10, 10)	, cv::Rect(0, 0, 10, 10)	));
	ASSERT_FLOAT_EQ(1.0f, Darknet::iou(cv::Rect(0, 0, 1, 1)		, cv::Rect(0, 0, 1, 1)		));
	ASSERT_FLOAT_EQ(1.0f, Darknet::iou(cv::Rect(10, 10, 10, 10)	, cv::Rect(10, 10, 10, 10)	));
}


TEST(IoUCVRect, T1)
{
	/* Rectangles are [0, 0, 10, 10] and [5, 5, 10, 10]
	 * Intersection is 5 * 5 = 25
	 * Union is 10 * 10 + 10 * 10 - 25 = 175
	 * IoU = 25 / 175 = 1 / 7 = 0.142857143
	 */
	ASSERT_FLOAT_EQ(1.0f / 7.0f, Darknet::iou(cv::Rect(0, 10, 10, 10), cv::Rect(5, 5, 10, 10)));
}


TEST(IoUCVRect, T2)
{
	/* Rectangles are [10, 2, 10, 10] and [5, 2, 10, 8]
	 * Intersection is 40
	 * Union is 10 * 10 + 10 * 8 - 40 = 140
	 * IoU = 40 / 140 = 2 / 7 = 0.285714286
	 */
	ASSERT_FLOAT_EQ(2.0f / 7.0f, Darknet::iou(cv::Rect(10, 2, 10, 10), cv::Rect(5, 2, 10, 8)));
}


TEST(IoUCVRect, T3)
{
	/* Rectangles are [50, 100, 150, 200] and [80, 120, 140, 190]
	 * Intersection is 120 * 180 = 21600
	 * Union is 30000 + 26600 - 21600 = 35000
	 * IoU = 21600 / 35000 = 108/175 = 0.617142857
	 */
	ASSERT_FLOAT_EQ(108.0f / 175.0f, Darknet::iou(cv::Rect(50, 100, 150, 200), cv::Rect(80, 120, 140, 190)));
}


TEST(IoUCVRect, Range)
{
	/* Generate a bunch of random rectangles, check the IoU, and perform some simple validation on each one to ensure the
	 * values are all within the expected range.  E.g., IoU must be >= 0.0 and <= 1.0.  And if the intersection is zero,
	 * the the IoU must always be zero.
	 */

	const size_t number_of_rectangles = 50000;

	std::vector<cv::Rect> v;
	v.reserve(number_of_rectangles);
	while (v.size() < number_of_rectangles)
	{
		cv::Rect r(random_gen(0, 2000), random_gen(0, 2000), random_gen(1, 2000), random_gen(1, 2000));
		v.push_back(r);
	}

	std::chrono::high_resolution_clock::duration duration = std::chrono::milliseconds(0);

	for (size_t idx = 0; idx < v.size() - 1; idx ++)
	{
		const auto & r1 = v[idx + 0];
		const auto & r2 = v[idx + 1];

		const auto timestamp1 = std::chrono::high_resolution_clock::now();
		const auto iou = Darknet::iou(r1, r2);
		const auto timestamp2 = std::chrono::high_resolution_clock::now();
		duration += (timestamp2 - timestamp1);

//		std::cout << "r1=" << r1 << " r2=" << r2 << " iou=" << iou << std::endl;

		ASSERT_GE(iou, 0.0f);
		ASSERT_LE(iou, 1.0f);

		const float intersection = (r1 & r2).area();
		ASSERT_GE(intersection, 0.0f);

		if (intersection == 0.0f)
		{
			ASSERT_EQ(iou, 0.0f);
		}
		else
		{
			ASSERT_GT(iou, 0.0f);
		}
	}

//	std::cout << "IoU took " << Darknet::format_duration_string(duration) << std::endl;
}


TEST(IoUCVRect2f, Zero)
{
	// no overlap between these rectangles, IoU should be exactly zero

	ASSERT_FLOAT_EQ(0.0f, Darknet::iou(cv::Rect2f(0.0f, 0.0f, 0.1f, 0.1f), cv::Rect2f(0.2f, 0.2f, 0.1f, 0.1f)));
	ASSERT_FLOAT_EQ(0.0f, Darknet::iou(cv::Rect2f(0.0f, 0.0f, 0.0f, 0.0f), cv::Rect2f(0.2f, 0.2f, 0.1f, 0.1f)));
	ASSERT_FLOAT_EQ(0.0f, Darknet::iou(cv::Rect2f(0.0f, 0.0f, 0.0f, 0.0f), cv::Rect2f(0.0f, 0.0f, 0.0f, 0.0f)));
	ASSERT_FLOAT_EQ(0.0f, Darknet::iou(cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f), cv::Rect2f(0.0f, 0.0f, 0.0f, 0.0f)));
	ASSERT_FLOAT_EQ(0.0f, Darknet::iou(cv::Rect2f(0.0f, 0.0f, 0.0f, 0.0f), cv::Rect2f(0.0f, 0.0f, 1.0f, 1.0f)));
}


TEST(IoUCVRect2f, One)
{
	// perfect overlap between these rectangles, IoU should be exactly one

	ASSERT_FLOAT_EQ(1.0f, Darknet::iou(cv::Rect2f(0.00f, 0.00f, 0.10f, 0.10f), cv::Rect2f(0.00f, 0.00f, 0.10f, 0.10f)));
	ASSERT_FLOAT_EQ(1.0f, Darknet::iou(cv::Rect2f(0.00f, 0.00f, 0.01f, 0.01f), cv::Rect2f(0.00f, 0.00f, 0.01f, 0.01f)));
	ASSERT_FLOAT_EQ(1.0f, Darknet::iou(cv::Rect2f(0.10f, 0.10f, 0.10f, 0.10f), cv::Rect2f(0.10f, 0.10f, 0.10f, 0.10f)));
	ASSERT_FLOAT_EQ(1.0f, Darknet::iou(cv::Rect2f(0.25f, 0.25f, 0.50f, 0.50f), cv::Rect2f(0.25f, 0.25f, 0.50f, 0.50f)));
	ASSERT_FLOAT_EQ(1.0f, Darknet::iou(cv::Rect2f(0.00f, 0.00f, 1.00f, 1.00f), cv::Rect2f(0.00f, 0.00f, 1.00f, 1.00f)));
}


TEST(IoUCVRect2f, T1)
{
	// similar to IoUCVRect::T1, but all coordinates are divided by 100 to create cv::Rect2f
	ASSERT_FLOAT_EQ(1.0f / 7.0f, Darknet::iou(cv::Rect2f(0.00f, 0.10f, 0.10f, 0.10f), cv::Rect2f(0.05f, 0.05f, 0.10f, 0.10f)));
}


TEST(IoUCVRect2f, T2)
{
	// similar to IoUCVRect::T2, but all coordinates are divided by 100 to create cv::Rect2f
	ASSERT_FLOAT_EQ(2.0f / 7.0f, Darknet::iou(cv::Rect2f(0.10f, 0.02f, 0.10f, 0.10f), cv::Rect2f(0.05f, 0.02f, 0.10f, 0.08f)));
}


TEST(IoUCVRect2f, T3)
{
	// similar to IoUCVRect::T3, but all coordinates are divided by 1000 to create cv::Rect2f
	ASSERT_FLOAT_EQ(108.0f / 175.0f, Darknet::iou(cv::Rect2f(0.050f, 0.100f, 0.150f, 0.200f), cv::Rect2f(0.080f, 0.120f, 0.140f, 0.190f)));
}


TEST(IoUCVRect2f, Range)
{
	/* Generate a bunch of random rectangles, check the IoU, and perform some simple validation on each one to ensure the
	 * values are all within the expected range.  E.g., IoU must be >= 0.0 and <= 1.0.  And if the intersection is zero,
	 * the the IoU must always be zero.
	 */

	const size_t number_of_rectangles = 50000;

	std::vector<cv::Rect2f> v;
	v.reserve(number_of_rectangles);
	while (v.size() < number_of_rectangles)
	{
		cv::Rect2f r(rand_uniform_strong(0.0f, 1.0f), rand_uniform_strong(0.0f, 1.0f), rand_uniform_strong(0.0f, 1.0f), rand_uniform_strong(0.0f, 1.0f));
		if (r.br().x <= 1.0f and r.br().y <= 1.0f)
		{
			v.push_back(r);
		}
	}

	std::chrono::high_resolution_clock::duration duration = std::chrono::milliseconds(0);

	for (size_t idx = 0; idx < v.size() - 1; idx ++)
	{
		const auto & r1 = v[idx + 0];
		const auto & r2 = v[idx + 1];

		const auto timestamp1 = std::chrono::high_resolution_clock::now();
		const auto iou = Darknet::iou(r1, r2);
		const auto timestamp2 = std::chrono::high_resolution_clock::now();
		duration += (timestamp2 - timestamp1);

//		std::cout << "r1=" << r1 << " r2=" << r2 << " iou=" << iou << std::endl;

		ASSERT_GE(iou, 0.0f);
		ASSERT_LE(iou, 1.0f);

		const float intersection = (r1 & r2).area();
		ASSERT_GE(intersection, 0.0f);

		if (intersection == 0.0f)
		{
			ASSERT_EQ(iou, 0.0f);
		}
		else
		{
			ASSERT_GT(iou, 0.0f);
		}
	}

//	std::cout << "IoU took " << Darknet::format_duration_string(duration) << std::endl;
}
