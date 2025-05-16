#include <gtest/gtest.h>
#include "darknet_internal.hpp"


const int IMG_W = 1024;
const int IMG_H = 768;
const int IMG_C = 3;
const char * const IMG_FN = "test.png";


TEST(Image, Initialize)
{
	cv::Mat mat(IMG_H, IMG_W, CV_8UC3);
	cv::randu(mat, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

	cv::putText(mat, "testing"	, {50, 150}, cv::FONT_HERSHEY_PLAIN, 10.0, {0, 0, 0		}, 8);
	cv::putText(mat, "red"		, {50, 300}, cv::FONT_HERSHEY_PLAIN, 10.0, {0, 0, 255	}, 8);
	cv::putText(mat, "green"	, {50, 450}, cv::FONT_HERSHEY_PLAIN, 10.0, {0, 255, 0	}, 8);
	cv::putText(mat, "blue"		, {50, 600}, cv::FONT_HERSHEY_PLAIN, 10.0, {255, 0, 0	}, 8);

	const bool result = cv::imwrite(IMG_FN, mat, {cv::IMWRITE_PNG_COMPRESSION, 3});

	ASSERT_TRUE(result);
	ASSERT_FALSE(mat.empty());
	ASSERT_EQ(mat.cols, IMG_W);
	ASSERT_EQ(mat.rows, IMG_H);
}


TEST(Image, Free)
{
	Darknet::Image i1 = Darknet::load_image(IMG_FN);
	ASSERT_EQ(i1.c, IMG_C);
	ASSERT_EQ(i1.w, IMG_W);
	ASSERT_EQ(i1.h, IMG_H);
	ASSERT_NE(i1.data, nullptr);
	Darknet::free_image(i1);
	ASSERT_EQ(i1.c, 0);
	ASSERT_EQ(i1.w, 0);
	ASSERT_EQ(i1.h, 0);
	ASSERT_EQ(i1.data, nullptr);

	// freeing an image object that has already been freed should not be harmful
	Darknet::free_image(i1);
	ASSERT_EQ(i1.c, 0);
	ASSERT_EQ(i1.w, 0);
	ASSERT_EQ(i1.h, 0);
	ASSERT_EQ(i1.data, nullptr);
}


TEST(Image, Load)
{
	cv::Mat m1 = cv::imread(IMG_FN);
	ASSERT_FALSE(m1.empty());
	ASSERT_EQ(m1.channels()	, IMG_C);
	ASSERT_EQ(m1.cols		, IMG_W);
	ASSERT_EQ(m1.rows		, IMG_H);

	Darknet::Image i1 = Darknet::load_image(IMG_FN);
	ASSERT_EQ(i1.c, IMG_C);
	ASSERT_EQ(i1.w, IMG_W);
	ASSERT_EQ(i1.h, IMG_H);

	Darknet::Image i2 = Darknet::bgr_mat_to_rgb_image(m1);
	ASSERT_EQ(i2.c, i1.c);
	ASSERT_EQ(i2.w, i1.w);
	ASSERT_EQ(i2.h, i1.h);

	const size_t number_of_floats = i2.c * i2.w * i2.h;
	for (size_t idx = 0; idx < number_of_floats; idx ++)
	{
		ASSERT_FLOAT_EQ(i1.data[idx], i2.data[idx]);
	}

	cv::Mat m2 = Darknet::rgb_image_to_bgr_mat(i2);
	ASSERT_EQ(m1.cols, m2.cols);
	ASSERT_EQ(m1.rows, m2.rows);

	cv::Mat diff;
	cv::compare(m1.reshape(1), m2.reshape(1), diff, cv::CMP_NE);
	ASSERT_EQ(0, cv::countNonZero(diff));

	Darknet::free_image(i1);
	Darknet::free_image(i2);
}


TEST(Image, LoadTiming)
{
	const size_t count = 100;

	std::chrono::high_resolution_clock::duration duration = std::chrono::milliseconds(0);
	for (size_t idx = 0; idx < count; idx ++)
	{
		const auto t1 = std::chrono::high_resolution_clock::now();
		Darknet::Image img = Darknet::load_image(IMG_FN);
		const auto t2 = std::chrono::high_resolution_clock::now();

		duration += (t2 - t1);

		Darknet::free_image(img);
	}

	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	std::cout << "Loading using Darknet::load_image() " << count << " times took " << milliseconds << " milliseconds" << std::endl;

	// now try the same thing using OpenCV

	duration = std::chrono::milliseconds(0);
	for (size_t idx = 0; idx < count; idx ++)
	{
		// Darknet::load_image() not only loads the image, but also does the BGR2RGB conversion,
		// so if we're timing things we need to time both calls
		const auto t1 = std::chrono::high_resolution_clock::now();
		cv::Mat m1 = cv::imread(IMG_FN);
		cv::Mat m2;
		cv::cvtColor(m1, m2, cv::COLOR_BGR2RGB);
		const auto t2 = std::chrono::high_resolution_clock::now();

		duration += (t2 - t1);
	}

	milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	std::cout << "Loading using OpenCV's cv::imread() " << count << " times took " << milliseconds << " milliseconds" << std::endl;
}


TEST(Image, LoadAndResizeTiming)
{
	const size_t count = 100;

	std::chrono::high_resolution_clock::duration duration = std::chrono::milliseconds(0);
	for (size_t idx = 0; idx < count; idx ++)
	{
		const auto t1 = std::chrono::high_resolution_clock::now();
		Darknet::Image img = Darknet::load_image(IMG_FN, IMG_W / 3, IMG_H / 3, IMG_C);
		const auto t2 = std::chrono::high_resolution_clock::now();

		ASSERT_EQ(img.c, IMG_C		);
		ASSERT_EQ(img.w, IMG_W / 3	);
		ASSERT_EQ(img.h, IMG_H / 3	);

		Darknet::free_image(img);

		duration += (t2 - t1);
	}

	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	std::cout << "Loading and resizing using Darknet::load_image() " << count << " times took " << milliseconds << " milliseconds" << std::endl;
}


TEST(Image, Resize)
{
	Darknet::Image i1 = Darknet::load_image(IMG_FN);
	ASSERT_EQ(i1.c, IMG_C);
	ASSERT_EQ(i1.w, IMG_W);
	ASSERT_EQ(i1.h, IMG_H);

	Darknet::Image i2 = Darknet::resize_image(i1, IMG_W / 2, IMG_H / 2);
	// original image should remain unchanged
	ASSERT_EQ(i1.c, IMG_C);
	ASSERT_EQ(i1.w, IMG_W);
	ASSERT_EQ(i1.h, IMG_H);
	// new image should have been resized
	ASSERT_EQ(i2.c, IMG_C);
	ASSERT_EQ(i2.w, IMG_W / 2);
	ASSERT_EQ(i2.h, IMG_H / 2);

	cv::Mat m1 = cv::imread(IMG_FN);
	cv::Mat m2;
	cv::resize(m1, m2, {0, 0}, 0.5, 0.5);

#if 0
	cv::imshow("i1", Darknet::rgb_image_to_bgr_mat(i1));
	cv::imshow("i2", Darknet::rgb_image_to_bgr_mat(i2));
	cv::imshow("m1", m1);
	cv::imshow("m2", m2);
	cv::waitKey();
#endif

	Darknet::free_image(i1);
	Darknet::free_image(i2);
}


TEST(Image, ResizeTiming)
{
	const size_t count = 100;
	Darknet::Image i1 = Darknet::load_image(IMG_FN);

	std::chrono::high_resolution_clock::duration duration = std::chrono::milliseconds(0);
	for (size_t idx = 0; idx < count; idx ++)
	{
		const auto t1 = std::chrono::high_resolution_clock::now();
		Darknet::Image i2 = Darknet::resize_image(i1, IMG_W / 2, IMG_H / 2);
		const auto t2 = std::chrono::high_resolution_clock::now();

		duration += (t2 - t1);

		Darknet::free_image(i2);
	}

	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	std::cout << "Resize using Darknet::resize_image() " << count << " times took " << milliseconds << " milliseconds" << std::endl;

	Darknet::free_image(i1);

	// now try the same thing using OpenCV's resize

	cv::Mat m1 = cv::imread(IMG_FN);
	cv::Mat m2;
	duration = std::chrono::milliseconds(0);
	for (size_t idx = 0; idx < count ; idx ++)
	{
		const auto t1 = std::chrono::high_resolution_clock::now();
		cv::resize(m1, m2, {0, 0}, 0.5, 0.5);
		const auto t2 = std::chrono::high_resolution_clock::now();

		duration += (t2 - t1);
	}

	milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
	std::cout << "Resize using OpenCV's cv::resize() " << count << " times took " << milliseconds << " milliseconds" << std::endl;
}
