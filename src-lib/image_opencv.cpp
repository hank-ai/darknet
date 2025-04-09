#include "darknet_internal.hpp"

// includes for OpenCV >= 3.x
#ifndef CV_VERSION_EPOCH
#include <opencv2/core/types.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#endif

// OpenCV includes for OpenCV 2.x
#ifdef CV_VERSION_EPOCH
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types_c.h>
#include <opencv2/core/version.hpp>
#endif


#ifndef CV_RGB
#define CV_RGB(r, g, b) cvScalar( (b), (g), (r) )
#endif

#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

#ifndef CV_AA
#define CV_AA cv::LINE_AA
#endif


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


cv::Mat load_rgb_mat_image(const char * const filename, int channels)
{
	TAT(TATPARMS);

	if (filename == nullptr or
		filename[0] == '\0')
	{
		darknet_fatal_error(DARKNET_LOC, "cannot load an image without a filename");
	}

	auto flag = cv::IMREAD_UNCHANGED;

	if (channels == 1)
	{
		flag = cv::IMREAD_GRAYSCALE;
	}
	else if (channels == 3)
	{
		flag = cv::IMREAD_COLOR;
	}
	else if (channels != 0)
	{
		darknet_fatal_error(DARKNET_LOC, "OpenCV cannot load an image with %d channels: %s", channels, filename);
	}

	cv::Mat mat = cv::imread(filename, flag);
	if (mat.empty())
	{
		darknet_fatal_error(DARKNET_LOC, "failed to load image file \"%s\"", filename);
	}

	if (mat.channels() == 3)
	{
		cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
	}
	else if (mat.channels() == 4 and channels == 3)
	{
		cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGB);
	}
	else if (mat.channels() == 4)
	{
		cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGBA);
	}

	return mat;
}


static float get_pixel(Darknet::Image m, int x, int y, int c)
{
	TAT(TATPARMS);

	assert(x < m.w && y < m.h && c < m.c);
	return m.data[c*m.h*m.w + y*m.w + x];
}


void show_image_cv(Darknet::Image p, const char *name)
{
	TAT(TATPARMS);

	try
	{
		Darknet::Image copy = Darknet::copy_image(p);
		Darknet::constrain_image(copy);

		cv::Mat mat = Darknet::image_to_mat(copy);
		if (mat.channels() == 3)
		{
			cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
		}
		else if (mat.channels() == 4)
		{
			cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
		}
		cv::namedWindow(name, cv::WINDOW_NORMAL);
		cv::imshow(name, mat);
		Darknet::free_image(copy);
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while showing an image: %s", e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while showing an image");
	}
}


// ====================================================================
// Image Saving
// ====================================================================


void save_mat_png(cv::Mat mat, const char *name)
{
	TAT(TATPARMS);

	/// @todo merge with @ref Darknet::save_image_png()

	const bool success = cv::imwrite(name, mat, {cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION, 9});
	if (not success)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to save the image %s", name);
	}
}


void save_mat_jpg(cv::Mat mat, const char *name)
{
	TAT(TATPARMS);

	/// @todo merge with @ref Darknet::save_image_jpg()

	const bool success = cv::imwrite(name, mat, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 75});
	if (not success)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to save the image %s", name);
	}
}


// ====================================================================
// Draw Detection
// ====================================================================


void draw_detections_cv_v3(cv::Mat mat, Darknet::Detection * dets, int num, float thresh, char **names, int classes, int ext_output)
{
	TAT(TATPARMS);

	try
	{
		for (int i = 0; i < num; ++i)
		{
			char labelstr[4096] = { 0 };
			int class_id = -1;
			for (int j = 0; j < classes; ++j)
			{
				int show = strncmp(names[j], "dont_show", 9);
				if (dets[i].prob[j] > thresh && show)
				{
					if (class_id < 0)
					{
						strcat(labelstr, names[j]);
						class_id = j;
						char buff[20];
						if (dets[i].track_id)
						{
							sprintf(buff, " (id: %d)", dets[i].track_id);
							strcat(labelstr, buff);
						}
						sprintf(buff, " (%2.0f%%)", dets[i].prob[j] * 100.0f);
						strcat(labelstr, buff);

						*cfg_and_state.output << names[j] << ": " << (int)std::round(100.0f * dets[i].prob[j]) << "%";

						if (dets[i].track_id)
						{
							*cfg_and_state.output << " (track=" << dets[i].track_id << ", sim=" << dets[i].sim << ")";
						}
					}
					else
					{
						strcat(labelstr, ", ");
						strcat(labelstr, names[j]);
						*cfg_and_state.output << names[j] << ": " << (int)std::round(100.0f * dets[i].prob[j]) << "%";
					}
				}
			}
			if (class_id >= 0)
			{
				int width = std::max(1.0f, mat.rows * 0.002f);

				int offset = class_id * 123457 % classes;
				float red	= Darknet::get_color(2, offset, classes);
				float green	= Darknet::get_color(1, offset, classes);
				float blue	= Darknet::get_color(0, offset, classes);

				Darknet::Box b = dets[i].bbox;
				if (std::isnan(b.w) || std::isinf(b.w)) b.w = 0.5f;
				if (std::isnan(b.h) || std::isinf(b.h)) b.h = 0.5f;
				if (std::isnan(b.x) || std::isinf(b.x)) b.x = 0.5f;
				if (std::isnan(b.y) || std::isinf(b.y)) b.y = 0.5f;
				b.w = (b.w < 1) ? b.w : 1.0f;
				b.h = (b.h < 1) ? b.h : 1.0f;
				b.x = (b.x < 1) ? b.x : 1.0f;
				b.y = (b.y < 1) ? b.y : 1.0f;

				int left	= std::round((b.x - b.w / 2.0f) * mat.cols);
				int right	= std::round((b.x + b.w / 2.0f) * mat.cols);
				int top		= std::round((b.y - b.h / 2.0f) * mat.rows);
				int bot		= std::round((b.y + b.h / 2.0f) * mat.rows);

				if (left < 0.0f) left = 0.0f;
				if (right > mat.cols - 1) right = mat.cols - 1;
				if (top < 0) top = 0;
				if (bot > mat.rows - 1) bot = mat.rows - 1;

				float const font_size = mat.rows / 1000.0f;
				cv::Size const text_size = cv::getTextSize(labelstr, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, 1, 0);
				cv::Point pt1, pt2, pt_text, pt_text_bg1, pt_text_bg2;
				pt1.x = left;
				pt1.y = top;
				pt2.x = right;
				pt2.y = bot;
				pt_text.x = left;
				pt_text.y = top - 4;// 12;
				pt_text_bg1.x = left;
				pt_text_bg1.y = top - (3 + 18 * font_size);
				pt_text_bg2.x = right;
				if ((right - left) < text_size.width) pt_text_bg2.x = left + text_size.width;
				pt_text_bg2.y = top;
				cv::Scalar color;
				color.val[0] = red * 256;
				color.val[1] = green * 256;
				color.val[2] = blue * 256;

				cv::rectangle(mat, pt1, pt2, color, width);

				*cfg_and_state.output
					<< "\tx="	<< left
					<< " y="	<< top
					<< " w="	<< (right - left)
					<< " h="	<< (bot - top)
					<< std::endl;

				cv::rectangle(mat, pt_text_bg1, pt_text_bg2, color, width);
				cv::rectangle(mat, pt_text_bg1, pt_text_bg2, color, CV_FILLED);    // filled
				cv::Scalar black_color = CV_RGB(0, 0, 0);
				cv::putText(mat, labelstr, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, black_color, 2 * font_size, CV_AA);
			}
		}
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while drawing detections: %s", e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while drawing detections");
	}
}


// ====================================================================
// Data augmentation
// ====================================================================


/// @todo COLOR - cannot do hue in hyperspectal land
Darknet::Image image_data_augmentation(cv::Mat mat, int w, int h,
	int pleft, int ptop, int swidth, int sheight, int flip,
	float dhue, float dsat, float dexp,
	int gaussian_noise, int blur, int num_boxes, int truth_size, float *truth)
{
	TAT(TATPARMS);

	Darknet::Image out;
	try
	{
		// crop
		cv::Rect src_rect(pleft, ptop, swidth, sheight);
		cv::Rect img_rect(cv::Point2i(0, 0), mat.size());
		cv::Rect new_src_rect = src_rect & img_rect;

		cv::Rect dst_rect(cv::Point2i(std::max<int>(0, -pleft), std::max<int>(0, -ptop)), new_src_rect.size());
		cv::Mat sized;

		if (src_rect.x == 0 && src_rect.y == 0 && src_rect.size() == mat.size())
		{
			cv::resize(mat, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
		}
		else
		{
			cv::Mat cropped(src_rect.size(), mat.type());
			cropped.setTo(cv::mean(mat));

			mat(new_src_rect).copyTo(cropped(dst_rect));

			// resize
			cv::resize(cropped, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
		}

		// flip
		if (flip)
		{
			cv::Mat cropped;
			cv::flip(sized, cropped, 1);    // 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
			sized = cropped.clone();
		}

		// HSV augmentation
		// cv::COLOR_BGR2HSV, cv::COLOR_RGB2HSV, cv::COLOR_HSV2BGR, cv::COLOR_HSV2RGB
		if (dsat != 1 || dexp != 1 || dhue != 0)
		{
			if (mat.channels() >= 3)	// This only (really) works for c == 3
			{
				cv::Mat hsv_src;
				cvtColor(sized, hsv_src, cv::COLOR_RGB2HSV);    // RGB to HSV

				std::vector<cv::Mat> hsv;
				cv::split(hsv_src, hsv);

				hsv[1] *= dsat;
				hsv[2] *= dexp;
				hsv[0] += 179 * dhue;

				cv::merge(hsv, hsv_src);

				cvtColor(hsv_src, sized, cv::COLOR_HSV2RGB);    // HSV to RGB (the same as previous)
			}
			else
			{
				sized *= dexp;
			}
		}

		//std::stringstream window_name;
		//window_name << "augmentation - " << ipl;
		//cv::imshow(window_name.str(), sized);
		//cv::waitKey(0);

		if (blur)
		{
			cv::Mat dst(sized.size(), sized.type());
			if (blur == 1)
			{
				cv::GaussianBlur(sized, dst, cv::Size(17, 17), 0);
				//cv::bilateralFilter(sized, dst, 17, 75, 75);
			}
			else
			{
				int ksize = (blur / 2) * 2 + 1;
				cv::Size kernel_size = cv::Size(ksize, ksize);
				cv::GaussianBlur(sized, dst, kernel_size, 0);
				//cv::medianBlur(sized, dst, ksize);
				//cv::bilateralFilter(sized, dst, ksize, 75, 75);

				// sharpen
				//cv::Mat img_tmp;
				//cv::GaussianBlur(dst, img_tmp, cv::Size(), 3);
				//cv::addWeighted(dst, 1.5, img_tmp, -0.5, 0, img_tmp);
				//dst = img_tmp;
			}
			//*cfg_and_state.output << " blur num_boxes = " << num_boxes << std::endl;

			if (blur == 1)
			{
				cv::Rect r(0, 0, sized.cols, sized.rows);
				for (int t = 0; t < num_boxes; ++t)
				{
					Darknet::Box b = float_to_box_stride(truth + t*truth_size, 1);
					if (!b.x) break;
					int left = (b.x - b.w / 2.)*sized.cols;
					int width = b.w*sized.cols;
					int top = (b.y - b.h / 2.)*sized.rows;
					int height = b.h*sized.rows;
					cv::Rect roi(left, top, width, height);
					roi = roi & r;

					sized(roi).copyTo(dst(roi));
				}
			}
			dst.copyTo(sized);
		}

		if (gaussian_noise)
		{
			cv::Mat noise = cv::Mat(sized.size(), sized.type());
			gaussian_noise = std::min(gaussian_noise, 127);
			gaussian_noise = std::max(gaussian_noise, 0);
			cv::randn(noise, 0, gaussian_noise);  //mean and variance
			cv::Mat sized_norm = sized + noise;
			//cv::normalize(sized_norm, sized_norm, 0.0, 255.0, cv::NORM_MINMAX, sized.type());
			//cv::imshow("source", sized);
			//cv::imshow("gaussian noise", sized_norm);
			//cv::waitKey(0);
			sized = sized_norm;
		}

		// Mat -> image
		out = Darknet::mat_to_image(sized);
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while augmenting image (%dx%d): %s", w, h, e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while augmenting image (%dx%d)", w, h);
	}

	return out;
}


// blend two images with (alpha and beta)
void blend_images_cv(Darknet::Image new_img, float alpha, Darknet::Image old_img, float beta)
{
	TAT(TATPARMS);

	cv::Mat new_mat(cv::Size(new_img.w, new_img.h), CV_32FC(new_img.c), new_img.data);// , size_t step = AUTO_STEP)
	cv::Mat old_mat(cv::Size(old_img.w, old_img.h), CV_32FC(old_img.c), old_img.data);
	cv::addWeighted(new_mat, alpha, old_mat, beta, 0.0, new_mat);
}


// ====================================================================
// Show Anchors
// ====================================================================


void show_anchors(int number_of_boxes, int num_of_clusters, float *rel_width_height_array, model anchors_data, int width, int height)
{
	TAT(TATPARMS);

	cv::Mat labels = cv::Mat(number_of_boxes, 1, CV_32SC1);
	cv::Mat points = cv::Mat(number_of_boxes, 2, CV_32FC1);
	cv::Mat centers = cv::Mat(num_of_clusters, 2, CV_32FC1);

	for (int i = 0; i < number_of_boxes; ++i) {
		points.at<float>(i, 0) = rel_width_height_array[i * 2];
		points.at<float>(i, 1) = rel_width_height_array[i * 2 + 1];
	}

	for (int i = 0; i < num_of_clusters; ++i) {
		centers.at<float>(i, 0) = anchors_data.centers.vals[i][0];
		centers.at<float>(i, 1) = anchors_data.centers.vals[i][1];
	}

	for (int i = 0; i < number_of_boxes; ++i) {
		labels.at<int>(i, 0) = anchors_data.assignments[i];
	}

	size_t img_size = 700;
	cv::Mat img = cv::Mat(img_size, img_size, CV_8UC3);

	for (int i = 0; i < number_of_boxes; ++i) {
		cv::Point pt;
		pt.x = points.at<float>(i, 0) * img_size / width;
		pt.y = points.at<float>(i, 1) * img_size / height;
		int cluster_idx = labels.at<int>(i, 0);
		int red_id = (cluster_idx * (uint64_t)123 + 55) % 255;
		int green_id = (cluster_idx * (uint64_t)321 + 33) % 255;
		int blue_id = (cluster_idx * (uint64_t)11 + 99) % 255;
		cv::circle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED);
	}

	for (int j = 0; j < num_of_clusters; ++j) {
		cv::Point pt1, pt2;
		pt1.x = pt1.y = 0;
		pt2.x = centers.at<float>(j, 0) * img_size / width;
		pt2.y = centers.at<float>(j, 1) * img_size / height;
		cv::rectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1);
	}
	save_mat_png(img, "cloud.png");
	cv::imshow("clusters", img);
	cv::waitKey(0);
	cv::destroyAllWindows();
}
