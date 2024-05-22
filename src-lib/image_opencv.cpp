#ifdef __GNUC__
// 2023-06-25:  hide some of the warnings which for now we need to ignore in this file
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

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


#ifdef DEBUG
#define OCV_D "d"
#else
#define OCV_D
#endif//DEBUG

#include "http_stream.hpp"

#ifndef CV_RGB
#define CV_RGB(r, g, b) cvScalar( (b), (g), (r), 0 )
#endif

#ifndef CV_FILLED
#define CV_FILLED cv::FILLED
#endif

#ifndef CV_AA
#define CV_AA cv::LINE_AA
#endif

extern "C" {

mat_cv * load_image_mat_cv(const char * const filename, int channels)
{
	TAT(TATPARMS);

	cv::Mat * mat = nullptr;

	try
	{
		int flag = cv::IMREAD_UNCHANGED;

		if (channels == 1)
		{
			flag = cv::IMREAD_GRAYSCALE;
		}
		else if (channels == 3)
		{
			flag = cv::IMREAD_COLOR;
		}
		else
		{
			darknet_fatal_error(DARKNET_LOC, "OpenCV cannot load an image with %d channels: %s", channels, filename);
		}

		cv::Mat input = cv::imread(filename, flag);
		if (input.empty())
		{
			darknet_fatal_error(DARKNET_LOC, "failed to load image file \"%s\"", filename);
		}

		mat = new cv::Mat();

		if (input.channels() == 3)
		{
			// This is misleading.  The cv::imread() function returns the image in BGR format.
			// So this swaps the first and last channel, so we'll end up with RGB.
			cv::cvtColor(input, *mat, cv::COLOR_RGB2BGR);
		}
		else if (input.channels() == 4)
		{
			// Same comment -- image was BGR, but will now be RGB
			cv::cvtColor(input, *mat, cv::COLOR_RGBA2BGRA);
		}
		else
		{
			*mat = input;
		}
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while loading image %s: %s", filename, e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while loading image %s", filename);
	}

	return reinterpret_cast<mat_cv*>(mat);
}


image load_image_cv(char *filename, int channels)
{
	TAT(TATPARMS);

	cv::Mat* mat_ptr = reinterpret_cast<cv::Mat*>(load_image_mat_cv(filename, channels));
	cv::Mat mat = *mat_ptr;
	delete mat_ptr;

	return mat_to_image(mat);
}


int get_width_mat(mat_cv *mat)
{
	TAT(TATPARMS);

	if (mat == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "cannot get width from an invalid NULL image pointer");
	}

	return reinterpret_cast<cv::Mat*>(mat)->cols;
}


int get_height_mat(mat_cv *mat)
{
	TAT(TATPARMS);

	if (mat == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "cannot get height from an invalid NULL image pointer");
	}

	return reinterpret_cast<cv::Mat*>(mat)->rows;
}


void release_mat(mat_cv **mat)
{
	TAT(TATPARMS);

	cv::Mat ** ptr = reinterpret_cast<cv::Mat**>(mat);
	if (ptr && *ptr)
	{
		delete *ptr;
		*ptr = nullptr;
	}
}


cv::Mat image_to_mat(image img)
{
	TAT(TATPARMS);

	int channels = img.c;
	int width = img.w;
	int height = img.h;
	cv::Mat mat(height, width, CV_8UC(channels));
	int step = mat.step;

	for (int y = 0; y < img.h; ++y)
	{
		for (int x = 0; x < img.w; ++x)
		{
			for (int c = 0; c < img.c; ++c)
			{
				float val = img.data[c*img.h*img.w + y*img.w + x];
				mat.data[y*step + x*img.c + c] = (unsigned char)(val * 255);	 ///< @todo Is this right?
			}
		}
	}

	return mat;
}
// ----------------------------------------

image mat_to_image(cv::Mat mat)
{
	TAT(TATPARMS);

	int w = mat.cols;
	int h = mat.rows;
	int c = mat.channels();
	image im = make_image(w, h, c);
	unsigned char *data = (unsigned char *)mat.data;
	int step = mat.step;
	for (int y = 0; y < h; ++y) {
		for (int k = 0; k < c; ++k) {
			for (int x = 0; x < w; ++x) {
				im.data[k*w*h + y*w + x] = data[y*step + x*c + k] / 255.0f;
			}
		}
	}
	return im;
}

image mat_to_image_cv(mat_cv *mat)
{
	TAT(TATPARMS);

	return mat_to_image(*reinterpret_cast<cv::Mat*>(mat));
}

// ====================================================================
// Window
// ====================================================================
void create_window_cv(char const* window_name, int full_screen, int width, int height)
{
	TAT(TATPARMS);

	try
	{
		int window_type = cv::WINDOW_NORMAL;
#ifdef CV_VERSION_EPOCH // OpenCV 2.x
		if (full_screen) window_type = CV_WINDOW_FULLSCREEN;
#else
		if (full_screen) window_type = cv::WINDOW_FULLSCREEN;
#endif
		cv::namedWindow(window_name, window_type);
		cv::moveWindow(window_name, 0, 0);
		cv::resizeWindow(window_name, width, height);
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while creating, moving, or resizing the window \"%s\": %s", window_name, e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while creating, moving, or resizing the window \"%s\"", window_name);
	}
}


void resize_window_cv(char const* window_name, int width, int height)
{
	TAT(TATPARMS);

	cv::resizeWindow(window_name, width, height);
}


void move_window_cv(char const* window_name, int x, int y)
{
	TAT(TATPARMS);

	cv::moveWindow(window_name, x, y);
}


void destroy_all_windows_cv()
{
	TAT(TATPARMS);

	cv::destroyAllWindows();
}


int wait_key_cv(int delay)
{
	TAT(TATPARMS);

	return cv::waitKey(delay);
}


int wait_until_press_key_cv()
{
	TAT(TATPARMS);

	return wait_key_cv(0);
}


static float get_pixel(image m, int x, int y, int c)
{
	TAT(TATPARMS);

	assert(x < m.w && y < m.h && c < m.c);
	return m.data[c*m.h*m.w + y*m.w + x];
}
// ----------------------------------------

void show_image_cv(image p, const char *name)
{
	TAT(TATPARMS);

	try
	{
		image copy = copy_image(p);
		constrain_image(copy);

		cv::Mat mat = image_to_mat(copy);
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
		free_image(copy);
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


void show_image_mat(mat_cv *mat_ptr, const char *name)
{
	TAT(TATPARMS);

	try
	{
		if (mat_ptr == NULL)
		{
			return;
		}

		cv::Mat & mat = *reinterpret_cast<cv::Mat*>(mat_ptr);
		cv::namedWindow(name, cv::WINDOW_NORMAL);
		cv::imshow(name, mat);
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
// Video Writer
// ====================================================================


write_cv *create_video_writer(char *out_filename, char c1, char c2, char c3, char c4, int fps, int width, int height, int is_color)
{
	TAT(TATPARMS);

	try
	{
		cv::VideoWriter * output_video_writer =
#ifdef CV_VERSION_EPOCH
			new cv::VideoWriter(out_filename, CV_FOURCC(c1, c2, c3, c4), fps, cv::Size(width, height), is_color);
#else
			new cv::VideoWriter(out_filename, cv::VideoWriter::fourcc(c1, c2, c3, c4), fps, cv::Size(width, height), is_color);
#endif

		return reinterpret_cast<write_cv*>(output_video_writer);
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while creating the video writer (%d fps, %dx%d): %s", fps, width, height, e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while creating the video writer (%d fps, %dx%d)", fps, width, height);
	}

	return nullptr;
}

void write_frame_cv(write_cv *output_video_writer, mat_cv *mat)
{
	TAT(TATPARMS);

	try
	{
		cv::VideoWriter * vw = reinterpret_cast<cv::VideoWriter*>(output_video_writer);
		cv::Mat * ptr = reinterpret_cast<cv::Mat*>(mat);

		vw->write(*ptr);
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while writing video frame: %s", e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while writing video frame");
	}
}


void release_video_writer(write_cv **output_video_writer)
{
	TAT(TATPARMS);

	try
	{
		if (output_video_writer)
		{
			cv::VideoWriter * vw = reinterpret_cast<cv::VideoWriter*>(*output_video_writer);
			if (vw)
			{
				vw->release();
				delete vw;
			}

			*output_video_writer = nullptr;
		}
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while closing or freeing the video writer: %s", e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while closing or freeing the video writer");
	}
}


// ====================================================================
// Video Capture
// ====================================================================


cap_cv* get_capture_video_stream(const char *path)
{
	TAT(TATPARMS);

	cv::VideoCapture * cap = nullptr;
	try
	{
		cap = new cv::VideoCapture(path);
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while creating video capture for %s: %s", path, e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while creating video capture for %s", path);
	}

	return reinterpret_cast<cap_cv*>(cap);
}


cap_cv* get_capture_webcam(int index)
{
	TAT(TATPARMS);

	cv::VideoCapture* cap = nullptr;
	try
	{
		cap = new cv::VideoCapture(index);
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while creating video capture for device index %d: %s", index, e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while creating video capture for device index %d", index);
	}

	return reinterpret_cast<cap_cv*>(cap);
}


void release_capture(cap_cv * cap)
{
	TAT(TATPARMS);

	try
	{
		cv::VideoCapture * ptr = reinterpret_cast<cv::VideoCapture*>(cap);
		delete ptr;
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while deleting video capture: %s", e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while deleting video capture");
	}
}


mat_cv* get_capture_frame_cv(cap_cv * ptr)
{
	TAT(TATPARMS);

	cv::Mat * mat = nullptr;
	try
	{
		mat = new cv::Mat();

		if (ptr)
		{
			cv::VideoCapture & cap = *reinterpret_cast<cv::VideoCapture*>(ptr);

			if (cap.isOpened())
			{
				cap >> *mat;
			}
			else
			{
				throw std::runtime_error("Video stream has stopped.");
			}
		}
		else
		{
			throw std::runtime_error("Video stream does not exist.");
		}
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while reading frame from video stream: %s", e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while reading frame from video stream");
	}

	return reinterpret_cast<mat_cv*>(mat);
}


int get_stream_fps_cpp_cv(cap_cv * ptr)
{
	/// @todo why is this returning an int?

	TAT(TATPARMS);

	int fps = 25;
	try
	{
		if (ptr)
		{
			cv::VideoCapture & cap = *reinterpret_cast<cv::VideoCapture*>(ptr);
#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
			fps = std::round(cap.get(cv::CAP_PROP_FPS));
#else                        // OpenCV 2.x
			fps = std::round(cap.get(CV_CAP_PROP_FPS));
#endif
		}
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while reading FPS from video stream: %s", e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while reading FPS from video stream");
	}

	return fps;
}


double get_capture_property_cv(cap_cv * ptr, int property_id)
{
	TAT(TATPARMS);

	double result = 0.0;

	try
	{
		if (ptr)
		{
			cv::VideoCapture & cap = *reinterpret_cast<cv::VideoCapture*>(ptr);
			result = cap.get(property_id);
		}
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while reading property %d from video stream: %s", property_id, e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while reading property %d from video stream", property_id);
	}

	return result;
}


double get_capture_frame_count_cv(cap_cv *cap)
{
	TAT(TATPARMS);

#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
		return get_capture_property_cv(cap, cv::CAP_PROP_FRAME_COUNT);
#else                        // OpenCV 2.x
		return get_capture_property_cv(cap, CV_CAP_PROP_FRAME_COUNT);
#endif
}


int set_capture_property_cv(cap_cv * ptr, int property_id, double value)
{
	/// @todo why does this return @p int?

	TAT(TATPARMS);

	try
	{
		if (ptr)
		{
			cv::VideoCapture & cap = *reinterpret_cast<cv::VideoCapture*>(ptr);
			return cap.set(property_id, value);
		}
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while setting property %d to \"%d\" for video stream: %s", property_id, value, e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while setting property %d to \"%d\" for video stream", property_id, value);
	}

	return false;
}


int set_capture_position_frame_cv(cap_cv *cap, int index)
{
	TAT(TATPARMS);

#ifndef CV_VERSION_EPOCH    // OpenCV 3.x
		return set_capture_property_cv(cap, cv::CAP_PROP_POS_FRAMES, index);
#else                        // OpenCV 2.x
		return set_capture_property_cv(cap, CV_CAP_PROP_POS_FRAMES, index);
#endif
}


// ====================================================================
// ... Video Capture
// ====================================================================


image get_image_from_stream_cpp(cap_cv *cap)
{
	TAT(TATPARMS);

	cv::Mat *src = NULL;
	static int once = 1;
	if (once) {
		once = 0;
		do {
			if (src) delete src;
			src = (cv::Mat*)get_capture_frame_cv(cap);
			if (!src) return make_empty_image(0, 0, 0);
		} while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
		printf("Video stream: %d x %d \n", src->cols, src->rows);
	}
	else
		src = (cv::Mat*)get_capture_frame_cv(cap);

	if (!src) return make_empty_image(0, 0, 0);
	image im = mat_to_image(*src);
	rgbgr_image(im);
	if (src) delete src;
	return im;
}


int wait_for_stream(cap_cv *cap, cv::Mat* src, int dont_close)
{
	TAT(TATPARMS);

	if (!src) {
		if (dont_close) src = new cv::Mat(416, 416, CV_8UC(3)); // cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);  ///< @todo #COLOR
		else return 0;
	}
	if (src->cols < 1 || src->rows < 1 || src->channels() < 1) {
		if (dont_close) {
			delete src;// cvReleaseImage(&src);
			int z = 0;
			for (z = 0; z < 20; ++z) {
				src = (cv::Mat*)get_capture_frame_cv(cap);
				delete src;// cvReleaseImage(&src);
			}
			src = new cv::Mat(416, 416, CV_8UC(3)); // cvCreateImage(cvSize(416, 416), IPL_DEPTH_8U, 3);
		}
		else return 0;
	}
	return 1;
}


image get_image_from_stream_resize(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close)
{
	TAT(TATPARMS);

	c = c ? c : 3;
	cv::Mat *src = NULL;

	static int once = 1;
	if (once) {
		once = 0;
		do {
			if (src) delete src;
			src = (cv::Mat*)get_capture_frame_cv(cap);
			if (!src) return make_empty_image(0, 0, 0);
		} while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
		printf("Video stream: %d x %d \n", src->cols, src->rows);
	}
	else
		src = (cv::Mat*)get_capture_frame_cv(cap);

	if (!wait_for_stream(cap, src, dont_close)) return make_empty_image(0, 0, 0);

	*(cv::Mat **)in_img = src;

	cv::Mat new_img = cv::Mat(h, w, CV_8UC(c));
	cv::resize(*src, new_img, new_img.size(), 0, 0, cv::INTER_LINEAR);
	if (c>1) cv::cvtColor(new_img, new_img, cv::COLOR_RGB2BGR);
	image im = mat_to_image(new_img);

	//show_image_cv(im, "im");
	//show_image_mat(*in_img, "in_img");
	return im;
}


image get_image_from_stream_letterbox(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close)
{
	TAT(TATPARMS);

	c = c ? c : 3;
	cv::Mat *src = NULL;
	static int once = 1;
	if (once) {
		once = 0;
		do {
			if (src) delete src;
			src = (cv::Mat*)get_capture_frame_cv(cap);
			if (!src) return make_empty_image(0, 0, 0);
		} while (src->cols < 1 || src->rows < 1 || src->channels() < 1);
		printf("Video stream: %d x %d \n", src->cols, src->rows);
	}
	else
		src = (cv::Mat*)get_capture_frame_cv(cap);

	if (!wait_for_stream(cap, src, dont_close)) return make_empty_image(0, 0, 0);   // passes (cv::Mat *)src while should be (cv::Mat **)src

	*in_img = (mat_cv *)new cv::Mat(src->rows, src->cols, CV_8UC(c));
	cv::resize(*src, **(cv::Mat**)in_img, (*(cv::Mat**)in_img)->size(), 0, 0, cv::INTER_LINEAR);

	if (c>1) cv::cvtColor(*src, *src, cv::COLOR_RGB2BGR);
	image tmp = mat_to_image(*src);
	image im = letterbox_image(tmp, w, h);
	free_image(tmp);
	release_mat((mat_cv **)&src);

	//show_image_cv(im, "im");
	//show_image_mat(*in_img, "in_img");
	return im;
}


void consume_frame(cap_cv *cap)
{
	TAT(TATPARMS);

	cv::Mat *src = NULL;
	src = (cv::Mat *)get_capture_frame_cv(cap);
	if (src)
	{
		delete src;
	}
}


// ====================================================================
// Image Saving
// ====================================================================


void save_mat_png(cv::Mat mat, const char *name)
{
	/// @todo merge with @ref save_image_png()

	TAT(TATPARMS);

	const bool success = cv::imwrite(name, mat, {cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION, 9});
	if (not success)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to save the image %s", name);
	}
}


void save_mat_jpg(cv::Mat mat, const char *name)
{
	/// @todo merge with @ref save_image_jpg()

	TAT(TATPARMS);

	const bool success = cv::imwrite(name, mat, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 75});
	if (not success)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to save the image %s", name);
	}
}


void save_cv_png(mat_cv *img_src, const char *name)
{
	TAT(TATPARMS);

	cv::Mat* img = (cv::Mat* )img_src;
	save_mat_png(*img, name);
}


void save_cv_jpg(mat_cv *img_src, const char *name)
{
	TAT(TATPARMS);

	cv::Mat* img = (cv::Mat*)img_src;
	save_mat_jpg(*img, name);
}


// ====================================================================
// Draw Detection
// ====================================================================


void draw_detections_cv_v3(mat_cv* mat, detection *dets, int num, float thresh, char **names, int classes, int ext_output)
{
	TAT(TATPARMS);

	try
	{
		cv::Mat *show_img = (cv::Mat*)mat;
		int i, j;
		if (!show_img) return;
		static int frame_id = 0;
		frame_id++;

		for (i = 0; i < num; ++i) {
			char labelstr[4096] = { 0 };
			int class_id = -1;
			for (j = 0; j < classes; ++j) {
				int show = strncmp(names[j], "dont_show", 9);
				if (dets[i].prob[j] > thresh && show) {
					if (class_id < 0) {
						strcat(labelstr, names[j]);
						class_id = j;
						char buff[20];
						if (dets[i].track_id) {
							sprintf(buff, " (id: %d)", dets[i].track_id);
							strcat(labelstr, buff);
						}
						sprintf(buff, " (%2.0f%%)", dets[i].prob[j] * 100);
						strcat(labelstr, buff);
						printf("%s: %.0f%% ", names[j], dets[i].prob[j] * 100);
						if (dets[i].track_id) printf("(track = %d, sim = %f) ", dets[i].track_id, dets[i].sim);
					}
					else {
						strcat(labelstr, ", ");
						strcat(labelstr, names[j]);
						printf(", %s: %.0f%% ", names[j], dets[i].prob[j] * 100);
					}
				}
			}
			if (class_id >= 0)
			{
				int width = std::max(1.0f, show_img->rows * .002f);

				//printf("%d %s: %.0f%%\n", i, names[class_id], prob*100);
				int offset = class_id * 123457 % classes;
				float red = get_color(2, offset, classes);
				float green = get_color(1, offset, classes);
				float blue = get_color(0, offset, classes);
				//float rgb[3];

				//width = prob*20+2;

				//rgb[0] = red;
				//rgb[1] = green;
				//rgb[2] = blue;
				box b = dets[i].bbox;
				if (std::isnan(b.w) || std::isinf(b.w)) b.w = 0.5;
				if (std::isnan(b.h) || std::isinf(b.h)) b.h = 0.5;
				if (std::isnan(b.x) || std::isinf(b.x)) b.x = 0.5;
				if (std::isnan(b.y) || std::isinf(b.y)) b.y = 0.5;
				b.w = (b.w < 1) ? b.w : 1;
				b.h = (b.h < 1) ? b.h : 1;
				b.x = (b.x < 1) ? b.x : 1;
				b.y = (b.y < 1) ? b.y : 1;
				//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

				int left = (b.x - b.w / 2.)*show_img->cols;
				int right = (b.x + b.w / 2.)*show_img->cols;
				int top = (b.y - b.h / 2.)*show_img->rows;
				int bot = (b.y + b.h / 2.)*show_img->rows;

				if (left < 0) left = 0;
				if (right > show_img->cols - 1) right = show_img->cols - 1;
				if (top < 0) top = 0;
				if (bot > show_img->rows - 1) bot = show_img->rows - 1;

				//int b_x_center = (left + right) / 2;
				//int b_y_center = (top + bot) / 2;
				//int b_width = right - left;
				//int b_height = bot - top;
				//sprintf(labelstr, "%d x %d - w: %d, h: %d", b_x_center, b_y_center, b_width, b_height);

				float const font_size = show_img->rows / 1000.F;
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

				// you should create directory: result_img
				//static int copied_frame_id = -1;
				//static IplImage* copy_img = NULL;
				//if (copied_frame_id != frame_id) {
				//    copied_frame_id = frame_id;
				//    if(copy_img == NULL) copy_img = cvCreateImage(cvSize(show_img->width, show_img->height), show_img->depth, show_img->nChannels);
				//    cvCopy(show_img, copy_img, 0);
				//}
				//static int img_id = 0;
				//img_id++;
				//char image_name[1024];
				//sprintf(image_name, "result_img/img_%d_%d_%d_%s.jpg", frame_id, img_id, class_id, names[class_id]);
				//CvRect rect = cvRect(pt1.x, pt1.y, pt2.x - pt1.x, pt2.y - pt1.y);
				//cvSetImageROI(copy_img, rect);
				//cvSaveImage(image_name, copy_img, 0);
				//cvResetImageROI(copy_img);

				cv::rectangle(*show_img, pt1, pt2, color, width);
				if (ext_output)
					printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
					(float)left, (float)top, b.w*show_img->cols, b.h*show_img->rows);
				else
					printf("\n");

				cv::rectangle(*show_img, pt_text_bg1, pt_text_bg2, color, width);
				cv::rectangle(*show_img, pt_text_bg1, pt_text_bg2, color, CV_FILLED);    // filled
				cv::Scalar black_color = CV_RGB(0, 0, 0);
				cv::putText(*show_img, labelstr, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, black_color, 2 * font_size, CV_AA);
				// cv::FONT_HERSHEY_COMPLEX_SMALL, cv::FONT_HERSHEY_SIMPLEX
			}
		}
		if (ext_output) {
			fflush(stdout);
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
// Draw Loss & Accuracy chart
// ====================================================================


mat_cv* draw_initial_train_chart(char *windows_name, float max_img_loss, int max_batches, int number_of_lines, int img_size, int dont_show, char* chart_path)
{
	TAT(TATPARMS);

	// the room that we need to reserve on the left and the bottom of the image to draw the axis and various other information
	const int img_offset = 60;

	// the size of the "grid" drawing area, which should be 1000 - 60 = 940x940
	const int draw_size = img_size - img_offset;

	cv::Mat * img_ptr = new cv::Mat(img_size, img_size, CV_8UC3, CV_RGB(255, 255, 255));
	cv::Mat & img = *img_ptr;

	cv::Point pt1;
	cv::Point pt2;
	cv::Point pt_text;
	char char_buff[400];

	try
	{
		// If we have the previous chart.png file, we should use the grid from that.  Not sure what chart_path is supposed to be for,
		// but from what I can tell that is normally NULL.
		//
		// The filename "chart.png" -- without a path -- is hard-coded as the filename where the chart gets
		// updated in update_train_loss_chart(), so try that path and see if the image can be loaded.

		std::string filename = "chart.png";
		if (chart_path != nullptr && chart_path[0] != '\0')
		{
			filename = chart_path;
		}

		if (std::filesystem::exists(filename))
		{
			cv::Mat tmp = cv::imread(filename);
			if (tmp.size() == cv::Size(img_size, img_size))
			{
				// the image was read, and it is the correct size!

				// convert the previous image to greyscale, and then back to 3-channel BGR
				cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
				cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);

				// lighten (fade) the old grid information
				cv::Mat lighter;
				const double percent = 0.25 * 255.0; // how much lighter to make the image (0.25 = 25%)
				tmp.convertTo(lighter, -1, 1, percent);

				// the grid starts at (60, 0) and extends for "draw_size" (940) pixels to the bottom right
				const cv::Rect r(img_offset, 0, draw_size, draw_size);
				lighter(r).copyTo(img(r));
			}
		}

		// draw new chart

		// vertical lines
		pt1.x = img_offset;
		pt2.x = img_size;
		pt_text.x = 30;
		for (int i = 1; i <= number_of_lines; ++i)
		{
			pt1.y = pt2.y = (float)i * draw_size / number_of_lines;
			cv::line(img, pt1, pt2, CV_RGB(224, 224, 224));
			if (i % 10 == 0)
			{
				sprintf(char_buff, "%2.1f", max_img_loss*(number_of_lines - i) / number_of_lines);
				pt_text.y = pt1.y + 3;

				cv::putText(img, char_buff, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
				cv::line(img, pt1, pt2, CV_RGB(128, 128, 128));
			}
		}
		// horizontal lines
		pt1.y = draw_size; pt2.y = 0, pt_text.y = draw_size + 15;
		for (int i = 0; i <= number_of_lines; ++i)
		{
			pt1.x = pt2.x = img_offset + (float)i * draw_size / number_of_lines;
			cv::line(img, pt1, pt2, CV_RGB(224, 224, 224));
			if (i % 10 == 0)
			{
				sprintf(char_buff, "%d", max_batches * i / number_of_lines);
				pt_text.x = pt1.x - 20;
				cv::putText(img, char_buff, pt_text, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
				cv::line(img, pt1, pt2, CV_RGB(128, 128, 128));
			}
		}

		cv::putText(img, "Loss", cv::Point(10, 60), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 255), 1, CV_AA);
		cv::putText(img, "Iteration number", cv::Point(draw_size / 2, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, CV_AA);
		char max_batches_buff[100];
		sprintf(max_batches_buff, "max_batches = %d", max_batches);
		cv::putText(img, max_batches_buff, cv::Point(draw_size - 125, img_size - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, CV_RGB(128, 128, 128), 1, CV_AA);

		if (!dont_show)
		{
			printf("If an error occurs, add the flag -dont_show to supress the GUI window.\n");
			cv::namedWindow(windows_name, cv::WINDOW_NORMAL);
			cv::moveWindow(windows_name, 0, 0);
			cv::resizeWindow(windows_name, img_size, img_size);
			cv::imshow(windows_name, img);
			cv::waitKey(20);
		}
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while drawing the training chart: %s", e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while drawing the training chart");
	}

	return reinterpret_cast<mat_cv*>(img_ptr);
}


void update_train_loss_chart(char *windows_name, mat_cv* img_src, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches,
	float precision, int draw_precision, char *accuracy_name, float contr_acc, int dont_show, int mjpeg_port, double time_remaining)
{
	TAT(TATPARMS);

	/** @note This is called at @em every iteration to update the chart, but the output file isn't saved to disk every time.
	 *
	 * @note In some parts of the code, this is called at the @em end of the iteration, once the batch number (iteration)
	 * has already been updated.  So the very first time through, the @p "current_batch" is already set to @p 1, not @p 0.
	 * In othe parts, this is called at the start of the iteration, so the first time through the @p "current_batch" is zero.
	 */

	try
	{
		cv::Mat & img = *reinterpret_cast<cv::Mat*>(img_src);

		char char_buff[400];

		// Offset describes how much space (in pixels) is reserved to draw the axis details
		// at the left side and bottom side of the image.
		const int img_offset = 60;

		// 1000 - 60 gives us a grid area of 940x940
		const int draw_size = img_size - img_offset;

		// draw the blue "loss" point
		cv::Point pt1;
		pt1.x = img_offset + draw_size * (float)current_batch / max_batches;
		pt1.y = draw_size * (1 - avg_loss / max_img_loss);
		if (pt1.y < 0)
		{
			pt1.y = 1;
		}
		cv::circle(img, pt1, 1, CV_RGB(0, 0, 255));

		// contrastive accuracy
		if (contr_acc >= 0.0f)
		{
			static float old_contr_acc = 0.0f;

			if (current_batch > 0)
			{
				cv::line(img,
					cv::Point(img_offset + draw_size * (float)(current_batch - 1) / max_batches, draw_size * (1 - old_contr_acc)),
					cv::Point(img_offset + draw_size * (float)current_batch / max_batches, draw_size * (1 - contr_acc)),
					CV_RGB(0, 150, 70));
			}
			old_contr_acc = contr_acc;

			sprintf(char_buff, "C:%2.1f%% ", contr_acc * 100.0f);
			cv::putText(img, char_buff, cv::Point(1, 45), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, CV_AA);
			cv::putText(img, char_buff, cv::Point(1, 45), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 150, 70), 1, CV_AA);
		}

		// precision
		if (draw_precision)
		{
			static float old_precision = 0.0f;
			static float max_precision = 0.0f;
			static int iteration_old = 0;
			static int text_iteration_old = 0;

			if (iteration_old == 0)
			{
				/// @todo why is this not done above in @ref draw_initial_train_chart()
				// where "accuracy_name" is "mAP%"
				cv::putText(img, accuracy_name, cv::Point(10, 12), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 0, 0), 1, CV_AA);
			}

			if (iteration_old != 0)
			{
				cv::line(img,
					cv::Point(img_offset + draw_size * (float)iteration_old / max_batches, draw_size * (1 - old_precision)),
					cv::Point(img_offset + draw_size * (float)current_batch / max_batches, draw_size * (1 - precision)),
					CV_RGB(255, 0, 0));
			}

			sprintf(char_buff, "%2.1f%% ", precision * 100.0f);
			cv::putText(img, char_buff, cv::Point(10, 28), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, CV_AA);
			cv::putText(img, char_buff, cv::Point(10, 28), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, CV_AA);

			// decide if the mAP% value has changed enough that we need to re-label it on the chart
			if ((std::fabs(old_precision - precision) > 0.1) ||
				(max_precision < precision) ||
				(current_batch - text_iteration_old) >= max_batches / 10)
			{
				text_iteration_old = current_batch;
				max_precision = std::max(max_precision, precision);
				sprintf(char_buff, "%2.0f%% ", precision * 100.0f);
				cv::putText(img, char_buff, cv::Point(pt1.x - 30, draw_size * (1 - precision) + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, CV_AA);
				cv::putText(img, char_buff, cv::Point(pt1.x - 30, draw_size * (1 - precision) + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, CV_AA);
			}
			old_precision = precision;
			iteration_old = current_batch;
		}

		// draw the text at the bottom of the chart
		sprintf(char_buff, "loss = %2.4f      iteration = %d      time remaining = %s", avg_loss, current_batch, Darknet::format_time_remaining(60.0 * 60.0 * time_remaining).c_str());
		pt1.x = 15;
		pt1.y = draw_size + 18;
		cv::Point pt2;
		pt2.x = pt1.x + 850;
		pt2.y = pt1.y + 20;
		cv::rectangle(img, pt1, pt2, CV_RGB(255, 255, 255), CV_FILLED); // fill it with white to clear out the previous text
		pt1.y += 15;
		cv::putText(img, char_buff, pt1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 100), 1, CV_AA);

		// draw the timestamp in the lower left
		static std::time_t previous_time = 0;
		std::time_t current_time = std::time(nullptr);
		if (previous_time != current_time / 10 || current_batch >= (max_batches - 1))
		{
			std::tm * tm = std::localtime(&current_time);
			std::strftime(char_buff, sizeof(char_buff), "%Y-%m-%d %H:%M:%S %Z", tm);
			pt1.x = 5;
			pt1.y = img_size - 20;
			pt2.x = pt1.x + 250;
			pt2.y = pt1.y + 15;
			cv::rectangle(img, pt1, pt2, CV_RGB(255, 255, 255), CV_FILLED); // fill it with white to clear out the previous text
			pt1.x = 15;
			pt1.y = img_size - 10;
			cv::putText(img, char_buff, pt1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, CV_RGB(128, 128, 128), 1, CV_AA);
			previous_time = current_time / 10; // only update the timestamp every 10 seconds
		}

		int k = 0;
		if (!dont_show)
		{
			cv::imshow(windows_name, img);
			k = cv::waitKey(20);
		}
		static int old_batch = 0;
		if (k == 's' || current_batch <= 1 || current_batch >= (max_batches - 1) || (current_batch / 100 > old_batch / 100))
		{
			old_batch = current_batch;
			save_mat_png(img, "chart.png");
		}

		if (mjpeg_port > 0)
		{
			send_mjpeg((mat_cv *)&img, mjpeg_port, 500000, 70);
		}
	}
	catch (const std::exception & e)
	{
		darknet_fatal_error(DARKNET_LOC, "exception caught while drawing the training loss: %s", e.what());
	}
	catch (...)
	{
		darknet_fatal_error(DARKNET_LOC, "unknown exception while drawing the training loss");
	}
}


// ====================================================================
// Data augmentation
// ====================================================================

/// @todo #COLOR - cannot do hue in hyperspectal land
image image_data_augmentation(mat_cv* mat, int w, int h,
	int pleft, int ptop, int swidth, int sheight, int flip,
	float dhue, float dsat, float dexp,
	int gaussian_noise, int blur, int num_boxes, int truth_size, float *truth)
{
	TAT(TATPARMS);

	image out;
	try {
		cv::Mat img = *(cv::Mat *)mat;

		// crop
		cv::Rect src_rect(pleft, ptop, swidth, sheight);
		cv::Rect img_rect(cv::Point2i(0, 0), img.size());
		cv::Rect new_src_rect = src_rect & img_rect;

		cv::Rect dst_rect(cv::Point2i(std::max<int>(0, -pleft), std::max<int>(0, -ptop)), new_src_rect.size());
		cv::Mat sized;

		if (src_rect.x == 0 && src_rect.y == 0 && src_rect.size() == img.size()) {
			cv::resize(img, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
		}
		else {
			cv::Mat cropped(src_rect.size(), img.type());
			cropped.setTo(cv::mean(img));

			img(new_src_rect).copyTo(cropped(dst_rect));

			// resize
			cv::resize(cropped, sized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
		}

		// flip
		if (flip) {
			cv::Mat cropped;
			cv::flip(sized, cropped, 1);    // 0 - x-axis, 1 - y-axis, -1 - both axes (x & y)
			sized = cropped.clone();
		}

		// HSV augmentation
		// cv::COLOR_BGR2HSV, cv::COLOR_RGB2HSV, cv::COLOR_HSV2BGR, cv::COLOR_HSV2RGB
		if (dsat != 1 || dexp != 1 || dhue != 0) {
			if (img.channels() >= 3)	// This only (really) works for c == 3
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

		if (blur) {
			cv::Mat dst(sized.size(), sized.type());
			if (blur == 1) {
				cv::GaussianBlur(sized, dst, cv::Size(17, 17), 0);
				//cv::bilateralFilter(sized, dst, 17, 75, 75);
			}
			else {
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
			//std::cout << " blur num_boxes = " << num_boxes << std::endl;

			if (blur == 1) {
				cv::Rect img_rect(0, 0, sized.cols, sized.rows);
				int t;
				for (t = 0; t < num_boxes; ++t) {
					box b = float_to_box_stride(truth + t*truth_size, 1);
					if (!b.x) break;
					int left = (b.x - b.w / 2.)*sized.cols;
					int width = b.w*sized.cols;
					int top = (b.y - b.h / 2.)*sized.rows;
					int height = b.h*sized.rows;
					cv::Rect roi(left, top, width, height);
					roi = roi & img_rect;

					sized(roi).copyTo(dst(roi));
				}
			}
			dst.copyTo(sized);
		}

		if (gaussian_noise) {
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

		//char txt[100];
		//sprintf(txt, "blur = %d", blur);
		//cv::putText(sized, txt, cv::Point(100, 100), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.7, CV_RGB(255, 0, 0), 1, CV_AA);

		// Mat -> image
		out = mat_to_image(sized);
	}
	catch (const std::exception & e)
	{
//		out = mat_to_image(*(cv::Mat*)mat);
		darknet_fatal_error(DARKNET_LOC, "exception caught while augmenting image (%dx%d): %s", w, h, e.what());
	}
	catch (...)
	{
//		out = mat_to_image(*(cv::Mat*)mat);
		darknet_fatal_error(DARKNET_LOC, "unknown exception while augmenting image (%dx%d)", w, h);
	}

	return out;
}


// blend two images with (alpha and beta)
void blend_images_cv(image new_img, float alpha, image old_img, float beta)
{
	TAT(TATPARMS);

	cv::Mat new_mat(cv::Size(new_img.w, new_img.h), CV_32FC(new_img.c), new_img.data);// , size_t step = AUTO_STEP)
	cv::Mat old_mat(cv::Size(old_img.w, old_img.h), CV_32FC(old_img.c), old_img.data);
	cv::addWeighted(new_mat, alpha, old_mat, beta, 0.0, new_mat);
}

// bilateralFilter bluring
image blur_image(image src_img, int ksize)
{
	TAT(TATPARMS);

	cv::Mat src = image_to_mat(src_img);
	cv::Mat dst;
	cv::Size kernel_size = cv::Size(ksize, ksize);
	cv::GaussianBlur(src, dst, kernel_size, 0);
	//cv::bilateralFilter(src, dst, ksize, 75, 75);
	image dst_img = mat_to_image(dst);
	return dst_img;
}


// ====================================================================
// Draw object - adversarial attack dnn
// ====================================================================


std::atomic<int> x_start, y_start;
std::atomic<int> x_end, y_end;
std::atomic<int> x_size, y_size;
std::atomic<bool> draw_select, selected;

void callback_mouse_click(int event, int x, int y, int flags, void* user_data)
{
	TAT(TATPARMS);

	if (event == cv::EVENT_LBUTTONDOWN)
	{
		draw_select = true;
		selected = false;
		x_start = x;
		y_start = y;

		//if (prev_img_rect.contains(Point2i(x, y))) add_id_img = -1;
		//else if (next_img_rect.contains(Point2i(x, y))) add_id_img = 1;
		//else add_id_img = 0;
		//std::cout << "cv::EVENT_LBUTTONDOWN \n";
	}
	else if (event == cv::EVENT_LBUTTONUP)
	{
		x_size = abs(x - x_start);
		y_size = abs(y - y_start);
		x_end = std::max(x, 0);
		y_end = std::max(y, 0);
		draw_select = false;
		selected = true;
		//std::cout << "cv::EVENT_LBUTTONUP \n";
	}
	else if (event == cv::EVENT_MOUSEMOVE)
	{
		x_size = abs(x - x_start);
		y_size = abs(y - y_start);
		x_end = std::max(x, 0);
		y_end = std::max(y, 0);
	}
}


void cv_draw_object(image sized, float *truth_cpu, int max_boxes, int num_truth, int *it_num_set, float *lr_set, int *boxonly, int classes, char **names)
{
	TAT(TATPARMS);

	cv::Mat frame = image_to_mat(sized);
	if(frame.channels() == 3) cv::cvtColor(frame, frame, cv::COLOR_RGB2BGR);
	cv::Mat frame_clone = frame.clone();


	std::string const window_name = "Marking image";
	cv::namedWindow(window_name, cv::WINDOW_NORMAL);
	cv::resizeWindow(window_name, 1280, 720);
	cv::imshow(window_name, frame);
	cv::moveWindow(window_name, 0, 0);
	cv::setMouseCallback(window_name, callback_mouse_click);


	int it_trackbar_value = 200;
	std::string const it_trackbar_name = "iterations";
	/* int it_tb_res = */ cv::createTrackbar(it_trackbar_name, window_name, &it_trackbar_value, 1000);

	int lr_trackbar_value = 10;
	std::string const lr_trackbar_name = "learning_rate exp";
	/* int lr_tb_res = */ cv::createTrackbar(lr_trackbar_name, window_name, &lr_trackbar_value, 20);

	int cl_trackbar_value = 0;
	std::string const cl_trackbar_name = "class_id";
	/* int cl_tb_res = */ cv::createTrackbar(cl_trackbar_name, window_name, &cl_trackbar_value, classes-1);

	std::string const bo_trackbar_name = "box-only";
	/* int bo_tb_res = */ cv::createTrackbar(bo_trackbar_name, window_name, boxonly, 1);

	int i = 0;

	while (!selected) {
#ifndef CV_VERSION_EPOCH
		int pressed_key = cv::waitKeyEx(20);	// OpenCV 3.x
#else
		int pressed_key = cv::waitKey(20);		// OpenCV 2.x
#endif
		if (pressed_key == 27 || pressed_key == 1048603) break;// break;  // ESC - save & exit

		frame_clone = frame.clone();
		//char buff[100];
		std::string lr_value = "learning_rate = " + std::to_string(1.0 / pow(2, lr_trackbar_value));
		cv::putText(frame_clone, lr_value, cv::Point2i(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(10, 50, 10), 3);
		cv::putText(frame_clone, lr_value, cv::Point2i(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 120, 60), 2);
		cv::putText(frame_clone, lr_value, cv::Point2i(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 200, 100), 1);

		if (names) {
			std::string obj_name = names[cl_trackbar_value];
			cv::putText(frame_clone, obj_name, cv::Point2i(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(10, 50, 10), 3);
			cv::putText(frame_clone, obj_name, cv::Point2i(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(20, 120, 60), 2);
			cv::putText(frame_clone, obj_name, cv::Point2i(10, 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(50, 200, 100), 1);
		}

		if (draw_select) {
			cv::Rect selected_rect(
				cv::Point2i((int)min(x_start, x_end), (int)min(y_start, y_end)),
				cv::Size(x_size, y_size));

			rectangle(frame_clone, selected_rect, cv::Scalar(150, 200, 150));
		}


		cv::imshow(window_name, frame_clone);
	}

	if (selected) {
		cv::Rect selected_rect(
			cv::Point2i((int)min(x_start, x_end), (int)min(y_start, y_end)),
			cv::Size(x_size, y_size));

		printf(" x_start = %d, y_start = %d, x_size = %d, y_size = %d \n",
			x_start.load(), y_start.load(), x_size.load(), y_size.load());

		rectangle(frame, selected_rect, cv::Scalar(150, 200, 150));
		cv::imshow(window_name, frame);
		cv::waitKey(100);

		float width = x_end - x_start;
		float height = y_end - y_start;

		float const relative_center_x = (float)(x_start + width / 2) / frame.cols;
		float const relative_center_y = (float)(y_start + height / 2) / frame.rows;
		float const relative_width = (float)width / frame.cols;
		float const relative_height = (float)height / frame.rows;

		truth_cpu[i * 5 + 0] = relative_center_x;
		truth_cpu[i * 5 + 1] = relative_center_y;
		truth_cpu[i * 5 + 2] = relative_width;
		truth_cpu[i * 5 + 3] = relative_height;
		truth_cpu[i * 5 + 4] = cl_trackbar_value;
	}

	*it_num_set = it_trackbar_value;
	*lr_set = 1.0 / pow(2, lr_trackbar_value);
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
		//if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
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


void show_opencv_info()
{
	TAT(TATPARMS);

	std::cout
		<< "OpenCV "
		<< Darknet::in_colour(Darknet::EColour::kBrightWhite)
		<< "v" << CV_VERSION_MAJOR << "." << CV_VERSION_MINOR << "." << CVAUX_STR(CV_VERSION_REVISION) OCV_D
		<< Darknet::in_colour(Darknet::EColour::kNormal)
		<< std::endl;
}

}   // extern "C"
