#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();


	static inline float get_pixel(Darknet::Image m, int x, int y, int c)
	{
		TAT(TATPARMS);

		assert(x < m.w && y < m.h && c < m.c);

		return m.data[c*m.h*m.w + y*m.w + x];
	}

	static inline float get_pixel_extend(Darknet::Image m, int x, int y, int c)
	{
		TAT(TATPARMS);

		if (x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;

		if (c < 0 || c >= m.c) return 0;

		return get_pixel(m, x, y, c);
	}

	static inline void set_pixel(Darknet::Image m, int x, int y, int c, float val)
	{
		TAT(TATPARMS);

		if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
		assert(x < m.w && y < m.h && c < m.c);
		m.data[c*m.h*m.w + y*m.w + x] = val;
	}

	static inline void add_pixel(Darknet::Image m, int x, int y, int c, float val)
	{
		TAT(TATPARMS);

		assert(x < m.w && y < m.h && c < m.c);
		m.data[c*m.h*m.w + y*m.w + x] += val;
	}

	static inline float three_way_max(const float a, const float b, const float c)
	{
		TAT(TATPARMS);

		return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
	}

	static inline float three_way_min(const float a, const float b, const float c)
	{
		TAT(TATPARMS);

		return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
	}
}


float Darknet::get_color(int c, int x, int max)
{
	TAT(TATPARMS);

	const float colors[6][3] = {{1,0,1},{0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0}};

	float ratio = ((float)x/max)*5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];

	return r;
}


void Darknet::composite_image(const Darknet::Image & source, Darknet::Image & dest, int dx, int dy)
{
	TAT(TATPARMS);

	for (int k = 0; k < source.c; ++k)
	{
		for (int y = 0; y < source.h; ++y)
		{
			for (int x = 0; x < source.w; ++x)
			{
				float val = get_pixel(source, x, y, k);
				float val2 = get_pixel_extend(dest, dx+x, dy+y, k);
				set_pixel(dest, dx+x, dy+y, k, val * val2);
			}
		}
	}
}


Darknet::Image get_opencv_label(const std::string & str, const int area)
{
	TAT(TATPARMS);

	/// @todo what are the performance implications of LINE_AA over LINE_4 or LINE_8?

	const auto font_lines		= cv::LineTypes::LINE_AA;
	const auto font_face		= cv::HersheyFonts::FONT_HERSHEY_PLAIN;
	const auto font_thickness	= 1;
	auto font_scale				= 0.0;
	auto text_size				= cv::Size(0, 0);

	// try and find a decent font size to use based on the size of the object
	for (font_scale = 2.0; font_scale > 0.6; font_scale -= 0.1)
	{
		text_size = cv::getTextSize(str, font_face, font_scale, font_thickness, nullptr);
		if (text_size.area() < area / 4)
		{
			// looks like this might be a good font scale to use for this object
			break;
		}
		// otherwise, try a smaller font scale
	}

	cv::Mat mat(text_size.height + 8, text_size.width + 4, CV_8UC3, cv::Scalar(255, 255, 255));

	cv::putText(mat, str, {2, text_size.height + 3}, font_face, font_scale, {0, 0, 0}, font_thickness, font_lines);

	/// @todo 2025-04-23:  should this be rgb or bgr?
	return Darknet::rgb_mat_to_rgb_image(mat);
}


void Darknet::draw_weighted_label(Darknet::Image & a, int r, int c, Darknet::Image & label, const float *rgb, const float alpha)
{
	TAT(TATPARMS);

	const int w = label.w;
	const int h = label.h;

	if (r - h >= 0)
	{
		r = r - h;
	}

	for (int j = 0; j < h && j + r < a.h; ++j)
	{
		for (int i = 0; i < w && i + c < a.w; ++i)
		{
			for (int k = 0; k < label.c; ++k)
			{
				float val1 = get_pixel(label, i, j, k);
				float val2 = get_pixel(a, i + c, j + r, k);
				float val_dst = val1 * rgb[k] * alpha + val2 * (1 - alpha);
				set_pixel(a, i + c, j + r, k, val_dst);
			}
		}
	}

	return;
}


void Darknet::draw_box_bw(Darknet::Image & a, int x1, int y1, int x2, int y2, float brightness)
{
	TAT(TATPARMS);

	// check to make sure the coordinates are within the image
	x1 = std::clamp(x1, 0, a.w - 1);
	x2 = std::clamp(x2, 0, a.w - 1);
	y1 = std::clamp(y1, 0, a.h - 1);
	y2 = std::clamp(y2, 0, a.h - 1);

	// draw horizontal lines
	for (int i = x1; i <= x2; ++i)
	{
		a.data[i + y1*a.w + 0 * a.w*a.h] = brightness;
		a.data[i + y2*a.w + 0 * a.w*a.h] = brightness;
	}

	// draw vertical lines
	for (int i = y1; i <= y2; ++i)
	{
		a.data[x1 + i*a.w + 0 * a.w*a.h] = brightness;
		a.data[x2 + i*a.w + 0 * a.w*a.h] = brightness;
	}
}


void Darknet::draw_box_width_bw(Darknet::Image & a, int x1, int y1, int x2, int y2, int w, float brightness)
{
	TAT(TATPARMS);

	for (int i = 0; i < w; ++i)
	{
		float alternate_color = (w % 2) ? (brightness) : (1.0 - brightness);
		draw_box_bw(a, x1 + i, y1 + i, x2 - i, y2 - i, alternate_color);
	}
}


void Darknet::draw_box(Darknet::Image & a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
	TAT(TATPARMS);

	// check to make sure the coordinates are within the image
	x1 = std::clamp(x1, 0, a.w - 1);
	x2 = std::clamp(x2, 0, a.w - 1);
	y1 = std::clamp(y1, 0, a.h - 1);
	y2 = std::clamp(y2, 0, a.h - 1);

	// draw 2 perfectly horizontal lines from x1 to x2, (top + bottom)
	for (int i = x1; i <= x2; ++i)
	{
		a.data[i + y1 * a.w + 0 * a.w * a.h] = r;
		a.data[i + y2 * a.w + 0 * a.w * a.h] = r;

		a.data[i + y1 * a.w + 1 * a.w * a.h] = g;
		a.data[i + y2 * a.w + 1 * a.w * a.h] = g;

		a.data[i + y1 * a.w + 2 * a.w * a.h] = b;
		a.data[i + y2 * a.w + 2 * a.w * a.h] = b;
	}

	// draw 2 perfectly vertical lines from y1 to y2 (left + right)
	for (int i = y1; i <= y2; ++i)
	{
		a.data[x1 + i * a.w + 0 * a.w * a.h] = r;
		a.data[x2 + i * a.w + 0 * a.w * a.h] = r;

		a.data[x1 + i * a.w + 1 * a.w * a.h] = g;
		a.data[x2 + i * a.w + 1 * a.w * a.h] = g;

		a.data[x1 + i * a.w + 2 * a.w * a.h] = b;
		a.data[x2 + i * a.w + 2 * a.w * a.h] = b;
	}

	return;
}


void Darknet::draw_box_width(Darknet::Image & a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
	TAT(TATPARMS);

	for (int i = 0; i < w; ++i)
	{
		draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
	}

	return;
}


void Darknet::draw_bbox(Darknet::Image & a, const Darknet::Box & bbox, int w, float r, float g, float b)
{
	TAT(TATPARMS);

	const int left	= (bbox.x - bbox.w / 2) * a.w;
	const int right	= (bbox.x + bbox.w / 2) * a.w;
	const int top	= (bbox.y - bbox.h / 2) * a.h;
	const int bot	= (bbox.y + bbox.h / 2) * a.h;

	for (int i = 0; i < w; ++i)
	{
		draw_box(a, left + i, top + i, right - i, bot - i, r, g, b);
	}

	return;
}


// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* get_actual_detections(const Darknet::Detection *dets, int dets_num, float thresh, int* selected_detections_num, const Darknet::VStr & names)
{
	TAT(TATPARMS);

	int selected_num = 0;
	detection_with_class* result_arr = (detection_with_class*)xcalloc(dets_num, sizeof(detection_with_class));
	int i;
	for (i = 0; i < dets_num; ++i)
	{
		int best_class = -1;
		float best_class_prob = thresh;
		int j;
		for (j = 0; j < dets[i].classes; ++j)
		{
//			int show = strncmp(names[j], "dont_show", 9);
			bool show = (names[j].find("dont_show") != 0);

			if (dets[i].prob[j] > best_class_prob && show) {
				best_class = j;
				best_class_prob = dets[i].prob[j];
			}
		}
		if (best_class >= 0) {
			result_arr[selected_num].det = dets[i];
			result_arr[selected_num].best_class = best_class;
			++selected_num;
		}
	}
	if (selected_detections_num)
		*selected_detections_num = selected_num;
	return result_arr;
}


// compare to sort detection** by bbox.x
int compare_by_lefts(const void *a_ptr, const void *b_ptr)
{
	TAT(TATPARMS);

	const detection_with_class* a = (detection_with_class*)a_ptr;
	const detection_with_class* b = (detection_with_class*)b_ptr;
	const float delta = (a->det.bbox.x - a->det.bbox.w/2) - (b->det.bbox.x - b->det.bbox.w/2);
	return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}


// compare to sort detection** by best_class probability
int compare_by_probs(const void *a_ptr, const void *b_ptr)
{
	TAT(TATPARMS);

	const detection_with_class* a = (detection_with_class*)a_ptr;
	const detection_with_class* b = (detection_with_class*)b_ptr;
	float delta = a->det.prob[a->best_class] - b->det.prob[b->best_class];
	return delta < 0 ? -1 : delta > 0 ? 1 : 0;
}


void Darknet::draw_detections_v3(Darknet::Image & im, const Darknet::Detection * dets, const int num, const float thresh, const Darknet::VStr & names, const int classes, const int ext_output)
{
	TAT(TATPARMS);

	int selected_detections_num;
	detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num, names);

	// text output
	/// @todo replace qsort() mid priority
	qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);

	int i;
	for (i = 0; i < selected_detections_num; ++i)
	{
		const int best_class = selected_detections[i].best_class;
		*cfg_and_state.output
			<< names[best_class]
			<< "\tc=" << (selected_detections[i].det.prob[best_class] * 100.0f) << "%"
			<< "\tx=" << (int)std::round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2.0f) * im.w)
			<< "\ty=" << (int)std::round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2.0f) * im.h)
			<< "\tw=" << (int)std::round(selected_detections[i].det.bbox.w * im.w)
			<< "\th=" << (int)std::round(selected_detections[i].det.bbox.h * im.h)
			<< std::endl;

		// now that the "best" has been printed, see if there are other classes to print
		for (int j = 0; j < classes; ++j)
		{
			if (selected_detections[i].det.prob[j] < thresh or j == best_class)
			{
				continue;
			}

			*cfg_and_state.output
				<< names[j] << ": " << (selected_detections[i].det.prob[j] * 100.0f) << "%"
				<< "\tx="	<< (int)std::round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2.0f) * im.w)
				<< " y="	<< (int)std::round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2.0f) * im.h)
				<< " w="	<< (int)std::round(selected_detections[i].det.bbox.w * im.w)
				<< " h="	<< (int)std::round(selected_detections[i].det.bbox.h * im.h)
				<< std::endl;
		}
	}

	// image output
	/// @todo replace qsort() mid priority
	qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_probs);

	for (i = 0; i < selected_detections_num; ++i)
	{
		int width = im.h * .002;
		if (width < 1)
		{
			width = 1;
		}

		int offset = selected_detections[i].best_class * 123457 % classes;
		float red	= Darknet::get_color(2, offset, classes);
		float green	= Darknet::get_color(1, offset, classes);
		float blue	= Darknet::get_color(0, offset, classes);
		float rgb[3];

		rgb[0] = red;
		rgb[1] = green;
		rgb[2] = blue;
		Darknet::Box b = selected_detections[i].det.bbox;

		int left = (b.x - b.w / 2.)*im.w;
		int right = (b.x + b.w / 2.)*im.w;
		int top = (b.y - b.h / 2.)*im.h;
		int bot = (b.y + b.h / 2.)*im.h;

		if (left < 0) left = 0;
		if (right > im.w - 1) right = im.w - 1;
		if (top < 0) top = 0;
		if (bot > im.h - 1) bot = im.h - 1;

		if (im.c == 1)
		{
			draw_box_width_bw(im, left, top, right, bot, width, 0.8);    // 1 channel Black-White
		}
		else
		{
			Darknet::draw_box_width(im, left, top, right, bot, width, red, green, blue); // 3 channels RGB
		}

		const std::string class_name = names[selected_detections[i].best_class];
		const float confidence = selected_detections[i].det.prob[selected_detections[i].best_class];
		std::stringstream ss;
		ss << class_name << " " << std::fixed << std::setprecision(2) << confidence;

		// if there are multiple predictions for this object, include all the names
		for (int j = 0; j < classes; ++j)
		{
			if (selected_detections[i].det.prob[j] > thresh && j != selected_detections[i].best_class)
			{
				ss << ", " << names[j];
			}
		}

		Darknet::Image label = get_opencv_label(ss.str(), (right - left) * (bot - top));
		Darknet::draw_weighted_label(im, top + width, left, label, rgb, 0.7);
		Darknet::free_image(label);

		if (selected_detections[i].det.mask)
		{
			Darknet::Image mask = float_to_image(14, 14, 1, selected_detections[i].det.mask);
			Darknet::Image resized_mask = Darknet::resize_image(mask, b.w*im.w, b.h*im.h);
			Darknet::Image tmask = threshold_image(resized_mask, .5);
			embed_image(tmask, im, left, top);
			Darknet::free_image(mask);
			Darknet::free_image(resized_mask);
			Darknet::free_image(tmask);
		}
	}
	free(selected_detections);
}


void Darknet::rotate_image_cw(Darknet::Image & im, int times)
{
	TAT(TATPARMS);

	assert(im.w == im.h);
	times = (times + 400) % 4;

	int n = im.w;

	for (int i = 0; i < times; ++i)
	{
		for (int c = 0; c < im.c; ++c)
		{
			for (int x = 0; x < n/2; ++x)
			{
				for (int y = 0; y < (n-1)/2 + 1; ++y)
				{
					float temp = im.data[y + im.w*(x + im.h*c)];
					im.data[y + im.w*(x + im.h*c)] = im.data[n-1-x + im.w*(y + im.h*c)];
					im.data[n-1-x + im.w*(y + im.h*c)] = im.data[n-1-y + im.w*(n-1-x + im.h*c)];
					im.data[n-1-y + im.w*(n-1-x + im.h*c)] = im.data[x + im.w*(n-1-y + im.h*c)];
					im.data[x + im.w*(n-1-y + im.h*c)] = temp;
				}
			}
		}
	}

	return;
}


void Darknet::flip_image(Darknet::Image & a)
{
	TAT(TATPARMS);

	for (int k = 0; k < a.c; ++k)
	{
		for (int i = 0; i < a.h; ++i)
		{
			for (int j = 0; j < a.w/2; ++j)
			{
				const int index = j + a.w * (i + a.h * k);
				const int flip = (a.w - j - 1) + a.w * (i + a.h * k);
				std::swap(a.data[flip], a.data[index]);
			}
		}
	}
}


void Darknet::embed_image(const Darknet::Image & source, Darknet::Image & dest, int dx, int dy)
{
	TAT(TATPARMS);

	for (int k = 0; k < source.c; ++k)
	{
		for (int y = 0; y < source.h; ++y)
		{
			for (int x = 0; x < source.w; ++x)
			{
				float val = get_pixel(source, x, y, k);
				set_pixel(dest, dx + x, dy + y, k, val);
			}
		}
	}

	return;
}


Darknet::Image Darknet::collapse_image_layers(const Darknet::Image & source, int border)
{
	TAT(TATPARMS);

	int h = (source.h + border) * source.c - border;
	Darknet::Image dest = make_image(source.w, h, 1);

	for (int i = 0; i < source.c; ++i)
	{
		Darknet::Image layer = get_image_layer(source, i);
		int h_offset = i * (source.h + border);
		Darknet::embed_image(layer, dest, 0, h_offset);
		Darknet::free_image(layer);
	}

	return dest;
}


void Darknet::constrain_image(Darknet::Image & im)
{
	TAT(TATPARMS);

	for (int i = 0; i < im.w * im.h * im.c; ++i)
	{
		im.data[i] = std::clamp(im.data[i], 0.0f, 1.0f);
	}

	return;
}


void Darknet::normalize_image(Darknet::Image & p)
{
	TAT(TATPARMS);

	float min = 9999999;
	float max = -999999;

	for (int i = 0; i < p.h * p.w * p.c; ++i)
	{
		float v = p.data[i];
		if (v < min) min = v;
		if (v > max) max = v;
	}

	if (max - min < 0.000000001f)
	{
		min = 0.0f;
		max = 1.0f;
	}

	for (int i = 0; i < p.c*p.w*p.h; ++i)
	{
		p.data[i] = (p.data[i] - min)/(max-min);
	}

	return;
}


Darknet::Image Darknet::copy_image(const Darknet::Image & p)
{
	TAT(TATPARMS);

	Darknet::Image copy = p;
	copy.data = (float*)xcalloc(p.h * p.w * p.c, sizeof(float));
	memcpy(copy.data, p.data, p.h * p.w * p.c * sizeof(float));

	return copy;
}


void Darknet::rgbgr_image(Darknet::Image & im)
{
	TAT(TATPARMS);

	for (int i = 0; i < im.w*im.h; ++i)
	{
		std::swap(im.data[i], im.data[i + im.w * im.h * 2]);
	}
}


void Darknet::show_image(const Darknet::Image & p, const char * name)
{
	TAT(TATPARMS);

	show_image_cv(p, name);
}


void Darknet::save_image_png(const Darknet::Image & im, const char *name)
{
	TAT(TATPARMS);

	/// @todo merge with @ref save_mat_png()

	std::string filename = name;
	filename += ".png";

	cv::Mat mat = image_to_mat(im);

	if (mat.channels() == 3)
	{
		cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
	}
	else if (mat.channels() == 4)
	{
		cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGRA);
	}

	const bool success = cv::imwrite(filename, mat, {cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION, 9});
	if (not success)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to save the image %s", filename.c_str());
	}
}


void Darknet::save_image_jpg(const Darknet::Image & im, const char *name)
{
	TAT(TATPARMS);

	/// @todo merge with @ref save_mat_jpg()

	std::string filename = name;
	filename += ".jpg";

	cv::Mat mat = image_to_mat(im);

	if (mat.channels() == 3)
	{
		cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
	}
	else if (mat.channels() == 4)
	{
		cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
	}

	const bool success = cv::imwrite(filename, mat, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 75});
	if (not success)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to save the image %s", filename.c_str());
	}

	return;
}


void Darknet::save_image(const Darknet::Image & im, const char *name)
{
	TAT(TATPARMS);

	save_image_jpg(im, name);

	return;
}


void Darknet::show_image_layers(const Darknet::Image & p, const char * name)
{
	TAT(TATPARMS);

	int i;
	char buff[256];
	for (i = 0; i < p.c; ++i)
	{
		sprintf(buff, "%s - Layer %d", name, i);
		Darknet::Image layer = get_image_layer(p, i);
		Darknet::show_image(layer, buff);
		Darknet::free_image(layer);
	}
}


void Darknet::show_image_collapsed(const Darknet::Image & p, const char * name)
{
	TAT(TATPARMS);

	Darknet::Image c = Darknet::collapse_image_layers(p, 1);
	Darknet::show_image(c, name);
	Darknet::free_image(c);
}


DarknetImage make_empty_image(int w, int h, int c)
{
	TAT(TATPARMS);

	// this is part of the original C API

	Darknet::Image out;
	out.data = nullptr;
	out.h = h;
	out.w = w;
	out.c = c;

	return out;
}


DarknetImage make_image(int w, int h, int c)
{
	TAT(TATPARMS);

	// this is part of the original C API

	Darknet::Image out = make_empty_image(w,h,c);
	out.data = (float*)xcalloc(h * w * c, sizeof(float));

	return out;
}


Darknet::Image Darknet::mat_to_image(const cv::Mat & mat)
{
	TAT(TATPARMS);

	// This code assumes the mat object is already in RGB format, not OpenCV's default BGR!
	//
	// DO NOT CALL!  You should be using rgb_mat_to_rgb_image() instead.

	const int w = mat.cols;
	const int h = mat.rows;
	const int c = mat.channels();
	Darknet::Image im = make_image(w, h, c);
	const unsigned char * data = static_cast<unsigned char *>(mat.data);
	const int step = mat.step;
	for (int y = 0; y < h; ++y)
	{
		for (int k = 0; k < c; ++k)
		{
			for (int x = 0; x < w; ++x)
			{
				im.data[k * w * h + y * w + x] = data[y * step + x * c + k] / 255.0f;
			}
		}
	}

	return im;
}


Darknet::Image Darknet::rgb_mat_to_rgb_image(const cv::Mat & mat)
{
	TAT(TATPARMS);

	// This code assumes the mat object is in (non-standard) RGB format!

	/// @todo COLOR this function assumes 3-channel images

	// create 3 "views" into 1 large "single-channel" image, one each for B, G, and R
	cv::Mat result(mat.rows * 3, mat.cols, CV_8UC1);
	std::vector<cv::Mat> views =
	{
		result.rowRange(mat.rows * 0, mat.rows * 1),	// B
		result.rowRange(mat.rows * 1, mat.rows * 2),	// G
		result.rowRange(mat.rows * 2, mat.rows * 3),	// R
	};
	cv::split(mat, views);

	// create an empty Darknet::Image where the float results will be stored by OpenCV once we convert to CV_32F
	Darknet::Image img = make_image(mat.cols, mat.rows, mat.channels());

	// note how the cv::Matf is using the Darknet::Image data buffer directly
	cv::Mat1f tmp(img.h * 3, img.w, img.data);

	// convert the results to floating point, and divide by 255 to normalize between 0.0 - 1.0
	result.convertTo(tmp, CV_32F, 1.0/255.0);

	return img;
}


Darknet::Image Darknet::bgr_mat_to_rgb_image(const cv::Mat & mat)
{
	TAT(TATPARMS);

	// This code assumes the mat object is in OpenCV's default BGR format!

	/// @todo COLOR this function assumes 3-channel images

	// create 3 "views" into 1 large "single-channel" image, one each for B, G, and R
	cv::Mat result(mat.rows * 3, mat.cols, CV_8UC1);
	std::vector<cv::Mat> views =
	{
		// note first and last "views" are swapped since we want RGB and cv::Mat contains BGR
		result.rowRange(mat.rows * 2, mat.rows * 3),	// B
		result.rowRange(mat.rows * 1, mat.rows * 2),	// G
		result.rowRange(mat.rows * 0, mat.rows * 1),	// R
	};
	cv::split(mat, views);

	// create an empty Darknet::Image where the float results will be stored by OpenCV once we convert to CV_32F
	Darknet::Image img = make_image(mat.cols, mat.rows, mat.channels());

	// note how the cv::Matf is using the Darknet::Image data buffer directly
	cv::Mat1f tmp(img.h * 3, img.w, img.data);

	// convert the results to floating point, and divide by 255 to normalize between 0.0 - 1.0
	result.convertTo(tmp, CV_32F, 1.0/255.0);

	return img;
}


cv::Mat Darknet::image_to_mat(const Darknet::Image & img)
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
				float val = img.data[c * img.h * img.w + y * img.w + x];
				mat.data[y * step + x * img.c + c] = std::round(val * 255.0f);
			}
		}
	}

	/* Remember you likely need to convert the image from RGB to BGR,
	 * since OpenCV normally uses BGR.  Search for "cv::COLOR_RGB2BGR".
	 */

	return mat;
}


cv::Mat Darknet::rgb_image_to_bgr_mat(const Darknet::Image & img)
{
	TAT(TATPARMS);

	// This code assumes the image object is in Darknet's default RGB format!

	cv::Mat1f tmp(img.h * img.c, img.w, img.data);

	// convert the floats to "int", giving us a channel-packed image
	cv::Mat result;
	tmp.convertTo(result, CV_8U, 255.0);

	// create 3 views into the result to access B, G, and R values
	std::vector<cv::Mat> views =
	{
		result.rowRange(img.h * 2, img.h * 3),	// B
		result.rowRange(img.h * 1, img.h * 2),	// G
		result.rowRange(img.h * 0, img.h * 1),	// R
	};
	cv::Mat dst;
	cv::merge(views, dst);

	return dst;
}


Darknet::Image Darknet::make_random_image(int w, int h, int c)
{
	TAT(TATPARMS);

	Darknet::Image out = make_empty_image(w,h,c);
	out.data = (float*)xcalloc(h * w * c, sizeof(float));
	for(int i = 0; i < w*h*c; ++i)
	{
		out.data[i] = (rand_normal() * .25) + .5;
	}
	return out;
}


Darknet::Image Darknet::float_to_image_scaled(int w, int h, int c, float *data)
{
	TAT(TATPARMS);

	Darknet::Image out = make_image(w, h, c);
	int abs_max = 0;
	for (int i = 0; i < w*h*c; ++i)
	{
		if (fabs(data[i]) > abs_max)
		{
			abs_max = fabs(data[i]);
		}
	}
	for (int i = 0; i < w*h*c; ++i)
	{
		out.data[i] = data[i] / abs_max;
	}

	return out;
}


Darknet::Image Darknet::float_to_image(int w, int h, int c, float *data)
{
	TAT(TATPARMS);

	Darknet::Image out = make_empty_image(w,h,c);
	out.data = data;

	return out;
}


Darknet::Image Darknet::rotate_crop_image(const Darknet::Image & im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
	TAT(TATPARMS);

	float cx = im.w/2.;
	float cy = im.h/2.;
	Darknet::Image rot = make_image(w, h, im.c);

	for (int c = 0; c < im.c; ++c)
	{
		for (int y = 0; y < h; ++y)
		{
			for (int x = 0; x < w; ++x)
			{
				float rx = cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - sin(rad)*((y - h/2.)/s + dy/s) + cx;
				float ry = sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + cos(rad)*((y - h/2.)/s + dy/s) + cy;
				float val = bilinear_interpolate(im, rx, ry, c);
				set_pixel(rot, x, y, c, val);
			}
		}
	}

	return rot;
}


Darknet::Image Darknet::rotate_image(const Darknet::Image & im, float rad)
{
	TAT(TATPARMS);

	int x, y, c;
	float cx = im.w/2.;
	float cy = im.h/2.;
	Darknet::Image rot = make_image(im.w, im.h, im.c);

	for (c = 0; c < im.c; ++c)
	{
		for (y = 0; y < im.h; ++y)
		{
			for (x = 0; x < im.w; ++x)
			{
				float rx = cos(rad)*(x-cx) - sin(rad)*(y-cy) + cx;
				float ry = sin(rad)*(x-cx) + cos(rad)*(y-cy) + cy;
				float val = bilinear_interpolate(im, rx, ry, c);
				set_pixel(rot, x, y, c, val);
			}
		}
	}
	return rot;
}


void translate_image(Darknet::Image m, float s)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] += s;
}

void Darknet::scale_image(Darknet::Image & m, const float s)
{
	TAT(TATPARMS);

	for(int i = 0; i < m.h * m.w * m.c; ++i)
	{
		m.data[i] *= s;
	}

	return;
}


Darknet::Image Darknet::crop_image(const Darknet::Image & im, const int dx, const int dy, const int w, const int h)
{
	TAT(TATPARMS);

	Darknet::Image cropped = make_image(w, h, im.c);

	for (int k = 0; k < im.c; ++k)
	{
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				const int row = std::clamp(j + dy, 0, im.h - 1);
				const int col = std::clamp(i + dx, 0, im.w - 1);
				const float val = get_pixel(im, col, row, k);
				set_pixel(cropped, i, j, k, val);
			}
		}
	}

	return cropped;
}


int Darknet::best_3d_shift_r(const Darknet::Image & a, const Darknet::Image & b, int min, int max)
{
	TAT(TATPARMS);

	if(min == max)
	{
		return min;
	}

	const int mid = floor((min + max) / 2.0f);
	Darknet::Image c1 = Darknet::crop_image(b, 0, mid, b.w, b.h);
	Darknet::Image c2 = Darknet::crop_image(b, 0, mid+1, b.w, b.h);
	float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
	float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
	Darknet::free_image(c1);
	Darknet::free_image(c2);
	if (d1 < d2)
	{
		return Darknet::best_3d_shift_r(a, b, min, mid);
	}

	return Darknet::best_3d_shift_r(a, b, mid+1, max);
}


int Darknet::best_3d_shift(const Darknet::Image & a, const Darknet::Image & b, int min, int max)
{
	TAT(TATPARMS);

	int i;
	int best = 0;
	float best_distance = FLT_MAX;
	for (i = min; i <= max; i += 2)
	{
		Darknet::Image c = Darknet::crop_image(b, 0, i, b.w, b.h);
		float d = dist_array(c.data, a.data, a.w*a.h*a.c, 100);
		if (d < best_distance)
		{
			best_distance = d;
			best = i;
		}

		*cfg_and_state.output << i << " " << d << std::endl;

		Darknet::free_image(c);
	}

	return best;
}

void Darknet::composite_3d(char *f1, char *f2, const char *out, int delta)
{
	TAT(TATPARMS);

	if (!out)
	{
		out = "out";
	}

	Darknet::Image a = Darknet::load_image(f1, 0,0,0);
	Darknet::Image b = Darknet::load_image(f2, 0,0,0);
	int shift = Darknet::best_3d_shift_r(a, b, -a.h/100, a.h/100);

	Darknet::Image c1 = Darknet::crop_image(b, 10, shift, b.w, b.h);
	float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 100);
	Darknet::Image c2 = Darknet::crop_image(b, -10, shift, b.w, b.h);
	float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 100);

	if(d2 < d1)
	{
		std::swap(a, b);
		shift = -shift;
		*cfg_and_state.output << "swapped, ";
	}

	*cfg_and_state.output << shift << std::endl;

	Darknet::Image c = Darknet::crop_image(b, delta, shift, a.w, a.h);
	for (int i = 0; i < c.w*c.h; ++i)
	{
		c.data[i] = a.data[i];
	}

	save_image_jpg(c, out);
}


void Darknet::fill_image(Darknet::Image & m, float s)
{
	TAT(TATPARMS);

	for (int i = 0; i < m.h * m.w * m.c; ++i)
	{
		m.data[i] = s;
	}

	return;
}


void Darknet::letterbox_image_into(const Darknet::Image & im, int w, int h, Darknet::Image & boxed)
{
	TAT(TATPARMS);

	int new_w = im.w;
	int new_h = im.h;
	if (((float)w / im.w) < ((float)h / im.h))
	{
		new_w = w;
		new_h = (im.h * w) / im.w;
	}
	else
	{
		new_h = h;
		new_w = (im.w * h) / im.h;
	}
	Darknet::Image resized = Darknet::resize_image(im, new_w, new_h);
	Darknet::embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
	Darknet::free_image(resized);
}


Darknet::Image Darknet::letterbox_image(const Darknet::Image & im, int w, int h)
{
	TAT(TATPARMS);

	int new_w = im.w;
	int new_h = im.h;
	if (((float)w / im.w) < ((float)h / im.h))
	{
		new_w = w;
		new_h = (im.h * w) / im.w;
	}
	else
	{
		new_h = h;
		new_w = (im.w * h) / im.h;
	}
	Darknet::Image resized = Darknet::resize_image(im, new_w, new_h);
	Darknet::Image boxed = make_image(w, h, im.c);
	Darknet::fill_image(boxed, .5);
	//int i;
	//for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
	Darknet::embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
	Darknet::free_image(resized);

	return boxed;
}


Darknet::Image Darknet::random_crop_image(const Darknet::Image & im, const int w, const int h)
{
	TAT(TATPARMS);

	const int dx = rand_int(0, im.w - w);
	const int dy = rand_int(0, im.h - h);

	return Darknet::crop_image(im, dx, dy, w, h);
}


Darknet::Image Darknet::random_augment_image(const Darknet::Image & im, const float angle, float aspect, const int low, const int high, const int size)
{
	TAT(TATPARMS);

	aspect = rand_scale(aspect);
	int r = rand_int(low, high);
	int min = (im.h < im.w * aspect) ? im.h : im.w * aspect;
	float scale = static_cast<float>(r) / min;

	float rad = rand_uniform(-angle, angle) * 2.0f * M_PI / 360.0f;

	float dx = (im.w*scale/aspect - size) / 2.;
	float dy = (im.h*scale - size) / 2.;
	if (dx < 0) dx = 0;
	if (dy < 0) dy = 0;
	dx = rand_uniform(-dx, dx);
	dy = rand_uniform(-dy, dy);

	return rotate_crop_image(im, rad, scale, size, size, dx, dy, aspect);
}


void Darknet::rgb_to_hsv(Darknet::Image & im)
{
	TAT(TATPARMS);

	// http://www.cs.rit.edu/~ncs/color/t_convert.html

	/// @todo COLOR - cannot do HSV if channels > 3

	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float h, s, v;
	for(j = 0; j < im.h; ++j){
		for(i = 0; i < im.w; ++i){
			r = get_pixel(im, i , j, 0);
			g = get_pixel(im, i , j, 1);
			b = get_pixel(im, i , j, 2);
			float max = three_way_max(r,g,b);
			float min = three_way_min(r,g,b);
			float delta = max - min;
			v = max;
			if(max == 0){
				s = 0;
				h = 0;
			}else{
				s = delta/max;
				if(r == max){
					h = (g - b) / delta;
				} else if (g == max) {
					h = 2 + (b - r) / delta;
				} else {
					h = 4 + (r - g) / delta;
				}
				if (h < 0) h += 6;
				h = h/6.;
			}
			set_pixel(im, i, j, 0, h);
			set_pixel(im, i, j, 1, s);
			set_pixel(im, i, j, 2, v);
		}
	}
}


void Darknet::hsv_to_rgb(Darknet::Image & im)
{
	TAT(TATPARMS);

	/// @todo COLOR - cannot do HSV if channels > 3

	assert(im.c == 3);
	int i, j;
	float r, g, b;
	float h, s, v;
	float f, p, q, t;
	for(j = 0; j < im.h; ++j){
		for(i = 0; i < im.w; ++i){
			h = 6 * get_pixel(im, i , j, 0);
			s = get_pixel(im, i , j, 1);
			v = get_pixel(im, i , j, 2);
			if (s == 0) {
				r = g = b = v;
			} else {
				int index = floor(h);
				f = h - index;
				p = v*(1-s);
				q = v*(1-s*f);
				t = v*(1-s*(1-f));
				if(index == 0){
					r = v; g = t; b = p;
				} else if(index == 1){
					r = q; g = v; b = p;
				} else if(index == 2){
					r = p; g = v; b = t;
				} else if(index == 3){
					r = p; g = q; b = v;
				} else if(index == 4){
					r = t; g = p; b = v;
				} else {
					r = v; g = p; b = q;
				}
			}
			set_pixel(im, i, j, 0, r);
			set_pixel(im, i, j, 1, g);
			set_pixel(im, i, j, 2, b);
		}
	}
}


Darknet::Image Darknet::grayscale_image(const Darknet::Image & im)
{
	TAT(TATPARMS);

	assert(im.c == 3);

	Darknet::Image gray = make_image(im.w, im.h, 1);
	float scale[] = {0.587f, 0.299f, 0.114f};

	for (int k = 0; k < im.c; ++k)
	{
		for (int j = 0; j < im.h; ++j)
		{
			for (int i = 0; i < im.w; ++i)
			{
				gray.data[i + im.w * j] += scale[k] * get_pixel(im, i, j, k);
			}
		}
	}

	return gray;
}


Darknet::Image Darknet::threshold_image(const Darknet::Image & im, float thresh)
{
	TAT(TATPARMS);

	Darknet::Image t = make_image(im.w, im.h, im.c);

	for (int i = 0; i < im.w * im.h * im.c; ++i)
	{
		t.data[i] = im.data[i]>thresh ? 1.0f : 0.0f;
	}

	return t;
}


Darknet::Image Darknet::blend_image(const Darknet::Image & fore, const Darknet::Image & back, float alpha)
{
	TAT(TATPARMS);

	assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
	Darknet::Image blend = make_image(fore.w, fore.h, fore.c);

	for (int k = 0; k < fore.c; ++k)
	{
		for (int j = 0; j < fore.h; ++j)
		{
			for (int i = 0; i < fore.w; ++i)
			{
				float val = alpha * get_pixel(fore, i, j, k) + (1 - alpha)* get_pixel(back, i, j, k);
				set_pixel(blend, i, j, k, val);
			}
		}
	}

	return blend;
}


void Darknet::scale_image_channel(Darknet::Image & im, int c, float v)
{
	TAT(TATPARMS);

	for (int j = 0; j < im.h; ++j)
	{
		for (int i = 0; i < im.w; ++i)
		{
			float pix = get_pixel(im, i, j, c);
			pix = pix*v;
			set_pixel(im, i, j, c, pix);
		}
	}

	return;
}


void Darknet::distort_image(Darknet::Image & im, float hue, float sat, float val)
{
	TAT(TATPARMS);

	/// @todo COLOR - needs to be fixed for 1 <= c <= N

	if (im.c >= 3)
	{
		rgb_to_hsv(im);
		scale_image_channel(im, 1, sat);
		scale_image_channel(im, 2, val);

		for (int i = 0; i < im.w*im.h; ++i)
		{
			im.data[i] = im.data[i] + hue;
			if (im.data[i] > 1) im.data[i] -= 1.0f;
			if (im.data[i] < 0) im.data[i] += 1.0f;
		}
		Darknet::hsv_to_rgb(im);
	}
	else
	{
		scale_image_channel(im, 0, val);
	}

	constrain_image(im);

	return;
}


void Darknet::random_distort_image(Darknet::Image & im, float hue, float saturation, float exposure)
{
	TAT(TATPARMS);

	/// @todo COLOR - HSV no beuno

	float dhue = rand_uniform(-hue, hue);
	float dsat = rand_scale(saturation);
	float dexp = rand_scale(exposure);
	Darknet::distort_image(im, dhue, dsat, dexp);

	return;
}


float Darknet::bilinear_interpolate(const Darknet::Image & im, float x, float y, int c)
{
	TAT(TATPARMS);

	int ix = (int) floorf(x);
	int iy = (int) floorf(y);

	float dx = x - ix;
	float dy = y - iy;

	float val =
		(1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) +
		dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) +
		(1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
		dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);

	return val;
}


void Darknet::make_image_red(Darknet::Image & im)
{
	TAT(TATPARMS);

	for (int r = 0; r < im.h; ++r)
	{
		for (int c = 0; c < im.w; ++c)
		{
			float val = 0;
			for (int k = 0; k < im.c; ++k)
			{
				val += get_pixel(im, c, r, k);
				set_pixel(im, c, r, k, 0);
			}

			set_pixel(im, c, r, 0, val);
		}
	}
}


Darknet::Image Darknet::make_attention_image(int img_size, float *original_delta_cpu, float *original_input_cpu, int w, int h, int c, float alpha)
{
	TAT(TATPARMS);

	Darknet::Image attention_img;
	attention_img.w = w;
	attention_img.h = h;
	attention_img.c = c;
	attention_img.data = original_delta_cpu;
	make_image_red(attention_img);

	float min_val = 999999, mean_val = 0, max_val = -999999;
	for (int k = 0; k < img_size; ++k)
	{
		if (original_delta_cpu[k] < min_val)
		{
			min_val = original_delta_cpu[k];
		}
		if (original_delta_cpu[k] > max_val)
		{
			max_val = original_delta_cpu[k];
		}
		mean_val += original_delta_cpu[k];
	}
	mean_val = mean_val / img_size;
	float range = max_val - min_val;

	for (int k = 0; k < img_size; ++k)
	{
		float val = original_delta_cpu[k];
		val = fabs(mean_val - val) / range;
		original_delta_cpu[k] = val * 4;
	}

	Darknet::Image resized = Darknet::resize_image(attention_img, w / 4, h / 4);
	attention_img = Darknet::resize_image(resized, w, h);
	Darknet::free_image(resized);
	for (int k = 0; k < img_size; ++k)
	{
		attention_img.data[k] = attention_img.data[k]*alpha + (1-alpha)*original_input_cpu[k];
	}

	return attention_img;
}


Darknet::Image Darknet::resize_image(const Darknet::Image & im, int w, int h)
{
	TAT(TATPARMS);

	if (im.w == w && im.h == h)
	{
		return copy_image(im);
	}

	Darknet::Image resized = make_image(w, h, im.c);
	Darknet::Image part = make_image(w, im.h, im.c);

	const float w_scale = (im.w - 1.0f) / (w - 1.0f);
	const float h_scale = (im.h - 1.0f) / (h - 1.0f);

	#pragma omp parallel for
	for (int channel = 0; channel < im.c; ++channel)
	{
		for (int height = 0; height < im.h; ++height)
		{
			for (int c = 0; c < w; ++c)
			{
				float val = 0;
				if (c == w - 1 || im.w == 1)
				{
					val = get_pixel(im, im.w - 1, height, channel);
				}
				else
				{
					float sx = c * w_scale;
					int ix = (int) sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(im, ix, height, channel) + dx * get_pixel(im, ix+1, height, channel);
				}
				set_pixel(part, c, height, channel, val);
			}
		}
	}

	#pragma omp parallel for
	for (int channel = 0; channel < im.c; ++channel)
	{
		for (int height = 0; height < h; ++height)
		{
			float sy = height * h_scale;
			int iy = (int) sy;
			float dy = sy - iy;
			for (int c = 0; c < w; ++c)
			{
				float val = (1 - dy) * get_pixel(part, c, iy, channel);
				set_pixel(resized, c, height, channel, val);
			}

			if (height == h - 1 || im.h == 1)
			{
				continue;
			}

			for (int c = 0; c < w; ++c)
			{
				float val = dy * get_pixel(part, c, iy + 1, channel);
				add_pixel(resized, c, height, channel, val);
			}
		}
	}

	Darknet::free_image(part);

	return resized;
}


void Darknet::test_resize(char *filename)
{
	TAT(TATPARMS);

	Darknet::Image im = Darknet::load_image(filename, 0,0, 3);
	float mag = mag_array(im.data, im.w*im.h*im.c);
	*cfg_and_state.output << "L2 Norm: " << mag << std::endl;
	Darknet::Image gray = Darknet::grayscale_image(im);

	Darknet::Image c1 = Darknet::copy_image(im);
	Darknet::Image c2 = Darknet::copy_image(im);
	Darknet::Image c3 = Darknet::copy_image(im);
	Darknet::Image c4 = Darknet::copy_image(im);
	Darknet::distort_image(c1, .1, 1.5, 1.5);
	Darknet::distort_image(c2, -.1, .66666, .66666);
	Darknet::distort_image(c3, .1, 1.5, .66666);
	Darknet::distort_image(c4, .1, .66666, 1.5);

	Darknet::show_image(im,   "Original");
	Darknet::show_image(gray, "Gray");
	Darknet::show_image(c1, "C1");
	Darknet::show_image(c2, "C2");
	Darknet::show_image(c3, "C3");
	Darknet::show_image(c4, "C4");

	while(1)
	{
		Darknet::Image aug = random_augment_image(im, 0, .75, 320, 448, 320);
		Darknet::show_image(aug, "aug");
		Darknet::free_image(aug);

		float exposure = 1.15;
		float saturation = 1.15;
		float hue = .05;

		Darknet::Image c = copy_image(im);

		float dexp = rand_scale(exposure);
		float dsat = rand_scale(saturation);
		float dhue = rand_uniform(-hue, hue);

		Darknet::distort_image(c, dhue, dsat, dexp);
		Darknet::show_image(c, "rand");
		*cfg_and_state.output << dhue << " " << dsat << " " << dexp << std::endl;
		Darknet::free_image(c);
		cv::waitKey(0);
	}
}


DarknetImage load_image_v2(const char * filename, int desired_width, int desired_height, int channels)
{
	TAT(TATPARMS);

	return Darknet::load_image(filename, desired_width, desired_height, channels);
}


Darknet::Image Darknet::load_image(const char * filename, int desired_width, int desired_height, int channels)
{
	TAT(TATPARMS);

	Darknet::Image image;

	cv::Mat mat = cv::imread(filename);
	if (mat.empty())
	{
		darknet_fatal_error(DARKNET_LOC, "failed to load image file \"%s\"", filename);
	}

#if 1 // see the comment block at the bottom of this function

	// faster to use OpenCV to resize the image than using Darknet, so do it now if we know the size we need
	if (desired_width > 0 and desired_height > 0 and (desired_width != mat.cols or desired_height != mat.rows))
	{
		/* Normally we'd use INTER_AREA to shrink an image and INTER_CUBIC or INTER_LINEAR to grow and image.  But we want to
		 * try and retain as much of the original "Darknet resize" look-and-feel as possible.  Of all the OpenCV resize
		 * interpolation flags, the one that looks most similar to the original resize is INTER_LINEAR_EXACT.
		 */

		cv::resize(mat, mat, cv::Size(desired_width, desired_height), 0.0, 0.0, cv::InterpolationFlags::INTER_LINEAR_EXACT);
	}
#endif

	if (mat.channels() == 3 and (channels == 0 or channels == 3))
	{
		// use the new faster conversion code when dealing with 3-channel image (which is the norm)
		image = bgr_mat_to_rgb_image(mat);
	}
	else
	{
		// if we get here then we have a more complex scenario to deal with...

		if (channels == 0)
		{
			// we didn't specify how many channels we want...so assume it is 3-channel RGB
			channels = 3;
		}

		if (channels == 1 and mat.channels() == 1)
		{
			// nothing to do, we already have a greyscale image
			image = mat_to_image(mat);
		}
		else if (channels == 1 and mat.channels() == 3)
		{
			cv::cvtColor(mat, mat, cv::COLOR_BGR2GRAY);
			image = mat_to_image(mat);
		}
		else if (channels == 1 and mat.channels() == 4)
		{
			cv::cvtColor(mat, mat, cv::COLOR_BGRA2GRAY);
			image = mat_to_image(mat);
		}
		else if (channels == 3 and mat.channels() == 1)
		{
			cv::cvtColor(mat, mat, cv::COLOR_GRAY2RGB);
			image = rgb_mat_to_rgb_image(mat);
		}
		else if (channels == 3 and mat.channels() == 3)
		{
			// shouldn't this case have been handled at the top of the parent if() clause?
			image = bgr_mat_to_rgb_image(mat);
		}
		else if (channels == 3 and mat.channels() == 4)
		{
			cv::cvtColor(mat, mat, cv::COLOR_BGRA2RGB);
			image = rgb_mat_to_rgb_image(mat);
		}
		else
		{
			// what in the world do we have?
			image = bgr_mat_to_rgb_image(mat);
		}
	}

#if 0
	/** @todo V5:  OpenCV is faster at resizing than the Darknet resize function, so we now use Darknet to resize.  But
	 * the image quality is different, and I worry this may cause problems with people getting unexpected results.  Keep
	 * this old code here for a while in case we need to revert back to the old resize method.
	 */
	if (desired_width > 0 and desired_height > 0)
	{
		Darknet::Image resized = Darknet::resize_image(image, desired_width, desired_height);
		Darknet::free_image(image);
		image = resized;
	}
#endif

	return image;
}


Darknet::Image Darknet::get_image_layer(const Darknet::Image & m, int l)
{
	TAT(TATPARMS);

	Darknet::Image out = make_image(m.w, m.h, 1);

	for (int i = 0; i < m.h*m.w; ++i)
	{
		out.data[i] = m.data[i + l * m.h * m.w];
	}

	return out;
}


std::string Darknet::image_as_debug_string(const Darknet::Image & im)
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss << "Darknet::Image: " << im.w << "x" << im.h << "x" << im.c << ", data=" << (void*)im.data;

	if (im.data != nullptr)
	{
		const size_t number_of_elements = im.w * im.h * im.c;
		const size_t elements_per_channel = number_of_elements / im.c;

		for (size_t idx = 0; idx < number_of_elements; idx ++)
		{
			if (idx % im.w == 0) // indicates the start of a new row
			{
				ss	<< std::endl
					<< (idx < elements_per_channel ? "R" : idx < 2 * elements_per_channel ? "G" : "B")
					<< " "
					<< std::setfill('0') << std::setw(4) << idx << ":";
			}

			ss << " " << std::fixed << std::setprecision(2) << im.data[idx];
		}
	}

	return ss.str();
}


Darknet::Image Darknet::collapse_images_vert(const Darknet::Image * ims, int n)
{
	TAT(TATPARMS);

	int color = 1;
	int border = 1;

	int w = ims[0].w;
	int h = (ims[0].h + border) * n - border;
	int c = ims[0].c;
	if(c != 3 || !color)
	{
		w = (w+border)*c - border;
		c = 1;
	}

	Darknet::Image filters = make_image(w, h, c);

	for (int i = 0; i < n; ++i)
	{
		int h_offset = i*(ims[0].h+border);
		Darknet::Image copy = copy_image(ims[i]);
		//normalize_image(copy);
		if (c == 3 && color)
		{
			embed_image(copy, filters, 0, h_offset);
		}
		else
		{
			for (int j = 0; j < copy.c; ++j)
			{
				int w_offset = j*(ims[0].w+border);
				Darknet::Image layer = get_image_layer(copy, j);
				embed_image(layer, filters, w_offset, h_offset);
				Darknet::free_image(layer);
			}
		}
		Darknet::free_image(copy);
	}

	return filters;
}


Darknet::Image collapse_images_horz(const Darknet::Image * ims, int n)
{
	TAT(TATPARMS);

	int color = 1;
	int border = 1;
	int size = ims[0].h;
	int h = size;
	int w = (ims[0].w + border) * n - border;
	int c = ims[0].c;
	if(c != 3 || !color)
	{
		h = (h+border)*c - border;
		c = 1;
	}

	Darknet::Image filters = make_image(w, h, c);

	for (int i = 0; i < n; ++i)
	{
		int w_offset = i*(size+border);
		Darknet::Image copy = Darknet::copy_image(ims[i]);

		/// @todo COLOR
		if(c == 3 && color)
		{
			Darknet::embed_image(copy, filters, w_offset, 0);
		}
		else
		{
			for (int j = 0; j < copy.c; ++j)
			{
				int h_offset = j*(size+border);
				Darknet::Image layer = Darknet::get_image_layer(copy, j);
				Darknet::embed_image(layer, filters, w_offset, h_offset);
				Darknet::free_image(layer);
			}
		}
		Darknet::free_image(copy);
	}
	return filters;
}


void Darknet::show_images(Darknet::Image *ims, int n, const char * window)
{
	TAT(TATPARMS);

	Darknet::Image m = collapse_images_vert(ims, n);

	normalize_image(m);
	save_image(m, window);
	Darknet::show_image(m, window);
	Darknet::free_image(m);

	return;
}


// this is the "C" version of the call, where the image is not passed by reference
void free_image(image im)
{
	TAT(TATPARMS);

	Darknet::free_image(im);

	return;
}


// this is the C++ version of the call, where the image is passed by reference
void Darknet::free_image(Darknet::Image & im)
{
	TAT(TATPARMS);

	if (im.data)
	{
		free(im.data);
		im.data = nullptr;
	}

	im.w = 0;
	im.h = 0;
	im.c = 0;

	return;
}


// Fast copy data from a contiguous byte array into the image.
void copy_image_from_bytes(DarknetImage im, char *pdata)
{
	TAT(TATPARMS);

	unsigned char *data = (unsigned char*)pdata;
	int w = im.w;
	int h = im.h;
	int c = im.c;

	for (int k = 0; k < c; ++k)
	{
		for (int j = 0; j < h; ++j)
		{
			for (int i = 0; i < w; ++i)
			{
				int dst_index = i + w * j + w * h * k;
				int src_index = k + c * i + c * w * j;
				im.data[dst_index] = (float)data[src_index] / 255.0f;
			}
		}
	}
}
