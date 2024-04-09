#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "image.hpp"
#include "utils.hpp"
#include "blas.hpp"
#include "dark_cuda.hpp"
#include "darknet_utils.hpp"
#include "Timing.hpp"
#include <stdio.h>
#include <math.h>
#include <ciso646>

extern int check_mistakes;

float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };

float get_color(int c, int x, int max)
{
	TAT(TATPARMS);

	float ratio = ((float)x/max)*5;
	int i = floor(ratio);
	int j = ceil(ratio);
	ratio -= i;
	float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
	//printf("%f\n", r);
	return r;
}

static float get_pixel(image m, int x, int y, int c)
{
	TAT(TATPARMS);

	assert(x < m.w && y < m.h && c < m.c);
	return m.data[c*m.h*m.w + y*m.w + x];
}
static float get_pixel_extend(image m, int x, int y, int c)
{
	TAT(TATPARMS);

	if (x < 0 || x >= m.w || y < 0 || y >= m.h) return 0;

	if (c < 0 || c >= m.c) return 0;
	return get_pixel(m, x, y, c);
}
static void set_pixel(image m, int x, int y, int c, float val)
{
	TAT(TATPARMS);

	if (x < 0 || y < 0 || c < 0 || x >= m.w || y >= m.h || c >= m.c) return;
	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y*m.w + x] = val;
}
static void add_pixel(image m, int x, int y, int c, float val)
{
	TAT(TATPARMS);

	assert(x < m.w && y < m.h && c < m.c);
	m.data[c*m.h*m.w + y*m.w + x] += val;
}

void composite_image(image source, image dest, int dx, int dy)
{
	TAT(TATPARMS);

	int x,y,k;
	for(k = 0; k < source.c; ++k){
		for(y = 0; y < source.h; ++y){
			for(x = 0; x < source.w; ++x){
				float val = get_pixel(source, x, y, k);
				float val2 = get_pixel_extend(dest, dx+x, dy+y, k);
				set_pixel(dest, dx+x, dy+y, k, val * val2);
			}
		}
	}
}

image border_image(image a, int border)
{
	TAT(TATPARMS);

	image b = make_image(a.w + 2*border, a.h + 2*border, a.c);
	int x,y,k;
	for(k = 0; k < b.c; ++k){
		for(y = 0; y < b.h; ++y){
			for(x = 0; x < b.w; ++x){
				float val = get_pixel_extend(a, x - border, y - border, k);
				if(x - border < 0 || x - border >= a.w || y - border < 0 || y - border >= a.h) val = 1;
				set_pixel(b, x, y, k, val);
			}
		}
	}
	return b;
}

image tile_images(image a, image b, int dx)
{
	TAT(TATPARMS);

	if(a.w == 0) return copy_image(b);
	image c = make_image(a.w + b.w + dx, (a.h > b.h) ? a.h : b.h, (a.c > b.c) ? a.c : b.c);
	fill_cpu(c.w*c.h*c.c, 1, c.data, 1);
	embed_image(a, c, 0, 0);
	composite_image(b, c, a.w + dx, 0);
	return c;
}

image get_label(image **characters, char *string, int size)
{
	TAT(TATPARMS);

	if(size > 7) size = 7;
	image label = make_empty_image(0,0,0);
	while(*string){
		image l = characters[size][(int)*string];
		image n = tile_images(label, l, -size - 1 + (size+1)/2);
		free_image(label);
		label = n;
		++string;
	}
	image b = border_image(label, label.h*.25);
	free_image(label);
	return b;
}

image get_opencv_label(const std::string & str, const int area)
{
	/// @todo what are the performance implications of LINE_AA over LINE_4 or LINE_8?

	TAT(TATPARMS);

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

	return mat_to_image(mat);
}

image get_label_v3(image **characters, char *string, int size)
{
	TAT(TATPARMS);

	size = size / 10;
	if (size > 7) size = 7;
	image label = make_empty_image(0, 0, 0);
	while (*string)
	{
		image l = characters[size][(int)*string];
		image n = tile_images(label, l, -size - 1 + (size + 1) / 2);
		free_image(label);
		label = n;
		++string;
	}
	image b = border_image(label, label.h*.05);
	free_image(label);
	return b;
}

void draw_label(image a, int r, int c, image label, const float *rgb)
{
	TAT(TATPARMS);

	int w = label.w;
	int h = label.h;
	if (r - h >= 0) r = r - h;

	int i, j, k;
	for(j = 0; j < h && j + r < a.h; ++j){
		for(i = 0; i < w && i + c < a.w; ++i){
			for(k = 0; k < label.c; ++k){
				float val = get_pixel(label, i, j, k);
				set_pixel(a, i+c, j+r, k, rgb[k] * val);
			}
		}
	}
}

void draw_weighted_label(image a, int r, int c, image label, const float *rgb, const float alpha)
{
	TAT(TATPARMS);

	int w = label.w;
	int h = label.h;
	if (r - h >= 0) r = r - h;

	int i, j, k;
	for (j = 0; j < h && j + r < a.h; ++j) {
		for (i = 0; i < w && i + c < a.w; ++i) {
			for (k = 0; k < label.c; ++k) {
				float val1 = get_pixel(label, i, j, k);
				float val2 = get_pixel(a, i + c, j + r, k);
				float val_dst = val1 * rgb[k] * alpha + val2 * (1 - alpha);
				set_pixel(a, i + c, j + r, k, val_dst);
			}
		}
	}
}

void draw_box_bw(image a, int x1, int y1, int x2, int y2, float brightness)
{
	TAT(TATPARMS);

	//normalize_image(a);
	int i;
	if (x1 < 0) x1 = 0;
	if (x1 >= a.w) x1 = a.w - 1;
	if (x2 < 0) x2 = 0;
	if (x2 >= a.w) x2 = a.w - 1;

	if (y1 < 0) y1 = 0;
	if (y1 >= a.h) y1 = a.h - 1;
	if (y2 < 0) y2 = 0;
	if (y2 >= a.h) y2 = a.h - 1;

	for (i = x1; i <= x2; ++i) {
		a.data[i + y1*a.w + 0 * a.w*a.h] = brightness;
		a.data[i + y2*a.w + 0 * a.w*a.h] = brightness;
	}
	for (i = y1; i <= y2; ++i) {
		a.data[x1 + i*a.w + 0 * a.w*a.h] = brightness;
		a.data[x2 + i*a.w + 0 * a.w*a.h] = brightness;
	}
}

void draw_box_width_bw(image a, int x1, int y1, int x2, int y2, int w, float brightness)
{
	TAT(TATPARMS);

	int i;
	for (i = 0; i < w; ++i) {
		float alternate_color = (w % 2) ? (brightness) : (1.0 - brightness);
		draw_box_bw(a, x1 + i, y1 + i, x2 - i, y2 - i, alternate_color);
	}
}

void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b)
{
	TAT(TATPARMS);

	//normalize_image(a);
	int i;
	if(x1 < 0) x1 = 0;
	if(x1 >= a.w) x1 = a.w-1;
	if(x2 < 0) x2 = 0;
	if(x2 >= a.w) x2 = a.w-1;

	if(y1 < 0) y1 = 0;
	if(y1 >= a.h) y1 = a.h-1;
	if(y2 < 0) y2 = 0;
	if(y2 >= a.h) y2 = a.h-1;

	for(i = x1; i <= x2; ++i){
		a.data[i + y1*a.w + 0*a.w*a.h] = r;
		a.data[i + y2*a.w + 0*a.w*a.h] = r;

		a.data[i + y1*a.w + 1*a.w*a.h] = g;
		a.data[i + y2*a.w + 1*a.w*a.h] = g;

		a.data[i + y1*a.w + 2*a.w*a.h] = b;
		a.data[i + y2*a.w + 2*a.w*a.h] = b;
	}
	for(i = y1; i <= y2; ++i){
		a.data[x1 + i*a.w + 0*a.w*a.h] = r;
		a.data[x2 + i*a.w + 0*a.w*a.h] = r;

		a.data[x1 + i*a.w + 1*a.w*a.h] = g;
		a.data[x2 + i*a.w + 1*a.w*a.h] = g;

		a.data[x1 + i*a.w + 2*a.w*a.h] = b;
		a.data[x2 + i*a.w + 2*a.w*a.h] = b;
	}
}

void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < w; ++i){
		draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b);
	}
}

void draw_bbox(image a, box bbox, int w, float r, float g, float b)
{
	TAT(TATPARMS);

	int left  = (bbox.x-bbox.w/2)*a.w;
	int right = (bbox.x+bbox.w/2)*a.w;
	int top   = (bbox.y-bbox.h/2)*a.h;
	int bot   = (bbox.y+bbox.h/2)*a.h;

	int i;
	for(i = 0; i < w; ++i){
		draw_box(a, left+i, top+i, right-i, bot-i, r, g, b);
	}
}

// Creates array of detections with prob > thresh and fills best_class for them
detection_with_class* get_actual_detections(detection *dets, int dets_num, float thresh, int* selected_detections_num, char **names)
{
	TAT(TATPARMS);

	int selected_num = 0;
	detection_with_class* result_arr = (detection_with_class*)xcalloc(dets_num, sizeof(detection_with_class));
	int i;
	for (i = 0; i < dets_num; ++i) {
		int best_class = -1;
		float best_class_prob = thresh;
		int j;
		for (j = 0; j < dets[i].classes; ++j) {
			int show = strncmp(names[j], "dont_show", 9);
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

void draw_detections_v3(image im, detection *dets, int num, float thresh, char **names, int classes, int ext_output)
{
	TAT(TATPARMS);

	static int frame_id = 0;
	frame_id++;

	int selected_detections_num;
	detection_with_class* selected_detections = get_actual_detections(dets, num, thresh, &selected_detections_num, names);

	// text output
	/// @todo replace qsort() mid priority
	qsort(selected_detections, selected_detections_num, sizeof(*selected_detections), compare_by_lefts);

	int i;
	for (i = 0; i < selected_detections_num; ++i)
	{
		const int best_class = selected_detections[i].best_class;
		printf("%s: %.0f%%", names[best_class], selected_detections[i].det.prob[best_class] * 100);
		if (ext_output)
		{
			printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
				round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w),
				round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h),
				round(selected_detections[i].det.bbox.w*im.w), round(selected_detections[i].det.bbox.h*im.h));
		}
		else
		{
			printf("\n");
		}
		int j;
		for (j = 0; j < classes; ++j)
		{
			if (selected_detections[i].det.prob[j] > thresh && j != best_class)
			{
				printf("%s: %.0f%%", names[j], selected_detections[i].det.prob[j] * 100);

				if (ext_output)
				{
					printf("\t(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)\n",
						round((selected_detections[i].det.bbox.x - selected_detections[i].det.bbox.w / 2)*im.w),
						round((selected_detections[i].det.bbox.y - selected_detections[i].det.bbox.h / 2)*im.h),
						round(selected_detections[i].det.bbox.w*im.w), round(selected_detections[i].det.bbox.h*im.h));
				}
				else
				{
					printf("\n");
				}
			}
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

		//printf("%d %s: %.0f%%\n", i, names[selected_detections[i].best_class], prob*100);
		int offset = selected_detections[i].best_class * 123457 % classes;
		float red = get_color(2, offset, classes);
		float green = get_color(1, offset, classes);
		float blue = get_color(0, offset, classes);
		float rgb[3];

		rgb[0] = red;
		rgb[1] = green;
		rgb[2] = blue;
		box b = selected_detections[i].det.bbox;
		//printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

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
			draw_box_width(im, left, top, right, bot, width, red, green, blue); // 3 channels RGB
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

		image label = get_opencv_label(ss.str(), (right - left) * (bot - top));
		draw_weighted_label(im, top + width, left, label, rgb, 0.7);
		free_image(label);

		if (selected_detections[i].det.mask)
		{
			image mask = float_to_image(14, 14, 1, selected_detections[i].det.mask);
			image resized_mask = resize_image(mask, b.w*im.w, b.h*im.h);
			image tmask = threshold_image(resized_mask, .5);
			embed_image(tmask, im, left, top);
			free_image(mask);
			free_image(resized_mask);
			free_image(tmask);
		}
	}
	free(selected_detections);
}

void draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, int classes)
{
	// wrong function!  see draw_detections_v3() instead!

	TAT(TATPARMS);

	int i;

	for(i = 0; i < num; ++i){
		int class_id = max_index(probs[i], classes);
		float prob = probs[i][class_id];
		if(prob > thresh){

			int width = im.h * .012;

			if(0)
			{
				width = pow(prob, 1./2.)*10+1;
			}

			int offset = class_id*123457 % classes;
			float red = get_color(2,offset,classes);
			float green = get_color(1,offset,classes);
			float blue = get_color(0,offset,classes);

			box b = boxes[i];

			int left  = (b.x-b.w/2.)*im.w;
			int right = (b.x+b.w/2.)*im.w;
			int top   = (b.y-b.h/2.)*im.h;
			int bot   = (b.y+b.h/2.)*im.h;

			if(left < 0) left = 0;
			if(right > im.w-1) right = im.w-1;
			if(top < 0) top = 0;
			if(bot > im.h-1) bot = im.h-1;
			printf("%s: %.0f%%", names[class_id], prob * 100);

			printf("\n");
			draw_box_width(im, left, top, right, bot, width, red, green, blue);
		}
	}
}

void transpose_image(image im)
{
	TAT(TATPARMS);

	assert(im.w == im.h);
	int n, m;
	int c;
	for(c = 0; c < im.c; ++c){
		for(n = 0; n < im.w-1; ++n){
			for(m = n + 1; m < im.w; ++m){
				float swap = im.data[m + im.w*(n + im.h*c)];
				im.data[m + im.w*(n + im.h*c)] = im.data[n + im.w*(m + im.h*c)];
				im.data[n + im.w*(m + im.h*c)] = swap;
			}
		}
	}
}

void rotate_image_cw(image im, int times)
{
	TAT(TATPARMS);

	assert(im.w == im.h);
	times = (times + 400) % 4;
	int i, x, y, c;
	int n = im.w;
	for(i = 0; i < times; ++i){
		for(c = 0; c < im.c; ++c){
			for(x = 0; x < n/2; ++x){
				for(y = 0; y < (n-1)/2 + 1; ++y){
					float temp = im.data[y + im.w*(x + im.h*c)];
					im.data[y + im.w*(x + im.h*c)] = im.data[n-1-x + im.w*(y + im.h*c)];
					im.data[n-1-x + im.w*(y + im.h*c)] = im.data[n-1-y + im.w*(n-1-x + im.h*c)];
					im.data[n-1-y + im.w*(n-1-x + im.h*c)] = im.data[x + im.w*(n-1-y + im.h*c)];
					im.data[x + im.w*(n-1-y + im.h*c)] = temp;
				}
			}
		}
	}
}

void flip_image(image a)
{
	TAT(TATPARMS);

	int i,j,k;
	for(k = 0; k < a.c; ++k){
		for(i = 0; i < a.h; ++i){
			for(j = 0; j < a.w/2; ++j){
				int index = j + a.w*(i + a.h*(k));
				int flip = (a.w - j - 1) + a.w*(i + a.h*(k));
				float swap = a.data[flip];
				a.data[flip] = a.data[index];
				a.data[index] = swap;
			}
		}
	}
}

image image_distance(image a, image b)
{
	TAT(TATPARMS);

	int i,j;
	image dist = make_image(a.w, a.h, 1);
	for(i = 0; i < a.c; ++i){
		for(j = 0; j < a.h*a.w; ++j){
			dist.data[j] += pow(a.data[i*a.h*a.w+j]-b.data[i*a.h*a.w+j],2);
		}
	}
	for(j = 0; j < a.h*a.w; ++j){
		dist.data[j] = sqrt(dist.data[j]);
	}
	return dist;
}

void embed_image(image source, image dest, int dx, int dy)
{
	TAT(TATPARMS);

	int x,y,k;
	for(k = 0; k < source.c; ++k){
		for(y = 0; y < source.h; ++y){
			for(x = 0; x < source.w; ++x){
				float val = get_pixel(source, x,y,k);
				set_pixel(dest, dx+x, dy+y, k, val);
			}
		}
	}
}

image collapse_image_layers(image source, int border)
{
	TAT(TATPARMS);

	int h = source.h;
	h = (h+border)*source.c - border;
	image dest = make_image(source.w, h, 1);
	int i;
	for(i = 0; i < source.c; ++i){
		image layer = get_image_layer(source, i);
		int h_offset = i*(source.h+border);
		embed_image(layer, dest, 0, h_offset);
		free_image(layer);
	}
	return dest;
}

void constrain_image(image im)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < im.w*im.h*im.c; ++i){
		if(im.data[i] < 0) im.data[i] = 0;
		if(im.data[i] > 1) im.data[i] = 1;
	}
}

void normalize_image(image p)
{
	TAT(TATPARMS);

	int i;
	float min = 9999999;
	float max = -999999;

	for(i = 0; i < p.h*p.w*p.c; ++i){
		float v = p.data[i];
		if(v < min) min = v;
		if(v > max) max = v;
	}
	if(max - min < .000000001){
		min = 0;
		max = 1;
	}
	for(i = 0; i < p.c*p.w*p.h; ++i){
		p.data[i] = (p.data[i] - min)/(max-min);
	}
}

void normalize_image2(image p)
{
	TAT(TATPARMS);

	float* min = (float*)xcalloc(p.c, sizeof(float));
	float* max = (float*)xcalloc(p.c, sizeof(float));
	int i,j;
	for(i = 0; i < p.c; ++i) min[i] = max[i] = p.data[i*p.h*p.w];

	for(j = 0; j < p.c; ++j){
		for(i = 0; i < p.h*p.w; ++i){
			float v = p.data[i+j*p.h*p.w];
			if(v < min[j]) min[j] = v;
			if(v > max[j]) max[j] = v;
		}
	}
	for(i = 0; i < p.c; ++i){
		if(max[i] - min[i] < .000000001){
			min[i] = 0;
			max[i] = 1;
		}
	}
	for(j = 0; j < p.c; ++j){
		for(i = 0; i < p.w*p.h; ++i){
			p.data[i+j*p.h*p.w] = (p.data[i+j*p.h*p.w] - min[j])/(max[j]-min[j]);
		}
	}
	free(min);
	free(max);
}

void copy_image_inplace(image src, image dst)
{
	TAT(TATPARMS);

	memcpy(dst.data, src.data, src.h*src.w*src.c * sizeof(float));
}

image copy_image(image p)
{
	TAT(TATPARMS);

	image copy = p;
	copy.data = (float*)xcalloc(p.h * p.w * p.c, sizeof(float));
	memcpy(copy.data, p.data, p.h*p.w*p.c*sizeof(float));
	return copy;
}

void rgbgr_image(image im)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < im.w*im.h; ++i){
		float swap = im.data[i];
		im.data[i] = im.data[i+im.w*im.h*2];
		im.data[i+im.w*im.h*2] = swap;
	}
}

void show_image(image p, const char *name)
{
	TAT(TATPARMS);

	show_image_cv(p, name);
}

void save_image_png(image im, const char *name)
{
	/// @todo merge with @ref save_mat_png()

	TAT(TATPARMS);

	std::string filename = name;
	filename += ".png";

	cv::Mat mat = image_to_mat(im);

	if (mat.channels() == 3)
	{
		cv::cvtColor(mat, mat, cv::COLOR_RGB2BGR);
	}
	else if (mat.channels() == 4)
	{
		cv::cvtColor(mat, mat, cv::COLOR_RGBA2BGR);
	}

	const bool success = cv::imwrite(filename, mat, {cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION, 9});
	if (not success)
	{
		darknet_fatal_error(DARKNET_LOC, "failed to save the image %s", filename.c_str());
	}
}

void save_image_jpg(image im, const char *name)
{
	/// @todo merge with @ref save_mat_jpg()

	TAT(TATPARMS);

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
}

void save_image(image im, const char *name)
{
	save_image_jpg(im, name);
}

void save_image_options(image im, const char *name, IMTYPE f, int quality)
{
	TAT(TATPARMS);

	if (f == IMTYPE::PNG)
	{
		save_image_png(im, name);
	}
	else
	{
		// otherwise, every other format will be saved as JPG
		save_image_jpg(im, name);
	}
}

void show_image_layers(image p, char *name)
{
	TAT(TATPARMS);

	int i;
	char buff[256];
	for(i = 0; i < p.c; ++i){
		sprintf(buff, "%s - Layer %d", name, i);
		image layer = get_image_layer(p, i);
		show_image(layer, buff);
		free_image(layer);
	}
}

void show_image_collapsed(image p, char *name)
{
	TAT(TATPARMS);

	image c = collapse_image_layers(p, 1);
	show_image(c, name);
	free_image(c);
}

image make_empty_image(int w, int h, int c)
{
	TAT(TATPARMS);

	image out;
	out.data = 0;
	out.h = h;
	out.w = w;
	out.c = c;
	return out;
}

image make_image(int w, int h, int c)
{
	TAT(TATPARMS);

	image out = make_empty_image(w,h,c);
	out.data = (float*)xcalloc(h * w * c, sizeof(float));

	return out;
}

image make_random_image(int w, int h, int c)
{
	TAT(TATPARMS);

	image out = make_empty_image(w,h,c);
	out.data = (float*)xcalloc(h * w * c, sizeof(float));
	int i;
	for(i = 0; i < w*h*c; ++i){
		out.data[i] = (rand_normal() * .25) + .5;
	}
	return out;
}

image float_to_image_scaled(int w, int h, int c, float *data)
{
	TAT(TATPARMS);

	image out = make_image(w, h, c);
	int abs_max = 0;
	int i = 0;
	for (i = 0; i < w*h*c; ++i) {
		if (fabs(data[i]) > abs_max) abs_max = fabs(data[i]);
	}
	for (i = 0; i < w*h*c; ++i) {
		out.data[i] = data[i] / abs_max;
	}
	return out;
}

image float_to_image(int w, int h, int c, float *data)
{
	TAT(TATPARMS);

	image out = make_empty_image(w,h,c);
	out.data = data;
	return out;
}


image rotate_crop_image(image im, float rad, float s, int w, int h, float dx, float dy, float aspect)
{
	TAT(TATPARMS);

	int x, y, c;
	float cx = im.w/2.;
	float cy = im.h/2.;
	image rot = make_image(w, h, im.c);
	for(c = 0; c < im.c; ++c){
		for(y = 0; y < h; ++y){
			for(x = 0; x < w; ++x){
				float rx = cos(rad)*((x - w/2.)/s*aspect + dx/s*aspect) - sin(rad)*((y - h/2.)/s + dy/s) + cx;
				float ry = sin(rad)*((x - w/2.)/s*aspect + dx/s*aspect) + cos(rad)*((y - h/2.)/s + dy/s) + cy;
				float val = bilinear_interpolate(im, rx, ry, c);
				set_pixel(rot, x, y, c, val);
			}
		}
	}
	return rot;
}

image rotate_image(image im, float rad)
{
	TAT(TATPARMS);

	int x, y, c;
	float cx = im.w/2.;
	float cy = im.h/2.;
	image rot = make_image(im.w, im.h, im.c);
	for(c = 0; c < im.c; ++c){
		for(y = 0; y < im.h; ++y){
			for(x = 0; x < im.w; ++x){
				float rx = cos(rad)*(x-cx) - sin(rad)*(y-cy) + cx;
				float ry = sin(rad)*(x-cx) + cos(rad)*(y-cy) + cy;
				float val = bilinear_interpolate(im, rx, ry, c);
				set_pixel(rot, x, y, c, val);
			}
		}
	}
	return rot;
}

void translate_image(image m, float s)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] += s;
}

void scale_image(image m, float s)
{
	TAT(TATPARMS);

	int i;
	for(i = 0; i < m.h*m.w*m.c; ++i) m.data[i] *= s;
}

image crop_image(image im, int dx, int dy, int w, int h)
{
	TAT(TATPARMS);

	image cropped = make_image(w, h, im.c);
	int i, j, k;
	for(k = 0; k < im.c; ++k){
		for(j = 0; j < h; ++j){
			for(i = 0; i < w; ++i){
				int r = j + dy;
				int c = i + dx;
				float val = 0;
				r = constrain_int(r, 0, im.h-1);
				c = constrain_int(c, 0, im.w-1);
				if (r >= 0 && r < im.h && c >= 0 && c < im.w) {
					val = get_pixel(im, c, r, k);
				}
				set_pixel(cropped, i, j, k, val);
			}
		}
	}
	return cropped;
}

int best_3d_shift_r(image a, image b, int min, int max)
{
	TAT(TATPARMS);

	if(min == max) return min;
	int mid = floor((min + max) / 2.);
	image c1 = crop_image(b, 0, mid, b.w, b.h);
	image c2 = crop_image(b, 0, mid+1, b.w, b.h);
	float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 10);
	float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 10);
	free_image(c1);
	free_image(c2);
	if(d1 < d2) return best_3d_shift_r(a, b, min, mid);
	else return best_3d_shift_r(a, b, mid+1, max);
}

int best_3d_shift(image a, image b, int min, int max)
{
	TAT(TATPARMS);

	int i;
	int best = 0;
	float best_distance = FLT_MAX;
	for(i = min; i <= max; i += 2){
		image c = crop_image(b, 0, i, b.w, b.h);
		float d = dist_array(c.data, a.data, a.w*a.h*a.c, 100);
		if(d < best_distance){
			best_distance = d;
			best = i;
		}
		printf("%d %f\n", i, d);
		free_image(c);
	}
	return best;
}

void composite_3d(char *f1, char *f2, char *out, int delta)
{
	TAT(TATPARMS);

	if(!out) out = "out";
	image a = load_image(f1, 0,0,0);
	image b = load_image(f2, 0,0,0);
	int shift = best_3d_shift_r(a, b, -a.h/100, a.h/100);

	image c1 = crop_image(b, 10, shift, b.w, b.h);
	float d1 = dist_array(c1.data, a.data, a.w*a.h*a.c, 100);
	image c2 = crop_image(b, -10, shift, b.w, b.h);
	float d2 = dist_array(c2.data, a.data, a.w*a.h*a.c, 100);

	if(d2 < d1 && 0){
		image swap = a;
		a = b;
		b = swap;
		shift = -shift;
		printf("swapped, %d\n", shift);
	}
	else{
		printf("%d\n", shift);
	}

	image c = crop_image(b, delta, shift, a.w, a.h);
	int i;
	for(i = 0; i < c.w*c.h; ++i){
		c.data[i] = a.data[i];
	}

	save_image_jpg(c, out);
}

void fill_image(image m, float s)
{
	TAT(TATPARMS);

	int i;
	for (i = 0; i < m.h*m.w*m.c; ++i) m.data[i] = s;
}

void letterbox_image_into(image im, int w, int h, image boxed)
{
	TAT(TATPARMS);

	int new_w = im.w;
	int new_h = im.h;
	if (((float)w / im.w) < ((float)h / im.h)) {
		new_w = w;
		new_h = (im.h * w) / im.w;
	}
	else {
		new_h = h;
		new_w = (im.w * h) / im.h;
	}
	image resized = resize_image(im, new_w, new_h);
	embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
	free_image(resized);
}

image letterbox_image(image im, int w, int h)
{
	TAT(TATPARMS);

	int new_w = im.w;
	int new_h = im.h;
	if (((float)w / im.w) < ((float)h / im.h)) {
		new_w = w;
		new_h = (im.h * w) / im.w;
	}
	else {
		new_h = h;
		new_w = (im.w * h) / im.h;
	}
	image resized = resize_image(im, new_w, new_h);
	image boxed = make_image(w, h, im.c);
	fill_image(boxed, .5);
	//int i;
	//for(i = 0; i < boxed.w*boxed.h*boxed.c; ++i) boxed.data[i] = 0;
	embed_image(resized, boxed, (w - new_w) / 2, (h - new_h) / 2);
	free_image(resized);
	return boxed;
}

image resize_max(image im, int max)
{
	TAT(TATPARMS);

	int w = im.w;
	int h = im.h;
	if(w > h){
		h = (h * max) / w;
		w = max;
	} else {
		w = (w * max) / h;
		h = max;
	}
	if(w == im.w && h == im.h) return copy_image(im);
	image resized = resize_image(im, w, h);
	return resized;
}

image resize_min(image im, int min)
{
	TAT(TATPARMS);

	int w = im.w;
	int h = im.h;
	if(w < h){
		h = (h * min) / w;
		w = min;
	} else {
		w = (w * min) / h;
		h = min;
	}
	if(w == im.w && h == im.h) return copy_image(im);
	image resized = resize_image(im, w, h);
	return resized;
}

image random_crop_image(image im, int w, int h)
{
	TAT(TATPARMS);

	int dx = rand_int(0, im.w - w);
	int dy = rand_int(0, im.h - h);
	image crop = crop_image(im, dx, dy, w, h);
	return crop;
}

image random_augment_image(image im, float angle, float aspect, int low, int high, int size)
{
	TAT(TATPARMS);

	aspect = rand_scale(aspect);
	int r = rand_int(low, high);
	int min = (im.h < im.w*aspect) ? im.h : im.w*aspect;
	float scale = (float)r / min;

	float rad = rand_uniform(-angle, angle) * 2.0 * M_PI / 360.;

	float dx = (im.w*scale/aspect - size) / 2.;
	float dy = (im.h*scale - size) / 2.;
	if(dx < 0) dx = 0;
	if(dy < 0) dy = 0;
	dx = rand_uniform(-dx, dx);
	dy = rand_uniform(-dy, dy);

	image crop = rotate_crop_image(im, rad, scale, size, size, dx, dy, aspect);

	return crop;
}

float three_way_max(float a, float b, float c)
{
	TAT(TATPARMS);

	return (a > b) ? ( (a > c) ? a : c) : ( (b > c) ? b : c) ;
}

float three_way_min(float a, float b, float c)
{
	TAT(TATPARMS);

	return (a < b) ? ( (a < c) ? a : c) : ( (b < c) ? b : c) ;
}

// http://www.cs.rit.edu/~ncs/color/t_convert.html
/// @todo #COLOR - cannot do HSV if channels > 3
void rgb_to_hsv(image im)
{
	TAT(TATPARMS);

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

/// @todo #COLOR - cannot do HSV if channels > 3
void hsv_to_rgb(image im)
{
	TAT(TATPARMS);

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

image grayscale_image(image im)
{
	TAT(TATPARMS);

	assert(im.c == 3);
	int i, j, k;
	image gray = make_image(im.w, im.h, 1);
	float scale[] = {0.587, 0.299, 0.114};
	for(k = 0; k < im.c; ++k){
		for(j = 0; j < im.h; ++j){
			for(i = 0; i < im.w; ++i){
				gray.data[i+im.w*j] += scale[k]*get_pixel(im, i, j, k);
			}
		}
	}
	return gray;
}

image threshold_image(image im, float thresh)
{
	TAT(TATPARMS);

	int i;
	image t = make_image(im.w, im.h, im.c);
	for(i = 0; i < im.w*im.h*im.c; ++i){
		t.data[i] = im.data[i]>thresh ? 1 : 0;
	}
	return t;
}

image blend_image(image fore, image back, float alpha)
{
	TAT(TATPARMS);

	assert(fore.w == back.w && fore.h == back.h && fore.c == back.c);
	image blend = make_image(fore.w, fore.h, fore.c);
	int i, j, k;
	for(k = 0; k < fore.c; ++k){
		for(j = 0; j < fore.h; ++j){
			for(i = 0; i < fore.w; ++i){
				float val = alpha * get_pixel(fore, i, j, k) +
					(1 - alpha)* get_pixel(back, i, j, k);
				set_pixel(blend, i, j, k, val);
			}
		}
	}
	return blend;
}

void scale_image_channel(image im, int c, float v)
{
	TAT(TATPARMS);

	int i, j;
	for(j = 0; j < im.h; ++j){
		for(i = 0; i < im.w; ++i){
			float pix = get_pixel(im, i, j, c);
			pix = pix*v;
			set_pixel(im, i, j, c, pix);
		}
	}
}

void translate_image_channel(image im, int c, float v)
{
	TAT(TATPARMS);

	int i, j;
	for(j = 0; j < im.h; ++j){
		for(i = 0; i < im.w; ++i){
			float pix = get_pixel(im, i, j, c);
			pix = pix+v;
			set_pixel(im, i, j, c, pix);
		}
	}
}

/// @todo #COLOR - needs to be fixed for 1 <= c <= N
void distort_image(image im, float hue, float sat, float val)
{
	TAT(TATPARMS);

	if (im.c >= 3)
	{
		rgb_to_hsv(im);
		scale_image_channel(im, 1, sat);
		scale_image_channel(im, 2, val);
		int i;
		for(i = 0; i < im.w*im.h; ++i){
			im.data[i] = im.data[i] + hue;
			if (im.data[i] > 1) im.data[i] -= 1;
			if (im.data[i] < 0) im.data[i] += 1;
		}
		hsv_to_rgb(im);
	}
	else
	{
		scale_image_channel(im, 0, val);
	}
	constrain_image(im);
}

/// @todo #COLOR - HSV no beuno
void random_distort_image(image im, float hue, float saturation, float exposure)
{
	TAT(TATPARMS);

	float dhue = rand_uniform_strong(-hue, hue);
	float dsat = rand_scale(saturation);
	float dexp = rand_scale(exposure);
	distort_image(im, dhue, dsat, dexp);
}


float bilinear_interpolate(image im, float x, float y, int c)
{
	TAT(TATPARMS);

	int ix = (int) floorf(x);
	int iy = (int) floorf(y);

	float dx = x - ix;
	float dy = y - iy;

	float val = (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c) +
		dy     * (1-dx) * get_pixel_extend(im, ix, iy+1, c) +
		(1-dy) *   dx   * get_pixel_extend(im, ix+1, iy, c) +
		dy     *   dx   * get_pixel_extend(im, ix+1, iy+1, c);
	return val;
}

void quantize_image(image im)
{
	TAT(TATPARMS);

	int size = im.c * im.w * im.h;
	int i;
	for (i = 0; i < size; ++i) im.data[i] = (int)(im.data[i] * 255) / 255. + (0.5 / 255);
}

void make_image_red(image im)
{
	TAT(TATPARMS);

	int r, c, k;
	for (r = 0; r < im.h; ++r) {
		for (c = 0; c < im.w; ++c) {
			float val = 0;
			for (k = 0; k < im.c; ++k) {
				val += get_pixel(im, c, r, k);
				set_pixel(im, c, r, k, 0);
			}
			for (k = 0; k < im.c; ++k) {
				//set_pixel(im, c, r, k, val);
			}
			set_pixel(im, c, r, 0, val);
		}
	}
}

image make_attention_image(int img_size, float *original_delta_cpu, float *original_input_cpu, int w, int h, int c, float alpha)
{
	TAT(TATPARMS);

	image attention_img;
	attention_img.w = w;
	attention_img.h = h;
	attention_img.c = c;
	attention_img.data = original_delta_cpu;
	make_image_red(attention_img);

	int k;
	float min_val = 999999, mean_val = 0, max_val = -999999;
	for (k = 0; k < img_size; ++k) {
		if (original_delta_cpu[k] < min_val) min_val = original_delta_cpu[k];
		if (original_delta_cpu[k] > max_val) max_val = original_delta_cpu[k];
		mean_val += original_delta_cpu[k];
	}
	mean_val = mean_val / img_size;
	float range = max_val - min_val;

	for (k = 0; k < img_size; ++k) {
		float val = original_delta_cpu[k];
		val = fabs(mean_val - val) / range;
		original_delta_cpu[k] = val * 4;
	}

	image resized = resize_image(attention_img, w / 4, h / 4);
	attention_img = resize_image(resized, w, h);
	free_image(resized);
	for (k = 0; k < img_size; ++k) attention_img.data[k] = attention_img.data[k]*alpha + (1-alpha)*original_input_cpu[k];

	return attention_img;
}

image resize_image(image im, int w, int h)
{
	TAT(TATPARMS);

	if (im.w == w && im.h == h)
	{
		return copy_image(im);
	}

	image resized = make_image(w, h, im.c);
	image part = make_image(w, im.h, im.c);
	int r, c, k;
	float w_scale = (float)(im.w - 1) / (w - 1);
	float h_scale = (float)(im.h - 1) / (h - 1);
	for(k = 0; k < im.c; ++k){
		for(r = 0; r < im.h; ++r){
			for(c = 0; c < w; ++c){
				float val = 0;
				if(c == w-1 || im.w == 1){
					val = get_pixel(im, im.w-1, r, k);
				} else {
					float sx = c*w_scale;
					int ix = (int) sx;
					float dx = sx - ix;
					val = (1 - dx) * get_pixel(im, ix, r, k) + dx * get_pixel(im, ix+1, r, k);
				}
				set_pixel(part, c, r, k, val);
			}
		}
	}
	for(k = 0; k < im.c; ++k){
		for(r = 0; r < h; ++r){
			float sy = r*h_scale;
			int iy = (int) sy;
			float dy = sy - iy;
			for(c = 0; c < w; ++c){
				float val = (1-dy) * get_pixel(part, c, iy, k);
				set_pixel(resized, c, r, k, val);
			}
			if(r == h-1 || im.h == 1) continue;
			for(c = 0; c < w; ++c){
				float val = dy * get_pixel(part, c, iy+1, k);
				add_pixel(resized, c, r, k, val);
			}
		}
	}

	free_image(part);
	return resized;
}


void test_resize(char *filename)
{
	TAT(TATPARMS);

	image im = load_image(filename, 0,0, 3);
	float mag = mag_array(im.data, im.w*im.h*im.c);
	printf("L2 Norm: %f\n", mag);
	image gray = grayscale_image(im);

	image c1 = copy_image(im);
	image c2 = copy_image(im);
	image c3 = copy_image(im);
	image c4 = copy_image(im);
	distort_image(c1, .1, 1.5, 1.5);
	distort_image(c2, -.1, .66666, .66666);
	distort_image(c3, .1, 1.5, .66666);
	distort_image(c4, .1, .66666, 1.5);


	show_image(im,   "Original");
	show_image(gray, "Gray");
	show_image(c1, "C1");
	show_image(c2, "C2");
	show_image(c3, "C3");
	show_image(c4, "C4");

	while(1){
		image aug = random_augment_image(im, 0, .75, 320, 448, 320);
		show_image(aug, "aug");
		free_image(aug);


		float exposure = 1.15;
		float saturation = 1.15;
		float hue = .05;

		image c = copy_image(im);

		float dexp = rand_scale(exposure);
		float dsat = rand_scale(saturation);
		float dhue = rand_uniform(-hue, hue);

		distort_image(c, dhue, dsat, dexp);
		show_image(c, "rand");
		printf("%f %f %f\n", dhue, dsat, dexp);
		free_image(c);
		wait_until_press_key_cv();
	}
}


image load_image(char * filename, int desired_width, int desired_height, int channels)
{
	TAT(TATPARMS);

	image out = load_image_cv(filename, channels);

	if (desired_height > 0 && desired_width > 0 && (desired_height != out.h || desired_width != out.w))
	{
		image resized = resize_image(out, desired_width, desired_height);
		free_image(out);
		out = resized;
	}
	return out;
}

image get_image_layer(image m, int l)
{
	TAT(TATPARMS);

	image out = make_image(m.w, m.h, 1);
	int i;
	for(i = 0; i < m.h*m.w; ++i){
		out.data[i] = m.data[i+l*m.h*m.w];
	}
	return out;
}

void print_image(image m)
{
	TAT(TATPARMS);

	int i, j, k;
	for(i =0 ; i < m.c; ++i){
		for(j =0 ; j < m.h; ++j){
			for(k = 0; k < m.w; ++k){
				printf("%.2lf, ", m.data[i*m.h*m.w + j*m.w + k]);
				if(k > 30) break;
			}
			printf("\n");
			if(j > 30) break;
		}
		printf("\n");
	}
	printf("\n");
}

image collapse_images_vert(image *ims, int n)
{
	TAT(TATPARMS);

	int color = 1;
	int border = 1;
	int h,w,c;
	w = ims[0].w;
	h = (ims[0].h + border) * n - border;
	c = ims[0].c;
	if(c != 3 || !color){
		w = (w+border)*c - border;
		c = 1;
	}

	image filters = make_image(w, h, c);
	int i,j;
	for(i = 0; i < n; ++i){
		int h_offset = i*(ims[0].h+border);
		image copy = copy_image(ims[i]);
		//normalize_image(copy);
		if(c == 3 && color){
			embed_image(copy, filters, 0, h_offset);
		}
		else{
			for(j = 0; j < copy.c; ++j){
				int w_offset = j*(ims[0].w+border);
				image layer = get_image_layer(copy, j);
				embed_image(layer, filters, w_offset, h_offset);
				free_image(layer);
			}
		}
		free_image(copy);
	}
	return filters;
}

image collapse_images_horz(image *ims, int n)
{
	TAT(TATPARMS);

	int color = 1;
	int border = 1;
	int h,w,c;
	int size = ims[0].h;
	h = size;
	w = (ims[0].w + border) * n - border;
	c = ims[0].c;
	if(c != 3 || !color){
		h = (h+border)*c - border;
		c = 1;
	}

	image filters = make_image(w, h, c);
	int i,j;
	for(i = 0; i < n; ++i){
		int w_offset = i*(size+border);
		image copy = copy_image(ims[i]);

		if(c == 3 && color){	///< @todo #COLOR
			embed_image(copy, filters, w_offset, 0);
		}
		else{
			for(j = 0; j < copy.c; ++j){
				int h_offset = j*(size+border);
				image layer = get_image_layer(copy, j);
				embed_image(layer, filters, w_offset, h_offset);
				free_image(layer);
			}
		}
		free_image(copy);
	}
	return filters;
}

void show_image_normalized(image im, const char *name)
{
	TAT(TATPARMS);

	image c = copy_image(im);
	normalize_image(c);
	show_image(c, name);
	free_image(c);
}

void show_images(image *ims, int n, char *window)
{
	TAT(TATPARMS);

	image m = collapse_images_vert(ims, n);

	normalize_image(m);
	save_image(m, window);
	show_image(m, window);
	free_image(m);
}

void free_image(image m)
{
	TAT(TATPARMS);

	if(m.data){
		free(m.data);
	}
}

// Fast copy data from a contiguous byte array into the image.
void copy_image_from_bytes(image im, char *pdata)
{
	TAT(TATPARMS);

	unsigned char *data = (unsigned char*)pdata;
	int i, k, j;
	int w = im.w;
	int h = im.h;
	int c = im.c;
	for (k = 0; k < c; ++k) {
		for (j = 0; j < h; ++j) {
			for (i = 0; i < w; ++i) {
				int dst_index = i + w * j + w * h*k;
				int src_index = k + c * i + c * w*j;
				im.data[dst_index] = (float)data[src_index] / 255.;
			}
		}
	}
}
