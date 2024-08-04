#pragma once

#include "darknet.h"

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

#include "image_opencv.hpp"

#include "box.hpp"


namespace Darknet
{
	struct Image
	{
		int w;
		int h;
		int c;
		float *data;
	};

	/// Generate some "random" colour value to use.  Mostly used for labels and charts.
	float get_color(int c, int x, int max);

	/// Flip image left <-> right.
	void flip_image(Darknet::Image a);
}


void draw_box_width(Darknet::Image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
void draw_bbox(Darknet::Image a, box bbox, int w, float r, float g, float b);
void draw_label(Darknet::Image a, int r, int c, Darknet::Image label, const float *rgb);
void draw_weighted_label(Darknet::Image a, int r, int c, Darknet::Image label, const float *rgb, const float alpha);

/// This is still called from a few isolated places, but you're probably looking for @ref draw_detections_v3().
void draw_detections(Darknet::Image im, int num, float thresh, box *boxes, float **probs, char **names, int classes);

/// This is the function that is called to draw annotations on an image.
void draw_detections_v3(Darknet::Image im, detection *dets, int num, float thresh, char **names, int classes, int ext_output);

Darknet::Image image_distance(Darknet::Image a, Darknet::Image b);
void scale_image(Darknet::Image m, float s);
// image crop_image(image im, int dx, int dy, int w, int h);
Darknet::Image random_crop_image(Darknet::Image im, int w, int h);
Darknet::Image random_augment_image(Darknet::Image im, float angle, float aspect, int low, int high, int size);
void random_distort_image(Darknet::Image im, float hue, float saturation, float exposure);
void fill_image(Darknet::Image m, float s);
void letterbox_image_into(Darknet::Image im, int w, int h, Darknet::Image boxed);
Darknet::Image resize_max(Darknet::Image im, int max);
void translate_image(Darknet::Image m, float s);
void normalize_image(Darknet::Image p);
Darknet::Image rotate_image(Darknet::Image m, float rad);
void rotate_image_cw(Darknet::Image im, int times);
void embed_image(Darknet::Image source, Darknet::Image dest, int dx, int dy);
void distort_image(Darknet::Image im, float hue, float sat, float val);
void hsv_to_rgb(Darknet::Image im);
void constrain_image(Darknet::Image im);
void composite_3d(char *f1, char *f2, char *out, int delta);
int best_3d_shift_r(Darknet::Image a, Darknet::Image b, int min, int max);

Darknet::Image grayscale_image(Darknet::Image im);
Darknet::Image threshold_image(Darknet::Image im, float thresh);

Darknet::Image collapse_image_layers(Darknet::Image source, int border);
Darknet::Image collapse_images_horz(Darknet::Image *ims, int n);
Darknet::Image collapse_images_vert(Darknet::Image *ims, int n);

void show_image(Darknet::Image p, const char * name);
void show_image_normalized(Darknet::Image im, const char * name);
void save_image_png(Darknet::Image im, const char * name);
void save_image(Darknet::Image p, const char * name);
void show_images(Darknet::Image *ims, int n, const char * window);
void show_image_layers(Darknet::Image p, const char * name);
void show_image_collapsed(Darknet::Image p, const char * name);

void print_image(Darknet::Image m);

//LIB_API image make_image(int w, int h, int c);
Darknet::Image make_random_image(int w, int h, int c);
Darknet::Image make_empty_image(int w, int h, int c);
Darknet::Image float_to_image_scaled(int w, int h, int c, float *data);
Darknet::Image float_to_image(int w, int h, int c, float *data);
Darknet::Image copy_image(Darknet::Image p);
void copy_image_inplace(Darknet::Image src, Darknet::Image dst);

/** Load the given image.  If both @p desired_width and @p desired_height have been set, then the image will be resized
 * to match those dimensions.  Otherwise, specify @p 0 (zero) to leave the image dimensions unchanged.
 */
Darknet::Image load_image(char * filename, int desired_width, int desired_height, int channels);

float bilinear_interpolate(Darknet::Image im, float x, float y, int c);

Darknet::Image get_image_layer(Darknet::Image m, int l);

void test_resize(char *filename);
