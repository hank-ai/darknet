#pragma once

#include "darknet.h"

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <string.h>
#include <math.h>

#include "image_opencv.hpp"

#include "box.hpp"
#ifdef __cplusplus
extern "C" {
#endif
/*
typedef struct {
    int w;
    int h;
    int c;
    float *data;
} image;
*/
float get_color(int c, int x, int max);
void flip_image(image a);
void draw_box(image a, int x1, int y1, int x2, int y2, float r, float g, float b);
void draw_box_width(image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b);
void draw_bbox(image a, box bbox, int w, float r, float g, float b);
void draw_label(image a, int r, int c, image label, const float *rgb);
void draw_weighted_label(image a, int r, int c, image label, const float *rgb, const float alpha);

/// This is still called from a few isolated places, but you're probably looking for @ref draw_detections_v3().
void draw_detections(image im, int num, float thresh, box *boxes, float **probs, char **names, int classes);

/// This is the function that is called to draw annotations on an image.
void draw_detections_v3(image im, detection *dets, int num, float thresh, char **names, int classes, int ext_output);

image image_distance(image a, image b);
void scale_image(image m, float s);
// image crop_image(image im, int dx, int dy, int w, int h);
image random_crop_image(image im, int w, int h);
image random_augment_image(image im, float angle, float aspect, int low, int high, int size);
void random_distort_image(image im, float hue, float saturation, float exposure);
void fill_image(image m, float s);
void letterbox_image_into(image im, int w, int h, image boxed);
image resize_max(image im, int max);
void translate_image(image m, float s);
void normalize_image(image p);
image rotate_image(image m, float rad);
void rotate_image_cw(image im, int times);
void embed_image(image source, image dest, int dx, int dy);
void distort_image(image im, float hue, float sat, float val);
void hsv_to_rgb(image im);
void constrain_image(image im);
void composite_3d(char *f1, char *f2, char *out, int delta);
int best_3d_shift_r(image a, image b, int min, int max);

image grayscale_image(image im);
image threshold_image(image im, float thresh);

image collapse_image_layers(image source, int border);
image collapse_images_horz(image *ims, int n);
image collapse_images_vert(image *ims, int n);

void show_image(image p, const char *name);
void show_image_normalized(image im, const char *name);
void save_image_png(image im, const char *name);
void save_image(image p, const char *name);
void show_images(image *ims, int n, char *window);
void show_image_layers(image p, char *name);
void show_image_collapsed(image p, char *name);

void print_image(image m);

//LIB_API image make_image(int w, int h, int c);
image make_random_image(int w, int h, int c);
image make_empty_image(int w, int h, int c);
image float_to_image_scaled(int w, int h, int c, float *data);
image float_to_image(int w, int h, int c, float *data);
image copy_image(image p);
void copy_image_inplace(image src, image dst);

/** Load the given image.  If both @p desired_width and @p desired_height have been set, then the image will be resized
 * to match those dimensions.  Otherwise, specify @p 0 (zero) to leave the image dimensions unchanged.
 */
image load_image(char * filename, int desired_width, int desired_height, int channels);

float bilinear_interpolate(image im, float x, float y, int c);

image get_image_layer(image m, int l);

void test_resize(char *filename);
#ifdef __cplusplus
}
#endif
