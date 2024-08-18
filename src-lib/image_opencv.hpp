#pragma once

#include "darknet_internal.hpp"


/// Hide a C++ cv::Mat object as a C style @p void* pointer.
typedef void* mat_cv;

/// Hide a C++ cv::VideoCapture object as a C style @p void* pointer.
typedef void* cap_cv;

/// Hide a C++ cv::VideoWriter object as a C style @p void* pointer.
typedef void* write_cv;


/** Load the given image using OpenCV and return it as a @p "C" style @ref mat_cv @p void* pointer.  Converts the image
 * from the usual OpenCV BGR format to RGB.  Remember to free the image with @ref release_mat().
 */
mat_cv * load_image_mat_cv(const char * const filename, int flag);


/** Similar to @ref load_image_mat_cv() but is explicit about the image channels and returns a Darknet-style
 * @ref Darknet::Image.
 *
 * Channels should be @p 0 (colour), @p 1 (grayscale) or @p 3 (colour).  This uses @ref load_image_mat_cv() so the
 * channels will be converted from BGR to RGB.
 */
Darknet::Image load_image_cv(char *filename, int channels);



int get_width_mat(mat_cv *mat);
int get_height_mat(mat_cv *mat);

/// Frees the @p cv::Mat object allocated in @ref load_image_mat_cv().
void release_mat(mat_cv **mat);

// Window
void create_window_cv(char const* window_name, int full_screen, int width, int height);
void show_image_cv(Darknet::Image p, const char *name);
void show_image_mat(mat_cv *mat_ptr, const char *name);

// Image Saving
void save_cv_png(mat_cv *img, const char *name);
void save_cv_jpg(mat_cv *img, const char *name);

// Draw Detection
void draw_detections_cv_v3(mat_cv* show_img, Darknet::Detection *dets, int num, float thresh, char **names, int classes, int ext_output);

// Data augmentation
Darknet::Image image_data_augmentation(mat_cv* mat, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float dhue, float dsat, float dexp,
    int gaussian_noise, int blur, int num_boxes, int truth_size, float *truth);

// blend two images with (alpha and beta)
void blend_images_cv(Darknet::Image new_img, float alpha, Darknet::Image old_img, float beta);

// bilateralFilter bluring
Darknet::Image blur_image(Darknet::Image src_img, int ksize);

// Show Anchors
void show_anchors(int number_of_boxes, int num_of_clusters, float *rel_width_height_array, model anchors_data, int width, int height);

void show_opencv_info();
