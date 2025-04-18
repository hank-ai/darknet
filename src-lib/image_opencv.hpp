#pragma once

#include "darknet_internal.hpp"


/** Load the given image using OpenCV.  Automatically converts the image from the usual OpenCV BGR format to RGB for
 * use in Darknet.
 *
 * @see @ref Darknet::load_image()
 */
cv::Mat load_rgb_mat_image(const std::string & filename, int flag);

void show_image_cv(Darknet::Image p, const char *name);

// Draw Detection
void draw_detections_cv_v3(cv::Mat show_img, Darknet::Detection *dets, int num, float thresh, char **names, int classes, int ext_output);

// Data augmentation
Darknet::Image image_data_augmentation(cv::Mat mat, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float dhue, float dsat, float dexp,
    int gaussian_noise, int blur, int num_boxes, int truth_size, float *truth);

// blend two images with (alpha and beta)
void blend_images_cv(Darknet::Image new_img, float alpha, Darknet::Image old_img, float beta);

// Show Anchors
void show_anchors(int number_of_boxes, int num_of_clusters, float *rel_width_height_array, model anchors_data, int width, int height);
