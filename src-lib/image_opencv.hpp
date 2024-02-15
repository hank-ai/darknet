#pragma once

#include "image.hpp"
#include "matrix.hpp"

#include <opencv2/opencv.hpp>

#ifdef __cplusplus
extern "C" {
#endif


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


/** Similar to @ref load_image_mat_cv() but is explicit about the image channels and returns a Darknet-style @ref image.
 * Channels should be @p 0 (colour), @p 1 (grayscale) or @p 3 (colour).  This uses @ref load_image_mat_cv() so the
 * channels will be converted fro BGR to RGB.
 */
image load_image_cv(char *filename, int channels);



int get_width_mat(mat_cv *mat);
int get_height_mat(mat_cv *mat);

/// Frees the @p cv::Mat object allocated in @ref load_image_mat_cv().
void release_mat(mat_cv **mat);

image mat_to_image(cv::Mat mat);

image mat_to_image_cv(mat_cv *mat);

cv::Mat image_to_mat(image img);

// Window
void create_window_cv(char const* window_name, int full_screen, int width, int height);
void resize_window_cv(char const* window_name, int width, int height);
void move_window_cv(char const* window_name, int x, int y);
void destroy_all_windows_cv();
int wait_key_cv(int delay);
int wait_until_press_key_cv();
//void make_window(char *name, int w, int h, int fullscreen); -- use create_window_cv() instead
void show_image_cv(image p, const char *name);
//void show_image_cv_ipl(mat_cv *disp, const char *name);
void show_image_mat(mat_cv *mat_ptr, const char *name);

// Video Writer
write_cv *create_video_writer(char *out_filename, char c1, char c2, char c3, char c4, int fps, int width, int height, int is_color);
void write_frame_cv(write_cv *output_video_writer, mat_cv *mat);
void release_video_writer(write_cv **output_video_writer);


// Video Capture
cap_cv* get_capture_video_stream(const char *path);
cap_cv* get_capture_webcam(int index);
void release_capture(cap_cv* cap);

mat_cv* get_capture_frame_cv(cap_cv *cap);
int get_stream_fps_cpp_cv(cap_cv *cap);
double get_capture_property_cv(cap_cv *cap, int property_id);
double get_capture_frame_count_cv(cap_cv *cap);
int set_capture_property_cv(cap_cv *cap, int property_id, double value);
int set_capture_position_frame_cv(cap_cv *cap, int index);

// ... Video Capture
image get_image_from_stream_cpp(cap_cv *cap);
image get_image_from_stream_resize(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close);
image get_image_from_stream_letterbox(cap_cv *cap, int w, int h, int c, mat_cv** in_img, int dont_close);
void consume_frame(cap_cv *cap);

// Image Saving
void save_cv_png(mat_cv *img, const char *name);
void save_cv_jpg(mat_cv *img, const char *name);

// Draw Detection
void draw_detections_cv_v3(mat_cv* show_img, detection *dets, int num, float thresh, char **names, int classes, int ext_output);

/// Draw the intial Loss & Accuracy chart.  This is called once at the very start.
mat_cv* draw_initial_train_chart(char *windows_name, float max_img_loss, int max_batches, int number_of_lines, int img_size, int dont_show, char* chart_path);

/// Update the Loss & Accuracy chart with the given information.  This is called repeatedly as more data is produced during training.
void update_train_loss_chart(char *windows_name, mat_cv* img, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches,
    float precision, int draw_precision, char *accuracy_name, float contr_acc, int dont_show, int mjpeg_port, double time_remaining);

// Data augmentation
image image_data_augmentation(mat_cv* mat, int w, int h,
    int pleft, int ptop, int swidth, int sheight, int flip,
    float dhue, float dsat, float dexp,
    int gaussian_noise, int blur, int num_boxes, int truth_size, float *truth);

// blend two images with (alpha and beta)
void blend_images_cv(image new_img, float alpha, image old_img, float beta);

// bilateralFilter bluring
image blur_image(image src_img, int ksize);

// draw objects for Adversarial attacks
void cv_draw_object(image sized, float *truth_cpu, int max_boxes, int num_truth, int *it_num_set, float *lr_set, int *boxonly, int classes, char **names);

// Show Anchors
void show_anchors(int number_of_boxes, int num_of_clusters, float *rel_width_height_array, model anchors_data, int width, int height);

void show_opencv_info();


#ifdef __cplusplus
}
#endif
