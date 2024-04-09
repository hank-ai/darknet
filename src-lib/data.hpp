#pragma once

#ifndef __cplusplus
#error "The darknet project requires the use of a C++ compiler."
#endif

#include <thread>
#include "darknet.h"
#include "list.hpp"


void print_letters(float *pred, int n);
data load_data_old(char **paths, int n, int m, char **labels, int k, int w, int h, int c);
data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes, int use_flip, int gaussian_noise, int use_blur, int use_mixup,
    float jitter, float resize, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip, int contrastive_color, int show_imgs);
data load_data_tag(char **paths, int n, int m, int k, int use_flip, int min, int max, int w, int h, int c, float angle, float aspect, float hue, float saturation, float exposure);
matrix load_image_augment_paths(char **paths, int n, int use_flip, int min, int max, int w, int h, int c, float angle, float aspect, float hue, float saturation, float exposure, int contrastive);
data load_data_super(char **paths, int n, int m, int w, int h, int c, int scale);
data load_data_augment(char **paths, int n, int m, char **labels, int k, tree *hierarchy, int use_flip, int min, int max, int w, int h, int c, float angle,
    float aspect, float hue, float saturation, float exposure, int use_mixup, int use_blur, int show_imgs, float label_smooth_eps, int contrastive);
data load_go(char *filename);

box_label *read_boxes(char *filename, int *n);
data load_cifar10_data(char *filename);
data load_all_cifar10();

data load_data_writing(char** paths, int n, int m, int w, int h, int c, int out_w, int out_h);
list *get_paths(char *filename);
char **get_labels(char *filename);
char **get_labels_custom(char *filename, int *size);
void get_random_batch(data d, int n, float *X, float *y);
data get_data_part(data d, int part, int total);
data get_random_data(data d, int num);
void get_next_batch(data d, int n, int offset, float *X, float *y);
data load_categorical_data_csv(char *filename, int target, int k);
void normalize_data_rows(data d);
void scale_data_rows(data d, float s);
void translate_data_rows(data d, float s);
void randomize_data(data d);
data *split_data(data d, int part, int total);
data concat_data(data d1, data d2);
data concat_datas(data *d, int n);
void fill_truth(char *path, char **labels, int k, float *truth);
void fill_truth_smooth(char *path, char **labels, int k, float *truth, float label_smooth_eps);


namespace Darknet
{
	/** This runs as a @p std::thread.  It is started by the main thread during training and ensures the data-loading
	 * threads are running.  This starts the thread that controls all of the permanent image loading threads.
	 *
	 * This was originally called @p load_threads() and used @p pthread, but has since been re-written to use C++11.
	 *
	 * @see @ref stop_image_loading_threads()
	 *
	 * @since 2024-03-31
	 */
	void run_image_loading_control_thread(load_args args);


	/** Stop and join the image loading threads started in @ref Darknet::run_image_loading_control_thread().
	 *
	 * This was originally called @p free_load_threads() and used @p pthread, but has since been re-written to use C++11.
	 *
	 * @since 2024-04-02
	 */
	void stop_image_loading_threads();


	/** Run the permanent thread image loading loop.  This is started by @ref Darknet::run_image_loading_control_thread(),
	 * and is stopped by @ref Darknet::stop_image_loading_threads().
	 *
	 * This was originally called @p run_thread_loop() and used @p pthread, but has since been re-written to use C++11.
	 *
	 * @since 2024-04-02
	 */
	void image_loading_loop(const int idx);


	/** Load the given image data as described by the @p load_args parameter.  This is typically used to load a single
	 * image on a secondary thread.
	 *
	 * @warning The @p args are currently dynamically allocated by the caller, and then freed at the
	 * bottom of this function.  This is carry-over from the @p pthread and @p C days, and will need to be fixed since
	 * there is zero need to dynamically allocate and free this structure every time we load a new image.
	 *
	 * This was originally called @p load_thread().
	 *
	 * @since 2024-04-02
	 */
	void load_single_image_data(load_args args);


	/// Frees the "data buffer" used to load images.
	void free_data(data & d);
}
