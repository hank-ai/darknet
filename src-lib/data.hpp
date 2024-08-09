#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

#include "darknet_internal.hpp"


data load_data_detection(int n, char **paths, int m, int w, int h, int c, int boxes, int truth_size, int classes, int use_flip, int gaussian_noise, int use_blur, int use_mixup, float jitter, float resize, float hue, float saturation, float exposure, int mini_batch, int track, int augment_speed, int letter_box, int mosaic_bound, int contrastive, int contrastive_jit_flip, int contrastive_color, int show_imgs);
box_label *read_boxes(char *filename, int *n);
list *get_paths(char *filename);

data get_data_part(data d, int part, int total);
void get_next_batch(data d, int n, int offset, float *X, float *y);
data concat_data(data d1, data d2);


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
	void image_loading_loop(const int idx, load_args args);


	/** Load the given image data as described by the @p load_args parameter.  This is typically used to load images on a
	 * secondary thread, such as @ref image_loading_loop().
	 *
	 * @note The name is misleading.  While I initially thought a single image at a time was being loaded, the @p args.n
	 * argument is used to describe the number of images that will be loaded together.  This will typically be the batch
	 * size divided by the number of worker threads (default is 6 threads).
	 *
	 * This was originally called @p load_thread().
	 *
	 * @since 2024-04-02
	 */
	void load_single_image_data(load_args args);


	/// Frees the "data buffer" used to load images.
	void free_data(data & d);
}
