#pragma once

#ifndef __cplusplus
#error "The darknet project requires the use of a C++ compiler."
#endif


#include "darknet_internal.hpp"


namespace Darknet
{
	/** Things that we can do on a secondary thread.
	 * @see @ref load_thread()
	 * @see @ref load_args.type
	 */
	enum class EDataType
	{
		CLASSIFICATION_DATA,
		DETECTION_DATA,
//		CAPTCHA_DATA, unused
		REGION_DATA,
		IMAGE_DATA, ///< causes @ref load_image() and @ref resize_image() to be called
		COMPARE_DATA,
		WRITING_DATA,
//		SWAG_DATA, unused
		TAG_DATA,
		OLD_CLASSIFICATION_DATA,
//		STUDY_DATA, unused
//		DET_DATA, unused
		SUPER_DATA,
		LETTERBOX_DATA,
//		REGRESSION_DATA, unused
//		SEGMENTATION_DATA, unused
//		INSTANCE_DATA, unused
//		ISEG_DATA unused
	};

	struct Data
	{
		int w;
		int h;
		matrix X; ///< why is it uppercase @p X when the rest of the members are lowercase?
		matrix y;
		int shallow;	///< @todo need to understand what this one is for
		int *num_boxes;	///< @todo need to understand what this one is for
		box **boxes;	///< @todo need to understand what this one is for
	};

	/** Used when a secondary thread is created to load things, such as images.
	 * @see @ref load_image()
	 * @see @ref data_type
	 */
	struct LoadArgs
	{
		int threads;	///< the number of threads to start
		char **paths;
		char *path;
		int n;
		int m;
		char **labels;
		int h;
		int w;
		int c;	///< Number of channels, typically 3 for RGB
		int out_w;
		int out_h;
		int nh;
		int nw;
		int num_boxes;
		int truth_size;
		int min, max, size;
		int classes;
		int background;
		int scale;
		int center;
		int coords;
		int mini_batch;
		int track;
		int augment_speed;
		int letter_box;
		int mosaic_bound;
		int show_imgs;
		int contrastive;
		int contrastive_jit_flip;
		int contrastive_color;
		float jitter;
		float resize;
		int flip;
		int gaussian_noise;
		int blur;
		int mixup;
		float label_smooth_eps;
		float angle;
		float aspect;
		float saturation;
		float exposure;
		float hue;
		data *d; ///< width, height, x, y, pointer to boxes?
		image *im;
		image *resized;
		data_type type;
		tree *hierarchy;
	};

	void free_data(data & d);

	/** This runs as a @p std::thread.  It is started by the main thread during training and ensures the data-loading
	 * threads are running.
	 *
	 * This was originally called @p load_threads() and used @p pthread, but has since been re-written to use C++11.
	 *
	 * @see @ref stop_image_loading_threads()
	 *
	 * @since 2024-03-31
	 */
	void run_image_loading_threads(load_args args);


	/** Starts the thread the conrols all of the permanent image loading threads.
	 *
	 * This was originally called @p load_data() and used @p pthread, but has since been re-written to use C++11.
	 *
	 * @todo delete this function since it is no longer referenced
	 *
	 * @since 2024-03-31
	 */
	std::thread to_be_deleted_start_permanent_image_loading_threads(const load_args & args);


	/** Stop and join the image loading threads started in @ref Darknet::rn_image_loading_threads().
	 *
	 * This was originally called @p free_load_threads() and used @p pthread, but has since been re-written to use C++11.
	 *
	 * @since 2024-04-02
	 */
	void stop_image_loading_threads();

	/** Run the permanent thread image loading loop.  This is started by @ref run_image_loading_threads(),
	 * and is stopped by @ref stop_image_loading_threads().
	 *
	 * This was originally called @p run_thread_loop() and used @p pthread, but has since been re-written to use C++11.
	 *
	 * @since 2024-04-02
	 */
	void image_loading_loop(load_args args, const int idx);


	/** Load the given image data as described by the @p load_args parameter.  This is typically used to load a single
	 * image on a secondary thread.
	 *
	 * This was originally called @p load_thread().
	 *
	 * @since 2024-04-02
	 */
	void load_single_image_data(load_args args);
}
