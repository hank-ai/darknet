#pragma once

#ifndef __cplusplus
#error "The darknet project requires the use of a C++ compiler."
#endif


//#include "darknet_internal.hpp"


// For future use when we convert darknet.h to darknet.hpp.  For now,
// you should be including either darknet.h or darknet_internal.hpp.

#if 0

namespace Darknet
{
	/// Things that we can do on a secondary thread.
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

	/// Used when a secondary thread is created to load things, such as images.
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
}
#endif
