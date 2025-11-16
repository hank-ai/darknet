#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

#if defined(DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
#define _CRTDBG_MAP_ALLOC
#endif

// C headers
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <ciso646>

// C++ headers
#include <chrono>
#include <condition_variable>
#include <deque>
// #include <execution> => "error: identifier __builtin_ia32_sttilecfg is undefined" due to GCC 13.x and some versions of nvcc 12.x
#include <fstream>
#include <list>
#include <mutex>
#include <optional>
#include <random>
#include <regex>

// 3rd-party lib headers
#include <opencv2/opencv.hpp>

#ifdef DARKNET_OPENMP
#include <omp.h>
#endif

#if DARKNET_HAS_PROTOBUF
#include "onnx.proto3.pb.h"
#endif

/** If you're using some old software that expects the original @p C API in the %Darknet library,
 * then make sure you @p "#define DARKNET_INCLUDE_ORIGINAL_API" before you include darknet.h.
 *
 * Internally, %Darknet still uses the old @p C V2 API which is why it is defined in darknet_internal.hpp.
 */
#define DARKNET_INCLUDE_ORIGINAL_API	// internally we need the old C API

#include "darknet.h"					// the old C header
#include "darknet.hpp"					// the new C++ header
#include "darknet_version.h"			// version macros


namespace Darknet
{
	using VThreads = std::vector<std::thread>;

	/** This is used to help keep some state between calls to functions fill_network_boxes(), get_yolo_detections(), etc.
	 * We use the cache to track objects within the output array, so we don't have to walk over the entire array every
	 * time we need to find all the objects and bounding boxes.
	 *
	 * @since 2024-06-02
	 *
	 * @see @ref yolo_num_detections_v3() where the cache is populated (in yolo_layer.cpp)
	 * @see @ref get_yolo_detections_v3() where the cache is accessed and converted to bounding boxes
	 * @see @ref make_network_boxes_v3()
	 */
	struct Output_Object
	{
		int layer_index;	///< The layer index where this was found.
		int n;				///< What is "n"...the mask (anchor?) number?
		int i;				///< The entry index into the W x H output array for the given YOLO layer.
		int obj_index;		///< The index into the YOLO output array -- as obtained from @ref yolo_entry_index() -- which is used to get the objectness value.  E.g., a value of @p "l.output[obj_index] == 0.999f" would indicate that there is an object at this location.
	};
	using Output_Object_Cache = std::list<Output_Object>;

	class CfgLine;
	class CfgSection;
	class CfgFile;
	struct Tree;
}


/// @todo V2 what is @ref SECRET_NUM?  And/or move to a different header.
#define SECRET_NUM -1234


/// @todo V3 Need to get rid of @ref UNUSED_ENUM_TYPE.  And/or move to a different header.
typedef enum { UNUSED_DEF_VAL } UNUSED_ENUM_TYPE;


// activations.h
typedef enum {
	LOGISTIC, RELU, RELU6, RELIE, LINEAR, RAMP, TANH, PLSE, REVLEAKY, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU, GELU, SWISH, MISH, HARD_MISH, NORM_CHAN, NORM_CHAN_SOFTMAX, NORM_CHAN_SOFTMAX_MAXVAL
}ACTIVATION;

// parser.h
typedef enum
{
	IOU, GIOU, MSE, DIOU, CIOU
} IOU_LOSS;

// parser.h
typedef enum
{
	YOLO_CENTER			= 1 << 0,
	YOLO_LEFT_TOP		= 1 << 1,
	YOLO_RIGHT_BOTTOM	= 1 << 2
} YOLO_POINT;

// parser.h
typedef enum
{
	NO_WEIGHTS,
	PER_FEATURE,
	PER_CHANNEL
} WEIGHTS_TYPE_T;

// parser.h
typedef enum
{
	NO_NORMALIZATION,
	RELU_NORMALIZATION,
	SOFTMAX_NORMALIZATION
} WEIGHTS_NORMALIZATION_T;

// image.h
typedef enum
{
	PNG,	// supported
	BMP,	// not supported (will be saved as JPG)
	TGA,	// not supported (will be saved as JPG)
	JPG		// supported
} IMTYPE;

// activations.h
typedef enum
{
	MULT,
	ADD,
	SUB,
	DIV
} BINARY_ACTIVATION;

// blas.h
typedef struct contrastive_params
{
	float sim;
	float exp_sim;
	float P;
	int i, j;
	int time_step_i, time_step_j;
} contrastive_params;


// layer.h
typedef enum
{
	SSE,
	MASKED,
	L1,
	SEG,
	SMOOTH,
	WGAN
} COST_TYPE;

// layer.h
typedef struct update_args {
	int batch;
	float learning_rate;
	float momentum;
	float decay;
	int adam;
	float B1;
	float B2;
	float eps;
	int t;
} update_args;


// network.h
typedef enum
{
	CONSTANT,
	STEP,
	EXP,
	POLY,
	STEPS,
	SIG,
	RANDOM,
	SGDR
} learning_rate_policy;


// box.h

// box.h
typedef struct boxabs {
	float left, right, top, bot;
} boxabs;

// box.h
typedef struct dxrep {
	float dt, db, dl, dr;
} dxrep;

// box.h
typedef struct ious {
	float iou, giou, diou, ciou;
	dxrep dx_iou;
	dxrep dx_giou;
} ious;


// network.c -batch inference
typedef struct det_num_pair {
	int num;
	Darknet::Detection *dets;
} det_num_pair, *pdet_num_pair;

// matrix.h
typedef struct matrix
{
	int rows;
	int cols;
	float **vals;
} matrix;

// data.h
typedef struct data
{
	int w;
	int h;
	matrix X; // Note uppercase.  Why?  I have no idea.
	matrix y;
	int shallow;
	int *num_boxes;
	Darknet::Box **boxes;
	uint64_t nanoseconds_to_load;	///< record how much time it took to load all images
} data;


/** Things that we can do on a secondary thread.
	* @see @ref Darknet::load_single_image_data()
	* @see @ref load_args::type
	*/
typedef enum
{
	DETECTION_DATA,
	IMAGE_DATA,		///< causes @ref Darknet::load_image() and @ref Darknet::resize_image() to be called
	LETTERBOX_DATA,	///< causes @ref Darknet::load_image() and @ref Darknet::letterbox_image() to be called
} data_type;


/** Used when a secondary thread is created to load things, such as images.
	* @see @ref Darknet::load_single_image_data()
	* @see @ref data_type
	*/
typedef struct load_args {
	int threads;
	char **paths;
	const char *path;
	int n; ///< number of images, or batch size?
	int m; ///< maximum number of images?
	int h;
	int w;
	int c;	///< Number of channels, typically 3 for RGB
	int num_boxes;
	int truth_size;
	int classes;
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
	float angle;
	float aspect;
	float saturation;
	float exposure;
	float hue;
	data *d;
	Darknet::Image *im;
	Darknet::Image *resized;
	data_type type;
} load_args;


// data.h
typedef struct box_label {
	int id;
	int track_id;
	float x, y, w, h;
	float left, right, top, bottom;
} box_label;


// layer.h
void free_layer_custom(Darknet::Layer & l, int keep_cudnn_desc);
void free_layer(Darknet::Layer & l);

// dark_cuda.h
void cuda_pull_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array_async(float *x_gpu, float *x, size_t n);
void *cuda_get_context();

// utils.h
void top_k(float *a, int n, int k, int *index);

// gemm.h
void init_cpu();

/** Count the number of objects found in the current image.  Only looks at the YOLO layer at @p index within the
 * network.  Starting with V3 JAZZ, this will also populate (appends, does not clear!) the object cache with the
 * location of all objects found so we don't have to look through the entire YOLO output again when creating the
 * boxes.
 */
int yolo_num_detections_v3(Darknet::Network * net, const int index, const float thresh, Darknet::Output_Object_Cache & cache);

/// Convert everything we've detected into bounding boxes and confidence scores for each class.
int get_yolo_detections_v3(Darknet::Network * net, int w, int h, int netw, int neth, float thresh, int *map, int relative, Darknet::Detection *dets, int letter, Darknet::Output_Object_Cache & cache);

#include "darknet_args_and_parms.hpp"
#include "darknet_cfg_and_state.hpp"
#include "darknet_enums.hpp"
#include "list.hpp"
#include "matrix.hpp"
#include "dark_cuda.hpp"
#include "darknet_layers.hpp"
#include "darknet_format_and_colour.hpp"
#include "darknet_utils.hpp"
#include "darknet_image.hpp"
#include "darknet_network.hpp"
#include "image_opencv.hpp"
#include "Timing.hpp"
#include "darknet_cfg.hpp"
#include "box.hpp"
#include "blas.hpp"
#include "utils.hpp"
#include "weights.hpp"
#include "data.hpp"
#include "option_list.hpp"
#include "dark_cuda.hpp"
#include "tree.hpp"
#include "activations.hpp"
#include "dump.hpp"

#if DARKNET_GPU_ROCM
#include "amd_rocm.hpp"
#endif
