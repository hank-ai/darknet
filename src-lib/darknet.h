#pragma once

#if defined(_MSC_VER) && _MSC_VER < 1900
#define inline __inline
#endif

#if defined(DEBUG) && !defined(_CRTDBG_MAP_ALLOC)
#define _CRTDBG_MAP_ALLOC
#endif

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>

/// @todo what is this?
#define SECRET_NUM -1234

typedef enum { UNUSED_DEF_VAL } UNUSED_ENUM_TYPE;

#ifdef GPU

#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>

#ifdef CUDNN
#include <cudnn.h>
#endif  // CUDNN
#endif  // GPU

namespace Darknet
{
	/// The @p layer structure has been renamed and moved to darknet_layer.hpp.
	struct Layer;

	/// The @p network_state structure has been renamed and moved to network.hpp.
	struct NetworkState;

	/// The @p image structure has been renamed and moved to darknet_image.hpp.
	struct Image;

	/** Some C++ structures we'd like to insert into the C "Network".  Needs to be a @p void* pointer so older C code can
	 * continue using the Network without causing any problems.
	 */
	struct NetworkDetails;

	/// The @p network structure has been renamed and moved to darknet_network.hpp.
	struct Network;
}

#ifdef __cplusplus
extern "C" {
#endif


struct detection;
typedef struct detection detection;

struct load_args;
typedef struct load_args load_args;

struct data;
typedef struct data data;


// tree.h
typedef struct tree {
	int *leaf;
	int n;
	int *parent;
	int *child;
	int *group;
	char **name;

	int groups;
	int *group_size;
	int *group_offset;
} tree;


// activations.h
typedef enum {
	LOGISTIC, RELU, RELU6, RELIE, LINEAR, RAMP, TANH, PLSE, REVLEAKY, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU, GELU, SWISH, MISH, HARD_MISH, NORM_CHAN, NORM_CHAN_SOFTMAX, NORM_CHAN_SOFTMAX_MAXVAL
}ACTIVATION;

// parser.h
typedef enum {
	IOU, GIOU, MSE, DIOU, CIOU
} IOU_LOSS;

// parser.h
typedef enum {
	DEFAULT_NMS, GREEDY_NMS, DIOU_NMS, CORNERS_NMS
} NMS_KIND;

// parser.h
typedef enum {
	YOLO_CENTER = 1 << 0, YOLO_LEFT_TOP = 1 << 1, YOLO_RIGHT_BOTTOM = 1 << 2
} YOLO_POINT;

// parser.h
typedef enum {
	NO_WEIGHTS, PER_FEATURE, PER_CHANNEL
} WEIGHTS_TYPE_T;

// parser.h
typedef enum {
	NO_NORMALIZATION, RELU_NORMALIZATION, SOFTMAX_NORMALIZATION
} WEIGHTS_NORMALIZATION_T;

// image.h
typedef enum{
	PNG,	// supported
	BMP,	// not supported (will be saved as JPG)
	TGA,	// not supported (will be saved as JPG)
	JPG		// supported
} IMTYPE;

// activations.h
typedef enum{
	MULT, ADD, SUB, DIV
} BINARY_ACTIVATION;

// blas.h
typedef struct contrastive_params {
	float sim;
	float exp_sim;
	float P;
	int i, j;
	int time_step_i, time_step_j;
} contrastive_params;


// layer.h
typedef enum{
	SSE, MASKED, L1, SEG, SMOOTH,WGAN
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
typedef enum {
	CONSTANT, STEP, EXP, POLY, STEPS, SIG, RANDOM, SGDR
} learning_rate_policy;


// box.h
typedef struct box {
	float x, y, w, h;
} box;

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


// box.h
typedef struct detection{
	box bbox; ///< bounding boxes are normalized (between 0.0f and 1.0f)
	int classes;
	int best_class_idx;
	float *prob;
	float *mask;
	float objectness;
	int sort_class;
	float *uc; ///< Gaussian_YOLOv3 - tx,ty,tw,th uncertainty
	int points; ///< bit-0 - center, bit-1 - top-left-corner, bit-2 - bottom-right-corner
	float *embeddings;  ///< embeddings for tracking
	int embedding_size;
	float sim;
	int track_id;
} detection;

// network.c -batch inference
typedef struct det_num_pair {
	int num;
	detection *dets;
} det_num_pair, *pdet_num_pair;

// matrix.h
typedef struct matrix
{
	int rows;
	int cols;
	float **vals;
} matrix;

// data.h
typedef struct data {
	int w;
	int h;
	matrix X; // Note uppercase.  Why?  I have no idea.
	matrix y;
	int shallow;
	int *num_boxes;
	box **boxes;
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
	char *path;
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

// -----------------------------------------------------


// box.h
void do_nms_sort(detection *dets, int total, int classes, float thresh);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void diounms_sort(detection *dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1);

// layer.h
void free_layer_custom(Darknet::Layer & l, int keep_cudnn_desc);
void free_layer(Darknet::Layer & l);

// dark_cuda.h
void cuda_pull_array(float *x_gpu, float *x, size_t n);
void cuda_pull_array_async(float *x_gpu, float *x, size_t n);
void cuda_set_device(int n);
void *cuda_get_context();

// utils.h
void free_ptrs(void **ptrs, int n);
void top_k(float *a, int n, int k, int *index);

// tree.h
tree *read_tree(const char *filename);

// gemm.h
void init_cpu();

#ifdef __cplusplus
}
#endif  // __cplusplus








// The "new" (V3, July 2024) Darknet/YOLO API starts here.

#ifdef __cplusplus
extern "C" {
#endif

/// @todo V3 rename to DarknetNetworkPtr?
	///< An opaque pointer to a @ref Darknet::Network object, without needing to expose the internals of the network structure.
	typedef void* NetworkPtr;

	/// This is the @p C equivalent to @ref Darknet::set_verbose().
	void darknet_set_verbose(const bool flag);

	/// This is the @p C equivalent to @ref Darknet::set_trace().
	void darknet_set_trace(const bool flag);

	/// This is the @p C equivalent to @ref Darknet::set_gpu_index().
	void darknet_set_gpu_index(int idx);

	/// This is the @p C equivalent to @ref Darknet::set_detection_threshold().
	void darknet_set_detection_threshold(NetworkPtr ptr, float threshold);

	/// This is the @p C equivalent to @ref Darknet::set_non_maximal_suppression_threshold().
	void darknet_set_non_maximal_suppression_threshold(NetworkPtr ptr, float threshold);

	/// This is the @p C equivalent to @ref Darknet::fix_out_of_bound_values().
	void darknet_fix_out_of_bound_values(NetworkPtr ptr, const bool toggle);

	/// This is the @p C equivalent to @ref Darknet::network_dimensions().
	void darknet_network_dimensions(NetworkPtr ptr, int * w, int * h, int * c);

	/// This is the @p C equivalent to @ref Darknet::load_neural_network().
	NetworkPtr darknet_load_neural_network(const char * const cfg_filename, const char * const names_filename, const char * const weights_filename);

#ifdef __cplusplus
}
#endif  // __cplusplus
