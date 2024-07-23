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
}

#ifdef __cplusplus
extern "C" {
#endif


struct image;
typedef struct image image;

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


/// @todo This enum will eventually be replaced by @ref Darknet::ELayerType
typedef enum {
	CONVOLUTIONAL,
	CONNECTED,
	MAXPOOL,
	LOCAL_AVGPOOL,
	SOFTMAX,
	DROPOUT,
	ROUTE,
	COST,
	AVGPOOL,
	SHORTCUT,
	SCALE_CHANNELS,
	SAM,
	RNN,
	LSTM,
	CRNN,
	NETWORK,
	REGION,
	YOLO,
	GAUSSIAN_YOLO,
	REORG,
	UPSAMPLE,	// or downsample if l.reverse=1
	EMPTY,
	BLANK,
	CONTRASTIVE,
	LAYER_LAST_IDX = CONTRASTIVE,
} LAYER_TYPE;

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

// network.h
typedef struct network {
	int n;	///< the number of layers in the network
	int batch;
	uint64_t *seen;
	float *badlabels_reject_threshold;
	float *delta_rolling_max;
	float *delta_rolling_avg;
	float *delta_rolling_std;
	int weights_reject_freq;
	int equidistant_point;
	float badlabels_rejection_percentage;
	float num_sigmas_reject_badlabels;
	float ema_alpha;
	int *cur_iteration;
	float loss_scale;
	int *t;
	float epoch;
	int subdivisions;
	Darknet::Layer *layers;
	float *output;
	learning_rate_policy policy;
	int benchmark_layers;
	int *total_bbox;
	int *rewritten_bbox;

	float learning_rate;
	float learning_rate_min;
	float learning_rate_max;
	int batches_per_cycle;
	int batches_cycle_mult;
	float momentum;
	float decay;
	float gamma;
	float scale;
	float power;
	int time_steps;
	int step;
	int max_batches;
	int num_boxes;
	int train_images_num;
	float *seq_scales;
	float *scales;
	int   *steps;
	int num_steps;
	int burn_in;
	int cudnn_half;

	int adam;
	float B1;
	float B2;
	float eps;

	int inputs;
	int outputs;
	int truths;
	int notruth;
	/// The height of the network.  Must be divisible by @p 32.  E.g, @p 480.
	int h;
	/// The width of the network.  Must be divisible by @p 32.  E.g., @p 640.
	int w;
	/// The number of channels for the network.  Typically @p 3 when working with RGB images.
	int c;
	int max_crop;
	int min_crop;
	float max_ratio;
	float min_ratio;
	int center;
	int flip; ///< horizontal flip 50% probability augmentaiont for classifier training (default = 1)
	int gaussian_noise;
	int blur;
	int mixup;
	float label_smooth_eps;
	int resize_step;
	int attention;
	int adversarial;
	float adversarial_lr;
	float max_chart_loss;
	int letter_box;
	int mosaic_bound;
	int contrastive;
	int contrastive_jit_flip;
	int contrastive_color;
	int unsupervised;
	float angle;
	float aspect;
	float exposure;
	float saturation;
	float hue;
	int random;
	int track;
	int augment_speed;
	int sequential_subdivisions;
	int init_sequential_subdivisions;
	int current_subdivision;
	int try_fix_nan;

	int gpu_index;
	tree *hierarchy;

	float *input;
	float *truth;
	float *delta;
	float *workspace;
	int train;
	int index;
	float *cost;
	float clip;

//#ifdef GPU
	//float *input_gpu;
	//float *truth_gpu;
	float *delta_gpu;
	float *output_gpu;

	float *input_state_gpu;
	float *input_pinned_cpu;	///< memory allocated using @p cudaHostAlloc() which is used to transfer between the GPU and CPU
	int input_pinned_cpu_flag;

	float **input_gpu;
	float **truth_gpu;
	float **input16_gpu;
	float **output16_gpu;
	size_t *max_input16_size;
	size_t *max_output16_size;
	int wait_stream;

	void *cuda_graph;
	void *cuda_graph_exec;
	int use_cuda_graph;
	int *cuda_graph_ready;

	float *global_delta_gpu;
	float *state_delta_gpu;
	size_t max_delta_gpu_size;
//#endif  // GPU
	int optimized_memory;
	int dynamic_minibatch;
	size_t workspace_size_limit;
} network;

// image.h
typedef struct image {
	int w;
	int h;
	int c;
	float *data;
} image;

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
	box bbox;
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
* @see @ref load_thread()
* @see @ref load_args.type
*/
typedef enum
{
	DETECTION_DATA,
	IMAGE_DATA, ///< causes @ref load_image() and @ref resize_image() to be called
	LETTERBOX_DATA,
} data_type;


/** Used when a secondary thread is created to load things, such as images.
* @see @ref load_image()
* @see @ref data_type
*/
typedef struct load_args {
	int threads;
	char **paths;
	char *path;
	int n; ///< number of images, or batch size?
	int m; ///< maximum number of images?
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
	data *d;
	image *im;
	image *resized;
	data_type type;
	tree *hierarchy;
} load_args;


// data.h
typedef struct box_label {
	int id;
	int track_id;
	float x, y, w, h;
	float left, right, top, bottom;
} box_label;

// -----------------------------------------------------


// parser.c
network *load_network(const char * cfg, const char * weights, int clear);
network *load_network_custom(const char * cfg, const char * weights, int clear, int batch);
void free_network(network net);
void free_network_ptr(network* net);


// box.h
void do_nms_sort(detection *dets, int total, int classes, float thresh);
void do_nms_obj(detection *dets, int total, int classes, float thresh);
void diounms_sort(detection *dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1);

// network.h
float *network_predict(network net, float *input);
float *network_predict_ptr(network *net, float *input);
detection *get_network_boxes(network *net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
det_num_pair* network_predict_batch(network *net, image im, int batch_size, int w, int h, float thresh, float hier, int *map, int relative, int letter);
void free_detections(detection *dets, int n);
void free_batch_detections(det_num_pair *det_num_pairs, int n);
void fuse_conv_batchnorm(network net);
void calculate_binary_weights(network net);
char *detection_to_json(detection *dets, int nboxes, int classes, char **names, long long int frame_id, char *filename);

Darknet::Layer * get_network_layer(network* net, int i);
detection *make_network_boxes(network *net, float thresh, int *num);
void reset_rnn(network *net);
float *network_predict_image(network *net, image im);
float *network_predict_image_letterbox(network *net, image im);
float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, network *existing_net);
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, float thresh, float iou_thresh, int mjpeg_port, int show_imgs, int benchmark_layers, char* chart_path);
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);
int network_width(network *net);
int network_height(network *net);
void optimize_picture(network *net, image orig, int max_layer, float scale, float rate, float thresh, int norm);

// image.h
void make_image_red(image im);
image make_attention_image(int img_size, float *original_delta_cpu, float *original_input_cpu, int w, int h, int c, float alpha);
image resize_image(image im, int w, int h);
void quantize_image(image im);
void copy_image_from_bytes(image im, char *pdata);
image letterbox_image(image im, int w, int h);
void rgbgr_image(image im);
image make_image(int w, int h, int c);
image load_image(char *filename, int w, int h, int c);
void free_image(image m);
image crop_image(image im, int dx, int dy, int w, int h);
image resize_min(image im, int min);

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

// http_stream.h
void delete_json_sender();
void send_json_custom(char const* send_buf, int port, int timeout);
double get_time_point();
void start_timer();
void stop_timer();
double get_time();
void stop_timer_and_show();
void stop_timer_and_show_name(char *name);
void show_total_time();

void set_track_id(detection *new_dets, int new_dets_num, float thresh, float sim_thresh, float track_ciou_norm, int deque_size, int dets_for_track, int dets_for_show);
int fill_remaining_id(detection *new_dets, int new_dets_num, int new_track_id, float thresh);


// gemm.h
void init_cpu();

#ifdef __cplusplus
}
#endif  // __cplusplus








// The "new" (V3, July 2024) Darknet/YOLO API starts here.

#ifdef __cplusplus
extern "C" {
#endif

	/// This is the @p C equivalent to @ref Darknet::set_verbose().
	void darknet_set_verbose(const bool flag);

	/// This is the @p C equivalent to @ref Darknet::set_trace().
	void darknet_set_trace(const bool flag);

#ifdef __cplusplus
}
#endif  // __cplusplus
