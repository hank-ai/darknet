/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

/** @file
 * Defines the layer structure and includes all of the different layer include files.
 */

#include "darknet_internal.hpp"


namespace Darknet
{
	struct Layer final
	{
		Darknet::ELayerType type; ///< @see @ref Darknet::to_string()
		ACTIVATION activation;
		ACTIVATION lstm_activation;
		COST_TYPE cost_type;

		void(*forward)		(Layer & l, Darknet::NetworkState network_state);
		void(*backward)		(Layer & l, Darknet::NetworkState network_state);
		void(*update)		(Layer & l, int, float, float, float);
		void(*forward_gpu)	(Layer & l, Darknet::NetworkState network_state);
		void(*backward_gpu)	(Layer & l, Darknet::NetworkState network_state);
		void(*update_gpu)	(Layer & l, int, float, float, float, float);

		Layer *share_layer;
		int train;
		int avgpool;
		int batch_normalize;
		int shortcut;
		int batch;
		int dynamic_minibatch;
		int forced;
		int flipped;
		int inputs;
		int outputs;
		float mean_alpha;
		int nweights; ///< number of floats stored in @ref weights
		int nbiases; ///< unused?  Seems to be no references to this in the codebase.
		int extra;
		int truths;
		int h; ///< height
		int w; ///< width
		int c; ///< channels
		int out_h;
		int out_w;
		int out_c;
		int n; ///< number of anchors, masks (?), weights (?); for example, with YOLOv4-tiny this is set to @p 3
		int max_boxes;
		int truth_size;
		int groups;
		int group_id;
		int size;
		int side;
		int stride;
		int stride_x;
		int stride_y;
		int dilation;
		int antialiasing;
		int maxpool_depth;
		int maxpool_zero_nonmax;
		int out_channels;
		float reverse;
		int coordconv;
		int flatten;
		int spatial;
		int pad;
		int sqrt;
		int flip;
		int index; ///< layer number starting at zero ([net] does not count)
		int scale_wh;
		int binary;
		int xnor;
		int peephole;
		int use_bin_output;
		int keep_delta_gpu;
		int optimized_memory;
		int steps;
		int history_size;
		int bottleneck;
		float time_normalizer;
		int state_constrain;
		int hidden;
		int truth;
		float smooth;
		float dot;
		int deform;
		int grad_centr;
		int sway;
		int rotate;
		int stretch;
		int stretch_sway;
		float angle;
		float jitter;
		float resize;
		float saturation;
		float exposure;
		float shift;
		float ratio;
		float learning_rate_scale;
		float clip;
		int focal_loss;
		float *classes_multipliers;
		float label_smooth_eps;
		int noloss;
		int softmax;
		int classes;
		int detection;
		int embedding_layer_id;
		float *embedding_output;
		int embedding_size;
		float sim_thresh;
		int track_history_size;
		int dets_for_track;
		int dets_for_show;
		float track_ciou_norm;
		int coords;
		int background;
		int rescore;
		int objectness;
		int does_cost;
		int joint;
		int noadjust;
		int reorg;
		int log;
		int tanh;
		int *mask;
		int total;
		float bflops;

		int adam;
		float B1;
		float B2;
		float eps;

		int t;

		float alpha;
		float beta;
		float kappa;

		float coord_scale;
		float object_scale;
		float noobject_scale;
		float mask_scale;
		float class_scale;
		int bias_match;
		float random;
		float ignore_thresh;
		float truth_thresh;
		float iou_thresh;
		float thresh;
		float focus;
		int classfix;
		int absolute;
		int assisted_excitation;

		int onlyforward;
		int stopbackward;
		int train_only_bn;
		int dont_update;
		int burnin_update;
		int dontload;
		int dontsave;
		int dontloadscales;
		int numload;

		float temperature;
		float probability;
		float dropblock_size_rel;
		int dropblock_size_abs;
		int dropblock;
		float scale;

		int receptive_w;
		int receptive_h;
		int receptive_w_scale;
		int receptive_h_scale;

		char  * cweights; ///< @todo V5: possibly unused?
		int   * indexes;
		int   * input_layers;
		int   * input_sizes;
		float **layers_output;
		float **layers_delta;
		WEIGHTS_TYPE_T weights_type;
		WEIGHTS_NORMALIZATION_T weights_normalization;
		int   * map;
		int   * counts;
		float ** sums;
		float * rand;
		float * cost;
		int *labels;
		int *class_ids;
		int contrastive_neg_max; ///< @todo V5: possibly unused?
		float *cos_sim;
		float *exp_cos_sim;
		float *p_constrastive;
		contrastive_params *contrast_p_gpu;
		float * state;
		float * prev_state;
		float * forgot_state; ///< @todo V5: unused?
		float * forgot_delta; ///< @todo V5: unused?
		float * state_delta; ///< @todo V5: possibly unused?
		float * combine_cpu_unused; ///< @todo V5: unused?
		float * combine_delta_cpu_unused; ///< @todo V5: unused?

		float *concat;
		float *concat_delta;

		float *binary_weights;

		float *biases;			/// biases loaded here by @ref load_convolutional_weights() and @ref load_connected_weights(), see @ref n
		float *bias_updates;

		float *scales;			/// scales loaded here by @ref load_convolutional_weights() when @ref batch_normalize is set
		float *scale_updates;

		float *weights_ema;
		float *biases_ema;
		float *scales_ema;

		float *weights;			/// weights loaded here by @ref load_convolutional_weights(), @ref load_connected_weights(), and @ref load_shortcut_weights()
		float *weight_updates;

		float scale_x_y;
		int objectness_smooth;
		int new_coords;
		int show_details;
		float max_delta;
		float uc_normalizer;
		float iou_normalizer;
		float obj_normalizer;
		float cls_normalizer;
		float delta_normalizer;
		IOU_LOSS iou_loss;
		IOU_LOSS iou_thresh_kind;
		NMS_KIND nms_kind;
		float beta_nms;
		YOLO_POINT yolo_point;

		char *align_bit_weights_gpu;
		float *mean_arr_gpu;
		float *align_workspace_gpu;
		float *transposed_align_workspace_gpu;
		int align_workspace_size;

		char *align_bit_weights;
		float *mean_arr;
		int align_bit_weights_size;
		int lda_align;
		int new_lda;
		int bit_align;

		float *col_image;
		float * delta;
		float * output;
		float * activation_input;
		int delta_pinned;
		int output_pinned;
		float * loss;
		float * squared;
		float * norms;

		float * spatial_mean;
		float * mean;
		float * variance;

		float * mean_delta;
		float * variance_delta;

		float * rolling_mean;		/// rolling means loaded here by @ref load_convolutional_weights() and @ref load_connected_weights() when @ref batch_normalize is set
		float * rolling_variance;	/// rolling variance loaded here by @ref load_convolutional_weights() and @ref load_connected_weights() when @ref batch_normalize is set

		float * x;
		float * x_norm;

		float * m;
		float * v;

		float * bias_m;
		float * bias_v;
		float * scale_m;
		float * scale_v;

		float *z_cpu;
		float *r_cpu;
		float *h_cpu;
		float *stored_h_cpu;
		float * prev_state_cpu;

		float *temp_cpu;
		float *temp2_cpu;
		float *temp3_cpu;

		float *dh_cpu;
		float *hh_cpu;
		float *prev_cell_cpu;
		float *cell_cpu;
		float *f_cpu;
		float *i_cpu;
		float *g_cpu;
		float *o_cpu;
		float *c_cpu;
		float *stored_c_cpu; ///< @todo V5: unused?
		float *dc_cpu;

		float *binary_input;
		uint32_t *bin_re_packed_input;
		char *t_bit_input;

		Layer *input_layer;
		Layer *self_layer;
		Layer *output_layer;

		Layer *reset_layer_unused; ///< @todo V5: unused?
		Layer *update_layer_unused; ///< @todo V5: unused?
		Layer *state_layer_unused; ///< @todo V5: unused?

		Layer *input_gate_layer_unused; ///< @todo V5: unused?
		Layer *state_gate_layer_unused; ///< @todo V5: unused?
		Layer *input_save_layer_unused; ///< @todo V5: unused?
		Layer *state_save_layer_unused; ///< @todo V5: unused?
		Layer *input_state_layer_unused; ///< @todo V5: unused?
		Layer *state_state_layer_unused; ///< @todo V5: unused?

		Layer *input_z_layer_unused; ///< @todo V5: unused?
		Layer *state_z_layer_unused; ///< @todo V5: unused?

		Layer *input_r_layer_unused; ///< @todo V5: unused?
		Layer *state_r_layer_unused; ///< @todo V5: unused?

		Layer *input_h_layer_unused; ///< @todo V5: unused?
		Layer *state_h_layer_unused; ///< @todo V5: unused?

		// "They are mostly relevant during training and not in the prediction/output phase."

		Layer *wz_unused; ///< @todo V5: unused?
		Layer *uz_unused; ///< @todo V5: unused?
		Layer *wr_unused; ///< @todo V5: unused?
		Layer *ur_unused; ///< @todo V5: unused?
		Layer *wh_unused; ///< @todo V5: unused?
		Layer *uh_unused; ///< @todo V5: unused?
		Layer *uo; ///< used in lstm (update for output gate?)
		Layer *wo; ///< used in lstm (weights for output forget gate?)
		Layer *vo_unused; ///< @todo V5: unused?
		Layer *uf; ///< used in lstm (update for forget gate?)
		Layer *wf; ///< used in lstm (weights for forget gate?)
		Layer *vf_unused; ///< @todo V5: unused?
		Layer *ui; ///< used in lstm (update input gate?)
		Layer *wi; ///< used in lstm (weight for input connections?)
		Layer *vi_unused; ///< @todo V5: unused?
		Layer *ug; ///< used in lstm (update gradient?)
		Layer *wg; ///< used in lstm (weight gradient?)

		Darknet::Tree *softmax_tree;

		size_t workspace_size;

		int *indexes_gpu;

		int stream;
		int wait_stream_id;

		float *z_gpu;
		float *r_gpu;
		float *h_gpu;
		float *stored_h_gpu;
		float *bottelneck_hi_gpu;
		float *bottelneck_delta_gpu;

		float *temp_gpu;
		float *temp2_gpu;
		float *temp3_gpu;

		float *dh_gpu;
		float *hh_gpu; ///< @todo V5: possibly unused?
		float *prev_cell_gpu;
		float *prev_state_gpu;
		float *last_prev_state_gpu; ///< @todo V5: unused?
		float *last_prev_cell_gpu; ///< @todo V5: unused?
		float *cell_gpu;
		float *f_gpu;
		float *i_gpu;
		float *g_gpu;
		float *o_gpu;
		float *c_gpu;
		float *stored_c_gpu; ///< @todo V5: possibly unused?
		float *dc_gpu;

		// adam
		float *m_gpu;
		float *v_gpu;
		float *bias_m_gpu; ///< @todo V5: possibly unused?
		float *scale_m_gpu; ///< @todo V5: possibly unused?
		float *bias_v_gpu; ///< @todo V5: possibly unused?
		float *scale_v_gpu; ///< @todo V5: possibly unused?

		float * combine_gpu; ///< @todo V5: unused?
		float * combine_delta_gpu; ///< @todo V5: unused?

		float * forgot_state_gpu; ///< @todo V5: possibly unused?
		float * forgot_delta_gpu; ///< @todo V5: possibly unused?
		float * state_gpu;
		float * state_delta_gpu; ///< @todo V5: possibly unused?
		float * gate_gpu; ///< @todo V5: possibly unused?
		float * gate_delta_gpu; ///< @todo V5: possibly unused?
		float * save_gpu; ///< @todo V5: possibly unused?
		float * save_delta_gpu; ///< @todo V5: unused?
		float * concat_gpu; ///< @todo V5: unused?
		float * concat_delta_gpu;  ///< @todo V5: unused?

		float *binary_input_gpu;
		float *binary_weights_gpu;
		float *bin_conv_shortcut_in_gpu; ///< @todo V5: possibly unused?
		float *bin_conv_shortcut_out_gpu; ///< @todo V5: possibly unused?

		float * mean_gpu;
		float * variance_gpu;
		float * m_cbn_avg_gpu; ///< @todo V5: possibly unused?
		float * v_cbn_avg_gpu; ///< @todo V5: possibly unused?

		float * rolling_mean_gpu;
		float * rolling_variance_gpu;

		float * variance_delta_gpu;
		float * mean_delta_gpu;

		float * col_image_gpu;

		float * x_gpu;
		float * x_norm_gpu;
		float * weights_gpu;
		float * weight_updates_gpu;
		float * weight_deform_gpu;
		float * weight_change_gpu;

		float * weights_gpu16;
		float * weight_updates_gpu16;

		float * biases_gpu;
		float * bias_updates_gpu;
		float * bias_change_gpu;

		float * scales_gpu;
		float * scale_updates_gpu;
		float * scale_change_gpu;

		float * input_antialiasing_gpu;
		float * output_gpu;
		float * output_avg_gpu;
		float * activation_input_gpu;
		float * loss_gpu;
		float * delta_gpu;
		float * cos_sim_gpu;
		float * rand_gpu;
		float * drop_blocks_scale;
		float * drop_blocks_scale_gpu;
		float * squared_gpu;
		float * norms_gpu;

		float *gt_gpu;
		float *a_avg_gpu;

		int *input_sizes_gpu;
		float **layers_output_gpu;
		float **layers_delta_gpu;
#ifdef CUDNN
		cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
		cudnnTensorDescriptor_t srcTensorDesc16, dstTensorDesc16;
		cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
		cudnnTensorDescriptor_t dsrcTensorDesc16, ddstTensorDesc16;
		cudnnTensorDescriptor_t normTensorDesc, normDstTensorDesc, normDstTensorDescF16;
		cudnnFilterDescriptor_t weightDesc, weightDesc16;
		cudnnFilterDescriptor_t dweightDesc, dweightDesc16;
		cudnnConvolutionDescriptor_t convDesc;
		cudnnConvolutionFwdAlgo_t fw_algo, fw_algo16;
		cudnnConvolutionBwdDataAlgo_t bd_algo, bd_algo16;
		cudnnConvolutionBwdFilterAlgo_t bf_algo, bf_algo16;
		cudnnPoolingDescriptor_t poolingDesc;
#else
		// pad the structure so it has the same size regardless of CPU, CUDA, CUDNN, etc.
		void* srcTensorDesc, *dstTensorDesc;
		void* srcTensorDesc16, *dstTensorDesc16;
		void* dsrcTensorDesc, *ddstTensorDesc;
		void* dsrcTensorDesc16, *ddstTensorDesc16;
		void* normTensorDesc, *normDstTensorDesc, *normDstTensorDescF16;
		void* weightDesc, *weightDesc16;
		void* dweightDesc, *dweightDesc16;
		void* convDesc;
		UNUSED_ENUM_TYPE fw_algo, fw_algo16;
		UNUSED_ENUM_TYPE bd_algo, bd_algo16;
		UNUSED_ENUM_TYPE bf_algo, bf_algo16;
		void* poolingDesc;
#endif  // CUDNN
	};
}


#include "avgpool_layer.hpp"
#include "batchnorm_layer.hpp"
#include "connected_layer.hpp"
#include "convolutional_layer.hpp"
#include "cost_layer.hpp"
#include "crnn_layer.hpp"
#include "dropout_layer.hpp"
#include "gaussian_yolo_layer.hpp"
#include "lstm_layer.hpp"
#include "maxpool_layer.hpp"
#include "region_layer.hpp"
#include "reorg_layer.hpp"
#include "rnn_layer.hpp"
#include "route_layer.hpp"
#include "sam_layer.hpp"
#include "scale_channels_layer.hpp"
#include "shortcut_layer.hpp"
#include "softmax_layer.hpp"
#include "upsample_layer.hpp"
#include "yolo_layer.hpp"
