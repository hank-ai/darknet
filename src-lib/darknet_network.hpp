// Oh boy, why am I about to do this....

// ^ That comment was part of the very first commit to the original Darknet repo by Joseph Redmon on Nov 4, 2013.

#pragma once

#include "darknet.h"


namespace Darknet
{
	/** A place to store other details related to the neural network which we cannot easily add to the usual
	 * @ref Darknet::Network structure.  These are typically C++ objects, or things added post %Darknet V3 (2024-08).
	 *
	 * @see @ref Darknet::Network::details
	 *
	 * @since 2024-08-06
	 */
	struct NetworkDetails
	{
		public:

			/// Default constructor.
			NetworkDetails();

			/// @{ Filename used to load the neural nework.  @since 2024-08-06
			std::filesystem::path cfg_path;
			std::filesystem::path names_path;
			std::filesystem::path weights_path;
			/// @}

			/** The name to use for every object class.  Will @em always match the number of classes in the neural network.
			* @see @ref Darknet::assign_default_class_colours()
			* @see @ref Darknet::load_names()
			* @since 2024-08-07
			*/
			VStr class_names;

			/** BGR colours to use for each class.
			* @see @ref Darknet::NetworkDetails::class_names
			* @see @ref Darknet::assign_default_class_colours()
			* @since 2024-08-07
			*/
			std::vector<cv::Scalar> class_colours;

			/** BGR colours to use for the label text.
			* @see @ref Darknet::NetworkDetails::class_names
			* @see @ref Darknet::assign_default_class_colours()
			* @since 2024-08-07
			*/
			std::vector<cv::Scalar> text_colours;

			/** Object detection threshold to apply.
			 * Default is @p 0.25.
			 * @see @ref Darknet::set_detection_threshold()
			 * @since 2024-07-24
			 */
			float detection_threshold;

			/** Non maximal suppression threshold to apply.
			 * Default is @p 0.45.
			 * @see @ref Darknet::set_non_maximal_suppression_threshold()
			 * @since 2024-07-24
			 */
			float non_maximal_suppression_threshold;

			/** Fix out-of-bound values for objects near the edges of images.
			 * Default is @p true.
			 * @see @ref Darknet::fix_out_of_bound_values()
			 * @since 2024-07-25
			 */
			bool fix_out_of_bound_normalized_coordinates;

			/** The OpenCV line type to use when drawing lines such as bounding boxes.  Possible values include
			 * @p cv::LineTypes::LINE_4, @p cv::LineTypes::LINE_8, and @p cv::LineTypes::CV_LINE_AA.  @p LINE_4 is the fastest
			 * but lowest quality, while @p LINE_AA (anti-alias) is the slowest with highest quality.
			* Default is @p cv::LineTypes::LINE_4.
			* @see @ref Darknet::set_annotation_font()
			* @since 2024-07-30
			*/
			cv::LineTypes cv_font_line_type;

			/** The OpenCV built-in font to use when generating text, such as the labels above bounding boxes.
			 * Default is @p cv::HersheyFonts::FONT_HERSHEY_PLAIN.
			 * @see @ref Darknet::set_annotation_font()
			 * @since 2024-07-30
			 */
			cv::HersheyFonts cv_font_face;

			/** The OpenCV font thickness to use when generating text, such as the labels above bounding boxes.
			 * Default is @p 1.
			 * @see @ref Darknet::set_annotation_font()
			 * @since 2024-07-30
			 */
			int cv_font_thickness;

			/** The OpenCV font scale to use when generating text, such as the labels above bounding boxes.
			 * Default is @p 1.0.
			 * @see @ref Darknet::set_annotation_font()
			 * @since 2024-07-30
			 */
			double cv_font_scale;

			/** Whether bounding boxes should use rounded corners.
			 * Default is @p false (meaning square bounding boxes).
			 * @see @ref Darknet::set_rounded_corner_bounding_boxes()
			 * @since 2024-07-30
			 */
			bool bounding_boxes_with_rounded_corners;

			/** The "roundness" of the corners when @ref bounding_boxes_with_rounded_corners is set to @p true.
			 * Default is @p 0.5.
			 * @see @ref Darknet::set_rounded_corner_bounding_boxes()
			 * @since 2024-07-30
			 */
			float bounding_boxes_corner_roundness;

			/** Whether bounding boxes are drawn when annotating images.
			 * Default is @p true.
			 * @see @ref Darknet::set_annotation_draw_bb()
			 * @see @ref Darknet::set_annotation_draw_label()
			 * @since 2024-07-30
			 */
			bool annotate_draw_bb;

			/** Whether text labels are drawn above bounding boxes when annotating images.
			 * Default is @p true.
			 * @see @ref Darknet::set_annotation_draw_bb()
			 * @see @ref Darknet::set_annotation_draw_label()
			 * @since 2024-07-30
			 */
			bool annotate_draw_label;
	};


	/// Neural network structure.  Contains all of the layers.  Created by @ref Darknet::CfgFile::create_network().
	struct Network
	{
		public:
			int n;	///< The number of layers in the network.  @see @ref layers
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
			Darknet::Layer *layers; ///< Each section in the @p .cfg file is converted into a layer.  @see @ref n
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
			Darknet::NetworkDetails * details;
	};

	struct NetworkState
	{
		float *truth;
		float *input;
		float *delta;
		float *workspace;
		int train;
		int index;
		Darknet::Network net;
	};

	char *detection_to_json(detection *dets, int nboxes, int classes, const Darknet::VStr & names, long long int frame_id, char *filename);
}


//extern "C"
//{







#ifdef GPU
float train_networks(Darknet::Network *nets, int n, data d, int interval);
void sync_nets(Darknet::Network *nets, int n, int interval);
float train_network_datum_gpu(Darknet::Network net, float *x, float *y); // xxx2
float * network_predict_gpu(Darknet::Network net, float *input);
float * get_network_output_gpu_layer(Darknet::Network net, int i);
float * get_network_output_gpu(const Darknet::Network net);
float * get_network_delta_gpu_layer(Darknet::Network net, int i);
void forward_network_gpu(Darknet::Network net, Darknet::NetworkState state);
void backward_network_gpu(Darknet::Network net, Darknet::NetworkState state);
void update_network_gpu(Darknet::Network net);
void forward_backward_network_gpu(Darknet::Network net, float *x, float *y); // xxx1
#endif

/** Think of this as the constructor for the @ref Darknet::Network object.
 * @param [in] n The number of network layers to initialize.
 */
Darknet::Network make_network(int n);

void free_network(Darknet::Network net);
void free_network_ptr(Darknet::Network * net);

Darknet::Network *load_network(const char * cfg, const char * weights, int clear);
Darknet::Network *load_network_custom(const char * cfg, const char * weights, int clear, int batch);

float get_current_seq_subdivisions(Darknet::Network net);
int get_sequence_value(Darknet::Network net);

float get_current_rate(Darknet::Network net);

int get_current_batch(Darknet::Network net);

int64_t get_current_iteration(Darknet::Network net);

void compare_networks(Darknet::Network n1, Darknet::Network n2, data d);

void forward_network(Darknet::Network net, Darknet::NetworkState state);
void backward_network(Darknet::Network net, Darknet::NetworkState state);
void update_network(Darknet::Network net);

float train_network(Darknet::Network net, data d); // xxx4
float train_network_waitkey(Darknet::Network net, data d, int wait_key); // xxx5
float train_network_batch(Darknet::Network net, data d, int n);
float train_network_datum(Darknet::Network net, float *x, float *y); // xxx3

matrix network_predict_data(Darknet::Network net, data test);
float network_accuracy(Darknet::Network net, data d);
float *network_accuracies(Darknet::Network net, data d, int n);
float network_accuracy_multi(Darknet::Network net, data d, int n);
void top_predictions(Darknet::Network net, int n, int *index);
float *get_network_output(Darknet::Network net);
float *get_network_output_layer(Darknet::Network net, int i);
float *get_network_delta_layer(Darknet::Network net, int i);
float *get_network_delta(Darknet::Network net);
int get_network_output_size_layer(Darknet::Network net, int i);

int get_network_output_size(Darknet::Network net);

Darknet::Image get_network_image(Darknet::Network net);
Darknet::Image get_network_image_layer(Darknet::Network net, int i);
int get_predicted_class_network(Darknet::Network net);
void print_network(Darknet::Network net);
void visualize_network(Darknet::Network net);
int resize_network(Darknet::Network * net, int w, int h);
void set_batch_network(Darknet::Network * net, int b);
int get_network_input_size(Darknet::Network net);

float get_network_cost(Darknet::Network net);

int get_network_nuisance(Darknet::Network net);
int get_network_background(Darknet::Network net);
Darknet::Network combine_train_valid_networks(Darknet::Network net_train, Darknet::Network net_map);
void copy_weights_net(Darknet::Network net_train, Darknet::Network *net_map);
void free_network_recurrent_state(Darknet::Network net);
void randomize_network_recurrent_state(Darknet::Network net);
void remember_network_recurrent_state(Darknet::Network net);
void restore_network_recurrent_state(Darknet::Network net);
int is_ema_initialized(Darknet::Network net);
void ema_update(Darknet::Network net, float ema_alpha);
void ema_apply(Darknet::Network net);
void reject_similar_weights(Darknet::Network net, float sim_threshold);
//}

float *network_predict(Darknet::Network net, float *input);
float *network_predict_ptr(Darknet::Network *net, float *input);
detection *get_network_boxes(Darknet::Network * net, int w, int h, float thresh, float hier, int *map, int relative, int *num, int letter);
det_num_pair* network_predict_batch(Darknet::Network *net, Darknet::Image im, int batch_size, int w, int h, float thresh, float hier, int *map, int relative, int letter);
void free_detections(detection *dets, int n);
void free_batch_detections(det_num_pair *det_num_pairs, int n);
void fuse_conv_batchnorm(Darknet::Network net);
void calculate_binary_weights(Darknet::Network net);

Darknet::Layer * get_network_layer(Darknet::Network* net, int i);
detection *make_network_boxes(Darknet::Network *net, float thresh, int *num);
void reset_rnn(Darknet::Network *net);
float *network_predict_image(Darknet::Network *net, Darknet::Image im);
float *network_predict_image_letterbox(Darknet::Network *net, Darknet::Image im);
float validate_detector_map(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, Darknet::Network *existing_net);
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, float thresh, float iou_thresh, int mjpeg_port, int show_imgs, int benchmark_layers, char* chart_path);
void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers);
int network_width(Darknet::Network *net);
int network_height(Darknet::Network *net);
void optimize_picture(Darknet::Network *net, Darknet::Image orig, int max_layer, float scale, float rate, float thresh, int norm);
