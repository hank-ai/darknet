// Oh boy, why am I about to do this....

// ^ That comment was part of the very first commit to the original Darknet repo by Joseph Redmon on Nov 4, 2013.

#pragma once

#include "darknet.h"


namespace Darknet
{
	struct NetworkState
	{
		float *truth;
		float *input;
		float *delta;
		float *workspace;
		int train;
		int index;
		network net;
	};

	/** Store other details related to the neural network which we cannot easily add to the usual @ref Darknet::Network
	 * structure.  These are typically C++ objects, or things added post %Darknet V3 (2024-08).
	 *
	 * @see @ref Darknet::Network::details
	 *
	 * @since 2024-08-06
	 */
	struct NetworkDetails
	{
		public:

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
			 * @see @ref Darknet::set_annotation_draw_rounded_bb()
			 * @since 2024-07-30
			 */
			bool bounding_boxes_with_rounded_corners;

			/** The "roundness" of the corners when @ref bounding_boxes_with_rounded_corners is set to @p true.
			 * Default is @p 0.5.
			 * @see @ref Darknet::set_annotation_draw_rounded_bb()
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


	char *detection_to_json(detection *dets, int nboxes, int classes, const Darknet::VStr & names, long long int frame_id, char *filename);
}


extern "C"
{
#ifdef GPU
float train_networks(network *nets, int n, data d, int interval);
void sync_nets(network *nets, int n, int interval);
float train_network_datum_gpu(network net, float *x, float *y);
float *network_predict_gpu(network net, float *input);
float * get_network_output_gpu_layer(network net, int i);
float * get_network_delta_gpu_layer(network net, int i);
float *get_network_output_gpu(network net);
void forward_network_gpu(network net, Darknet::NetworkState state);
void backward_network_gpu(network net, Darknet::NetworkState state);
void update_network_gpu(network net);
void forward_backward_network_gpu(network net, float *x, float *y);
#endif

/** Think of this as the constructor for the @ref Darknet::Network object.
 * @param [in] n The number of network layers to initialize.
 */
network make_network(int n);

void free_network(network net);
void free_network_ptr(network* net);

network *load_network(const char * cfg, const char * weights, int clear);
network *load_network_custom(const char * cfg, const char * weights, int clear, int batch);

float get_current_seq_subdivisions(network net);
int get_sequence_value(network net);
float get_current_rate(network net);
int get_current_batch(network net);
int64_t get_current_iteration(network net);
//void free_network(network net); // darknet.h
void compare_networks(network n1, network n2, data d);

void forward_network(network net, Darknet::NetworkState state);
void backward_network(network net, Darknet::NetworkState state);
void update_network(network net);

float train_network(network net, data d);
float train_network_waitkey(network net, data d, int wait_key);
float train_network_batch(network net, data d, int n);
float train_network_datum(network net, float *x, float *y);

matrix network_predict_data(network net, data test);
float network_accuracy(network net, data d);
float *network_accuracies(network net, data d, int n);
float network_accuracy_multi(network net, data d, int n);
void top_predictions(network net, int n, int *index);
float *get_network_output(network net);
float *get_network_output_layer(network net, int i);
float *get_network_delta_layer(network net, int i);
float *get_network_delta(network net);
int get_network_output_size_layer(network net, int i);
int get_network_output_size(network net);
Darknet::Image get_network_image(network net);
Darknet::Image get_network_image_layer(network net, int i);
int get_predicted_class_network(network net);
void print_network(network net);
void visualize_network(network net);
int resize_network(network *net, int w, int h);
void set_batch_network(network *net, int b);
int get_network_input_size(network net);
float get_network_cost(network net);

int get_network_nuisance(network net);
int get_network_background(network net);
network combine_train_valid_networks(network net_train, network net_map);
void copy_weights_net(network net_train, network *net_map);
void free_network_recurrent_state(network net);
void randomize_network_recurrent_state(network net);
void remember_network_recurrent_state(network net);
void restore_network_recurrent_state(network net);
int is_ema_initialized(network net);
void ema_update(network net, float ema_alpha);
void ema_apply(network net);
void reject_similar_weights(network net, float sim_threshold);
}
