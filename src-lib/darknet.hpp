/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

/** @file
 * Include this file to get access to the new Darknet V3 C++ API.
 * https://darknetcv.ai/api/api.html
 */

#include <filesystem>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <vector>
#include <ciso646>

#include <opencv2/opencv.hpp>

#include "darknet.h"


/** The namespace for the C++ %Darknet API.  Note this namespace contains both public and private API calls.
 * The structures, enums, classes and functions declared in darknet.hpp are part of the public API.
 */
namespace Darknet
{
	/// @{ Convenient simple types used in the Darknet/YOLO C++ API.  @since 2024-07-14
	using MStr			= std::map<std::string, std::string>;
	using MStrInt		= std::map<std::string, int>;
	using SStr			= std::set<std::string>;
	using SInt			= std::set<int>;
	using VFloat		= std::vector<float>;
	using VInt			= std::vector<int>;
	using VStr			= std::vector<std::string>;
	using VScalars		= std::vector<cv::Scalar>;
	using MMats			= std::map<int, cv::Mat>;
	using NetworkPtr	= DarknetNetworkPtr;
	using Box			= DarknetBox;
	using Detection		= DarknetDetection;
	using Image			= DarknetImage;
	/// @}

	/// The @p layer structure has been renamed and moved to darknet_layer.hpp.
	struct Layer;

	/// The @p network_state structure has been renamed and moved to network.hpp.
	struct NetworkState;

	/** Some C++ structures we'd like to insert into the C "Network".  Needs to be a @p void* pointer so older C code can
	 * continue using the Network without causing any problems.
	 */
	struct NetworkDetails;

	/// The @p network structure has been renamed and moved to darknet_network.hpp.
	struct Network;


	/** The likelyhood of a specific object class having been predicted.  This map contains all of the non-zero values.
	 * The key is the zero-based class indexes, and the values are the confidences for the classes, between @p 0.0f and
	 * @p 1.0f.
	 *
	 * For example, if "car" is class #2 and "truck" is class #3, and %Darknet is 95% certain it found a car and 82% certain
	 * it found a truck, the map would then contain
	 * @p "{ {2, 0.95}, {3, 0.82} }".
	 *
	 * @see @ref Darknet::Prediction
	 *
	 * @since 2024-07-24
	 */
	using Probabilities = std::map<int, float>;

	/** When parsing command-line parameters, each parameter is assigned a "type".
	 *
	 * @see @ref Darknet::Parm
	 * @see @ref Darknet::parse_arguments()
	 *
	 * @since 2024-07-29
	 */
	enum class EParmType
	{
		kUnknown		, ///< Should be unused.  See "kOther" instead for all other parameter types.
		kCfgFilename	, ///< The configuration file to load.  There should only be 1 parameter of this type.
		kNamesFilename	, ///< The names file to load.  There should only be 1 parameter of this type.
		kWeightsFilename, ///< The weights file to load.  There should only be 1 parameter of this type.
		kDirectory		, ///< An existing directory name.
		kFilename		, ///< An existing filename which is not one of the 3 neural network files.
		kOther			, ///< Any other parameter.
	};

	/** Structure returned by @ref Darknet::parse_arguments().
	 *
	 * @see @ref Darknet::Parms
	 *
	 * @since 2024-07-29
	 */
	struct Parm
	{
		int idx;				///< Index into the original @p argv[] array.
		EParmType type;			///< What we determined this parameter represents.
		std::string original;	///< The original text for this parameter.  Use @ref string instead.
		std::string string;		///< The value to use for this parameter.
	};

	/** Structure returned by @ref Darknet::parse_arguments().
	 *
	 * @since 2024-07-29
	 */
	using Parms = std::vector<Parm>;

	/** Display a few lines of text with some version information.
	 *
	 * @since 2024-08-29
	 */
	void show_version_info();

	/** Parse common %Darknet command-line parameters with the values from @p argc and @p argv in @p main().
	 * Output can be used with @ref Darknet::load_neural_network().
	 *
	 * This function will attempt to identify the following:
	 *
	 * - @p .cfg files (%Darknet configuration)
	 * - @p .names files (%Darknet classes)
	 * - @p .weights files (%Darknet weights)
	 * - all other files or subdirectories
	 * - any other non-file parameters
	 *
	 * If the @p .names or @p .weights files were not found, then attempts will be made to locate a suitable file to use
	 * based on the given @p .cfg file.  This means as long as the 3 %Darknet files are named in a similar way, there is
	 * no need to specify all 3 files.  The @p .cfg file is enough to find the neural network.
	 *
	 * In addition, if no %Darknet files were found but a "stem" was specified, then this stem will be used to attempt and
	 * find all the necessary files.  For example, if the following files exist:
	 *
	 * - @p animals.cfg
	 * - @p animals.names
	 * - @p animals_best.weights
	 *
	 * These files will be found if this function is called with the parameter @p "anim", since all the files begin with
	 * that stem.
	 *
	 * Given the example files listed above, the following commands are interpreted the exact same way:
	 *
	 * ~~~~.sh
	 *    darknet_01_inference_images animals.cfg animals.names animals_best.weights dog.jpg
	 *    darknet_01_inference_images animals.names dog.jpg animals_best.weights animals.cfg
	 *    darknet_01_inference_images animals.cfg animals.names dog.jpg
	 *    darknet_01_inference_images animals.cfg dog.jpg
	 *    darknet_01_inference_images ani dog.jpg
	 * ~~~~
	 *
	 * @since 2024-07-29
	 */
	Darknet::Parms parse_arguments(int argc, char * argv[]);

	/** Similar to the other @ref Darknet::parse_arguments(), but uses a vector of strings as input.
	 * Output can be used with @ref Darknet::load_neural_network().
	 *
	 * @note See the full description in the other @ref Darknet::parse_arguments().
	 *
	 * @since 2024-07-29
	 */
	Darknet::Parms parse_arguments(const Darknet::VStr & v);

	/** Set the @ref Darknet::CfgAndState::is_verbose flag.  When enabled, extra information will be sent to @p STDOUT.
	 * Default value is @p false.
	 *
	 * @note Disabling @p verbose will also disable @p trace.
	 *
	 * @see @ref Darknet::set_trace()
	 * @see @ref darknet_set_verbose()
	 *
	 * @since 2024-07-14
	 */
	void set_verbose(const bool flag);

	/** Set the @ref Darknet::CfgAndState::is_trace flag.  When enabled, debug information will be sent to @p STDOUT.
	 * Default value is @p false.
	 *
	 * @note Enabling @p trace will also enable @p verbose.
	 *
	 * @see @ref Darknet::set_verbose()
	 * @see @ref darknet_set_trace()
	 *
	 * @since 2024-07-14
	 */
	void set_trace(const bool flag);

	/** Set the GPU index to use.  This may be set to @p -1 to indicate no GPU has been selected, or may be set to a 0-based
	 * GPU.  In normal situations, this must be set prior to calling @ref Darknet::load_neural_network() where the GPU
	 * is usually initialized.
	 *
	 * If set to @p -1 and %Darknet was compiled with support for CUDA GPUs, then the GPU index will default to @p 0 when
	 * @ref Darknet::load_neural_network() is called.  If no CUDA GPU is detected, then the GPU index will be set to @p -1
	 * and the CPU will be used instead.
	 *
	 * Default is @p -1.
	 *
	 * @since 2024-07-25
	 */
	void set_gpu_index(int idx);

	/** Detection threshold to use when @ref Darknet::predict() is called.
	 *
	 * Default is @p 0.25.
	 *
	 * @see @ref Darknet::NetworkDetails::detection_threshold
	 *
	 * @since 2024-07-24
	 */
	void set_detection_threshold(Darknet::NetworkPtr ptr, float threshold);

	/** Non-maximal suppression threshold to use when @ref Darknet::predict() is called.
	 *
	 * Default is @p 0.45.
	 *
	 * @see @ref Darknet::NetworkDetails::non_maximal_suppression_threshold
	 *
	 * @since 2024-07-24
	 */
	void set_non_maximal_suppression_threshold(Darknet::NetworkPtr ptr, float threshold);

	/** Fix out-of-bound values returned by @ref Darknet::predict() for objects near the edges of images.
	 * When set to @p true, this will ensure that normalized coordinates are between @p 0.0 and @p 1.0, and do not extend
	 * beyond the borders of the image or video frame.
	 *
	 * Default is @p true.
	 *
	 * @see @ref Darknet::NetworkDetails::fix_out_of_bound_normalized_coordinates
	 *
	 * @since 2024-07-25
	 */
	void fix_out_of_bound_values(Darknet::NetworkPtr ptr, const bool toggle);

	/** Set the font characteristics to use when drawing the bounding boxes and labels in either @ref Darknet::annotate()
	 * or @ref Darknet::predict_and_annotate().
	 *
	 * @param [in] ptr Neural network pointer obtained from @ref Darknet::load_neural_network().
	 *
	 * @param [in] line_type should be @p cv::LineTypes::LINE_4, @p cv::LineTypes::LINE_8, or @p cv::LineTypes::CV_LINE_AA.
	 * @p LINE_4 is the fastest but lower quality, while @p LINE_AA (anti-alias) is the slowest with highest quality.
	 * Default is @p LINE_4.  Also see @ref set_annotation_line_type() which modifies the same setting.
	 *
	 * @param [in] font_face is the OpenCV built-in font to use.  Default is @p cv::HersheyFonts::FONT_HERSHEY_PLAIN.
	 *
	 * @param [in] font_thickness determines how thick the lines are drawn when the text is rendered.  Default is @p 1.
	 *
	 * @param [in] font_scale determines how large or small the text is rendered.  For example, this could be set to @p 0.5
	 * for extremely small text, and 1.75 for large text.  Default is @p 1.0.
	 *
	 * @see @ref Darknet::NetworkDetails::cv_line_type
	 * @see @ref Darknet::NetworkDetails::cv_font_face
	 * @see @ref Darknet::NetworkDetails::cv_font_thickness
	 * @see @ref Darknet::NetworkDetails::cv_font_scale
	 *
	 * @since 2024-07-30
	 */
	void set_annotation_font(Darknet::NetworkPtr ptr, const cv::LineTypes line_type, const cv::HersheyFonts font_face, const int font_thickness, const double font_scale);

	/** The OpenCV line type can impact performance.  Anti-aliased lines are expensive to draw.  Possible options for
	 * @p line_type is @p cv::LineTypes::LINE_4, @p cv::LineTypes::LINE_8, or @p cv::LineTypes::CV_LINE_AA.  @p LINE_4 is
	 * the fastest but lower quality, while @p LINE_AA (anti-alias) is the slowest with highest quality.
	 * Default is @p LINE_4.
	 *
	 * This setting can also be modified with @ref set_annotation_font().
	 *
	 * @since 2024-09-03
	 */
	void set_annotation_line_type(Darknet::NetworkPtr ptr, const cv::LineTypes line_type);

	/** This determines if annotations are drawn as circles or rounded rectangles in either @ref Darknet::annotate()
	 * or @ref Darknet::predict_and_annotate().  The defaul is to use square -- not rounded -- bounding boxes.
	 *
	 * @param [in] ptr Neural network pointer obtained from @ref Darknet::load_neural_network().
	 *
	 * @param [in] toggle Determines if rounded corners are used.  The default is @p false in which case normal "square"
	 * bounding boxes are used.
	 *
	 * @param [in] roundness Determines how large the rounded corners will appear.  The value must be between @p 0.0
	 * (small rounded corners) and @p 1.0 (large rounded corners).  At the extreme of @p 1.0, the bounding box will
	 * appear as a circle.  The default is @p 0.5, but will only take effect if @p rounded is also set to @p true.
	 *
	 * @since 2024-07-30
	 */
	void set_rounded_corner_bounding_boxes(Darknet::NetworkPtr ptr, const bool toggle, const float roundness);

	/** Determines if bounding boxes are drawn when calling either @ref Darknet::annotate() or
	 * @ref Darknet::predict_and_annotate().  The default is @p true.
	 *
	 * @since 2024-07-30
	 */
	void set_annotation_draw_bb(Darknet::NetworkPtr ptr, const bool toggle);

	/** Determines if text labels are drawn above the bounding boxes when calling either @ref Darknet::annotate() or
	 * @ref Darknet::predict_and_annotate().  The default is @p true.
	 *
	 * @since 2024-07-30
	 */
	void set_annotation_draw_label(Darknet::NetworkPtr ptr, const bool toggle);

	/** Load a neural network (.cfg) and the corresponding weights file.  Remember to call
	 * @ref Darknet::free_neural_network() once the neural network is no longer needed.
	 *
	 * @since 2024-07-24
	 */
	Darknet::NetworkPtr load_neural_network(const std::filesystem::path & cfg_filename, const std::filesystem::path & names_filename, const std::filesystem::path & weights_filename);

	/** Load a neural network.  Remember to call @ref Darknet::free_neural_network() once the neural network is no longer needed.
	 * @see @ref Darknet::parse_arguments()
	 * @since 2024-07-29
	 */
	Darknet::NetworkPtr load_neural_network(Darknet::Parms & parms);

	/** Free the neural network pointer allocated in @ref Darknet::load_neural_network().  Does nothing if the pointer has
	 * already been freed.  Will reset the pointer to @p nullptr once the structure has been freed.
	 *
	 * You should call this once you are done with %Darknet to avoid memory leaks in your application.
	 *
	 * @since 2024-07-25
	 */
	void free_neural_network(Darknet::NetworkPtr & ptr);

	/// Get the network dimensions (width, height, channels).  @since 2024-07-25
	void network_dimensions(Darknet::NetworkPtr & ptr, int & w, int & h, int & c);

	/** A much-simplified version of the old API structure @ref DarknetDetection.
	 *
	 * @see @ref Predictions
	 * @see @ref DarknetDetection
	 *
	 * @since 2024-07-24
	 */
	struct Prediction
	{
		int best_class; ///< Zero-based class index, or @p -1 if nothing was found in an image.
		Probabilities prob; ///< The probability for each object.  Only non-zero values are kept.
		cv::Point2f normalized_point; ///< The center point of the object.  This value is normalized and must be multiplied by the image dimensions.
		cv::Size2f normalized_size; ///< The dimensions of the object.  This value is normalized and must be multiplied by the image dimensions.
		cv::Rect rect; ///< The de-normalized bounding box, where the coordinates have been multiplied by the original image width and height.
	};

	/** Each image or video frame may contain many predictions.  These predictions are stored in a vector in no particular
	 * order.
	 *
	 * @see @ref Darknet::predict()
	 *
	 * @since 2024-07-24
	 */
	using Predictions = std::vector<Prediction>;

	/** Get %Darknet to look at the given image or video frame and return all predictions.
	 *
	 * This is similar to the other @ref Darknet::predict() that takes a @p Darknet::Image object as input.
	 *
	 * OpenCV @p cv::Mat images (and video frames) are typically stored in BGR format, not RGB.  This function expects the
	 * images to be in the usual BGR format for 3-channel networks.
	 *
	 * @since 2024-07-24
	 */
	Predictions predict(const Darknet::NetworkPtr ptr, const cv::Mat & mat);

	/** Get %Darknet to look at the given image or video frame and return all predictions.
	 *
	 * The provided image must be in %Darknet's RGB image format.  This is similar to the other @ref predict() that takes
	 * a @p cv::Mat object as input.
	 *
	 * @note If the original image size is known, it is best to pass it in so the bounding boxes can be scaled to those
	 * dimensions.
	 *
	 * @since 2024-08-02
	 */
	Predictions predict(const Darknet::NetworkPtr ptr, Darknet::Image & img, cv::Size original_image_size = cv::Size(0, 0));

	/** Get %Darknet to look at the given image and return all predictions.  The image must be in a format supported by
	 * OpenCV, such as @p JPG or @p PNG.
	 *
	 * @since 2024-07-25
	 */
	Predictions predict(const Darknet::NetworkPtr ptr, const std::filesystem::path & image_filename);

	/** Annotate the given image using the predictions from @ref Darknet::predict().
	 *
	 * @see @ref Darknet::predict_and_annotate()
	 *
	 * @since 2024-07-30
	 */
	cv::Mat annotate(const Darknet::NetworkPtr ptr, const Predictions & predictions, cv::Mat mat);

	/** Combination of @ref Darknet::predict() and @ref Darknet::annotate().
	 *
	 * Remember to clone @p mat prior to calling @p predict_and_annotate() if you need to keep a copy of the original image.
	 *
	 * @since 2024-07-30
	 */
	Predictions predict_and_annotate(const Darknet::NetworkPtr ptr, cv::Mat mat);

	/** Get access to the vector of names read from the .names file when the configuration was loaded.
	 *
	 * @since 2024-08-06
	 */
	const Darknet::VStr & get_class_names(const Darknet::NetworkPtr ptr);

	/** Get access to the vector of colours assigned to each class when the @p .names file was loaded.
	 *
	 * @since 2024-08-06
	 */
	const Darknet::VScalars & get_class_colours(const Darknet::NetworkPtr ptr);

	/** Set the colours to use when drawing annotations.  The colours are in OpenCV's usual BGR format, not RGB.  So pure
	 * red for example is @p "{0, 0, 255}" while pure blue would be @p "{255, 0, 0}".  The middle value is green.
	 *
	 * @returns the final colours used, same as if @ref get_class_colours() had been called.
	 *
	 * @since 2024-09-22
	 */
	const Darknet::VScalars & set_class_colours(Darknet::NetworkPtr ptr, const Darknet::VScalars & colours);

	/** Get the filename of the configuration file that was used to load this neural network.
	 *
	 * @since 2024-08-29
	 */
	std::filesystem::path get_config_filename(const Darknet::NetworkPtr ptr);

	/** Get the filename of the names file that was used to load this neural network.
	 *
	 * @since 2024-08-29
	 */
	std::filesystem::path get_names_filename(const Darknet::NetworkPtr ptr);

	/** Get the filename of the weights file that was used to load this neural network.
	 *
	 * @since 2024-08-29
	 */
	std::filesystem::path get_weights_filename(const Darknet::NetworkPtr ptr);

	/** Resize the image as close as we can to the given size, but keep the aspect ratio the same as the original image.
	 *
	 * This method will modify the image that is passed in, so clone it beforehand if you need a copy of the original.
	 *
	 * Several notes in regards to the resize method:
	 *
	 * @li @p cv::InterpolationFlags::INTER_NEAREST is the fastest resize method, but the quality is poor.
	 * @li @p cv::InterpolationFlags::INTER_AREA is good when shrinking an image.
	 * @li @p cv::InterpolationFlags::INTER_CUBIC is good when growing an image
	 * @li @p cv::InterpolationFlags::INTER_LINEAR is similar to INTER_CUBIC, but faster
	 *
	 * @since 2024-09-05
	 */
	cv::Mat resize_keeping_aspect_ratio(cv::Mat & mat, cv::Size desired_size, const cv::InterpolationFlags method = cv::InterpolationFlags::INTER_NEAREST);

	/** Calculate intersection-over-union given 2 OpenCV rectangles.  Will return a value between @p 0.0f and @p 1.0f.
	 *
	 * @see @ref box_iou()
	 *
	 * @since 2024-09-07
	 */
	float iou(const cv::Rect & lhs, const cv::Rect & rhs);

	/** Return the set of classes which %Darknet must ignore.  Default set is empty.
	 *
	 * @see @ref Darknet::skipped_classes()
	 * @see @ref Darknet::clear_skipped_classes()
	 * @see @ref Darknet::add_skipped_class()
	 * @see @ref Darknet::del_skipped_class()
	 *
	 * @since 2024-10-07
	 */
	SInt skipped_classes(const Darknet::NetworkPtr ptr);

	/** Set the classes which %Darknet must ignore, completely over-writing all previous values.  If you'd rather add a
	 * single class at a time, call @ref add_skipped_class() which can be called repeatedly without overwriting previous
	 * settings.
	 *
	 * @see @ref Darknet::skipped_classes()
	 * @see @ref Darknet::clear_skipped_classes()
	 * @see @ref Darknet::add_skipped_class()
	 * @see @ref Darknet::del_skipped_class()
	 *
	 * @since 2024-10-07
	 */
	SInt skipped_classes(Darknet::NetworkPtr ptr, const SInt & classes_to_skip);

	/** Clear the set of classes which %Darknet must ignore.  The default is for Darknet/YOLO to not skip any classes,
	 * as if this function has been called.
	 *
	 * @see @ref Darknet::skipped_classes()
	 * @see @ref Darknet::add_skipped_class()
	 * @see @ref Darknet::del_skipped_class()
	 *
	 * @since 2024-10-07
	 */
	SInt clear_skipped_classes(Darknet::NetworkPtr ptr);

	/** Add the given class index to the set of classes that %Darknet must ignore.  This may be called multiple times if
	 * you have many classes you want skipped, or you can call @ref skipped_classes() if you want to set them all at once.
	 *
	 * @see @ref Darknet::skipped_classes()
	 * @see @ref Darknet::clear_skipped_classes()
	 * @see @ref Darknet::del_skipped_class()
	 *
	 * @since 2024-10-07
	 */
	SInt add_skipped_class(Darknet::NetworkPtr ptr, const int class_to_skip);

	/** Remove the given class index from the set of classes that %Darknet must ignore.  This may be called multiple times
	 * if there are several class indexes you'll like to restore.
	 *
	 * @see @ref Darknet::skipped_classes()
	 * @see @ref Darknet::clear_skipped_classes()
	 * @see @ref Darknet::add_skipped_class()
	 *
	 * @since 2024-10-07
	 */
	SInt del_skipped_class(Darknet::NetworkPtr ptr, const int class_to_include);

	/** Display some information about this specific prediction.
	 *
	 * Use like this:
	 *
	 * ~~~~
	 *     std::cout << prediction << std::endl;
	 * ~~~~
	 *
	 * @since 2024-08-06
	 */
	std::ostream & operator<<(std::ostream & os, const Darknet::Prediction & pred);

	/** Display some information about all the predictions.
	 *
	 * Use like this:
	 *
	 * ~~~~
	 *     const auto results = Darknet::predict(ptr, mat);
	 *     std::cout << results << std::endl;
	 * ~~~~
	 *
	 * Output would look similar to this:
	 *
	 * ~~~~.txt
	 *     prediction results: 5
	 *     -> 1/5: #4 prob=0.999923 x=423 y=35 w=206 h=221 entries=1
	 *     -> 2/5: #3 prob=0.999828 x=285 y=85 w=138 h=135 entries=1
	 *     -> 3/5: #2 prob=0.988380 x=527 y=128 w=31 h=28 entries=1
	 *     -> 4/5: #1 prob=0.996240 x=498 y=187 w=26 h=29 entries=1
	 *     -> 5/5: #0 prob=0.994430 x=46 y=127 w=43 h=40 entries=1
	 * ~~~~
	 *
	 * @since 2024-08-06
	 */
	std::ostream & operator<<(std::ostream & os, const Darknet::Predictions & preds);

	/** Create several @p CV_32FC1 (array of 32-bit floats, single channel) @p cv::Mat objects representing heatmaps
	 * obtained from the YOLO layers in the network.  There is a heatmap for each class, and then another heatmap which
	 * combines all classes.  The class index is used to store each heatmap in the @p std::map result, while the combined
	 * heatmap is stored with an index of @p -1.
	 *
	 * The dimensions of each heatmap will match the network dimensions.  The values returned in each heatmap will be
	 * between @p 0.0f and some relatively small float value which may be larger than @p 1.0f.
	 *
	 * The heatmaps can be shown directly using OpenCV's @p cv::imshow(), but the results will appear much better if the
	 * values are normalized and coloured, similar to how it is done in @ref Darknet::visualize_heatmap().
	 *
	 * @since 2024-11-09
	 */
	MMats create_yolo_heatmaps(Darknet::Network * net, const float sigma = 15.0f);

	/** Convert a heatmap created with @ref Darknet::create_yolo_heatmaps() to an easy-to-view image.  This will normalize
	 * the image, and apply some false colours.  The OpenCV colour map @p COLORMAP_JET is quite colourful; others that can
	 * be tried include @p COLORMAP_RAINBOW, @p COLORMAP_HOT, @p COLORMAP_TURBO, and many others.
	 *
	 * @since 2024-11-09
	 */
	cv::Mat visualize_heatmap(const cv::Mat & heatmap, const cv::ColormapTypes colourmap = cv::ColormapTypes::COLORMAP_JET);
}
