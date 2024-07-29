#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

/** @file
 * Include this file to get access to the new v3 Darknet/YOLO C++ API.
 */

#include <filesystem>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "darknet.h"


/// The namespace for the C++ Darknet API.
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
	using NetworkPtr	= ::NetworkPtr;
	/// @}

	/** The likelyhood of a specific object class having been predicted.  This map contains all of the non-zero values.
	 * The key is the zero-based class indexes, and the values are the confidences for the classes, between @p 0.0f and
	 * @p 1.0f.
	 *
	 * For example, if "car" is class #2 and "truck" is class #3, and Darknet is 95% certain it found a car and 82% certain
	 * it found a truck, the map would then contain
	 * @p "{ {2, 0.95}, {3, 0.82} }".
	 *
	 * @see @ref Prediction
	 *
	 * @since 2024-07-24
	 */
	using Probabilities = std::map<int, float>;

	/** When parsing command-line parameters, each parameter is assigned a "type".
	 * @see @ref Darknet::Parm
	 * @see @ref parse_arguments()
	 * @since 2024-07-29
	 */
	enum class EParmType
	{
		kUnknown		,
		kCfgFilename	, ///< The configuration file to load.  There should only be 1 parameter of this type.
		kNamesFilename	, ///< The names file to load.  There should only be 1 parameter of this type.
		kWeightsFilename, ///< The weights file to load.  There should only be 1 parameter of this type.
		kDirectory		, ///< An existing directory name.
		kFilename		, ///< An existing filename which is not one of the 3 neural network files.
		kOther			, ///< Any other parameter.
	};

	/** Structure returned by @ref parse_arguments().
	 * @see @ref Parms
	 * @since 2024-07-29
	 */
	struct Parm
	{
		int idx;				///< Index into the original @p argv[] array.
		EParmType type;
		std::string original;
		std::string string;
	};

	/** Structure returned by @ref parse_arguments().
	 * @see @ref Parms
	 * @since 2024-07-29
	 */
	using Parms = std::vector<Parm>;

	/** Parse common Darknet command-line parameters with the values from @p argc and @p argv in @p main()
	 * @since 2024-07-29
	 */
	Darknet::Parms parse_arguments(int argc, char * argv[]);

	/// Similar to the other @ref parse_arguments().
	Darknet::Parms parse_arguments(const Darknet::VStr & v);

	/** Go through @p argv and find the likely filenames and flags.  Any parameter which is not a filename will be stored
	 * as a flag.
	 *
	 * @returns @p true if a @p .cfg and @p .weights file was found (meaning a neural network can be loaded).
	 * @returns @p false if both a @p .cfg and @p .weights were not found (meaning we don't have enough to load a neural network).
	 *
	 * @since 2024-07-26
	 */
	bool parse_neural_network_args(int argc, char * argv[], std::filesystem::path & cfg, std::filesystem::path & names, std::filesystem::path & weights, Darknet::VStr & other_filenames, Darknet::VStr & misc);

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
	 * GPU.  In normal situations, this must be set prior to calling @ref Darknet::load_network_and_weights() where the GPU
	 * is usually initialized.
	 *
	 * If set to @p -1 and %Darknet was compiled with support for CUDA GPUs, then the GPU index will default to @p 0 when
	 * @ref Darknet::load_network_and_weights() is called.  If no CUDA GPU is detected, then the GPU index will be set to
	 * @p -1 and the CPU will be used instead.
	 *
	 * Default is @p -1.
	 *
	 * @since 2024-07-25
	 */
	void set_gpu_index(int idx);

	/// Detection threshold to use when @ref Darknet::predict() is called.  Default is @p 0.25.  @since 2024-07-24
	void set_detection_threshold(float threshold);

	/// Non-maximal suppression threshold to use when @ref Darknet::predict() is called.  Default is @p 0.45.  @since 2024-07-24
	void set_non_maximal_suppression_threshold(float threshold);

	/** Fix out-of-bound values returned by @ref Darknet::predict() for objects near the edges of images.
	 * When set to @p true, this will ensure that normalized coordinates are between @p 0.0 and @p 1.0, and do not extend
	 * beyond the borders of the image or video frame.  Default is @p true.
	 *
	 * @since 2024-07-25
	 */
	void fix_out_of_bound_values(const bool toggle);

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
	Darknet::NetworkPtr load_neural_network(const Darknet::Parms & parms);

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

	/// @see @ref detection  @since 2024-07-24
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
	 * @see @ref predict()
	 *
	 * @since 2024-07-24
	 */
	using Predictions = std::vector<Prediction>;

	/** Get %Darknet to look at the given image or video frame and return all predictions.
	 *
	 * OpenCV @p cv::Mat images (and video frames) are typically stored in BGR format, not RGB.  This function expects the
	 * images to be in the usual BGR format for 3-channel networks.
	 *
	 * @since 2024-07-24
	 */
	Predictions predict(const Darknet::NetworkPtr ptr, cv::Mat mat);

	/** Get %Darknet to look at the given image.  The image must be in a format supported by OpenCV, such as JPG or PNG.
	 *
	 * @since 2024-07-25
	 */
	Predictions predict(const Darknet::NetworkPtr ptr, const std::filesystem::path & image_filename);

	std::ostream & operator<<(std::ostream & os, const Darknet::Prediction & pred);
	std::ostream & operator<<(std::ostream & os, const Darknet::Predictions & preds);
}


#include "darknet_version.h"
#include "darknet_enums.hpp"
#include "darknet_cfg.hpp"
