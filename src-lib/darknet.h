/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#pragma once

/** @file
 * Include this file to get access to the Darknet/YOLO C API.
 *
 * @li The original V2 Darknet/YOLO C API is at the bottom of this file (@ref DARKNET_INCLUDE_ORIGINAL_API).
 * @li The new (V3, July 2024) Darknet/YOLO @p C API is at the top of the file.
 * @li The new (V3, July 2024) Darknet/YOLO @p C++ API is in @ref darknet.hpp.
 *
 * https://darknetcv.ai/api/api.html
 *
 * The old C V2 API did not have @p "darknet" in the function names nor the structures returned.  It defined things
 * like @p network and @p image in the global namespace, which can cause problems since those are common words.  By
 * default this old API is no longer exposed.  If you're using some old software that expects the original @p C API
 * in the darknet library, then make sure you @p "#define DARKNET_INCLUDE_ORIGINAL_API" before you include this header
 * file.
 *
 * Internally, %Darknet still uses the old @p C V2 API.
 */


#include "darknet_version.h"


#ifdef __cplusplus
extern "C"
{
#else
#include <stdbool.h>
#endif


/* ******************************* */
/* The "new" V3 C API starts here. */
/* ******************************* */


/// An opaque pointer to a @ref Darknet::Network object, without needing to expose the internals of the network structure.
typedef void* DarknetNetworkPtr;

/// This is the @p C equivalent to @ref Darknet::show_version_info().
void darknet_show_version_info();

/// Return the full version string, such as @p "v3.0-163-g56145bf".  @see @ref DARKNET_VERSION_STRING
const char * darknet_version_string();

/// Return the short version string, such as @p "3.0.163".  @see @ref DARKNET_VERSION_SHORT
const char * darknet_version_short();

/// This is the @p C equivalent to @ref Darknet::set_verbose().
void darknet_set_verbose(const bool flag);

/// This is the @p C equivalent to @ref Darknet::set_trace().
void darknet_set_trace(const bool flag);

/// This is the @p C equivalent to @ref Darknet::set_gpu_index().
void darknet_set_gpu_index(int idx);

/// This is the @p C equivalent to @ref Darknet::set_detection_threshold().
void darknet_set_detection_threshold(DarknetNetworkPtr ptr, float threshold);

/// This is the @p C equivalent to @ref Darknet::set_non_maximal_suppression_threshold().
void darknet_set_non_maximal_suppression_threshold(DarknetNetworkPtr ptr, float threshold);

/// This is the @p C equivalent to @ref Darknet::fix_out_of_bound_values().
void darknet_fix_out_of_bound_values(DarknetNetworkPtr ptr, const bool toggle);

/// This is the @p C equivalent to @ref Darknet::network_dimensions().
void darknet_network_dimensions(DarknetNetworkPtr ptr, int * w, int * h, int * c);

/// This is the @p C equivalent to @ref Darknet::load_neural_network().
DarknetNetworkPtr darknet_load_neural_network(const char * const cfg_filename, const char * const names_filename, const char * const weights_filename);

/// This is the @p C equivalent to @ref Darknet::free_neural_network().
void darknet_free_neural_network(DarknetNetworkPtr * ptr);

/// This is the @p C equivalent to @ref Darknet::clear_skipped_classes().
void darknet_clear_skipped_classes(DarknetNetworkPtr ptr);

/// This is the @p C equivalent to @ref Darknet::add_skipped_class().
void darknet_add_skipped_class(DarknetNetworkPtr ptr, const int class_to_skip);

/// This is the @p C equivalent to @ref Darknet::del_skipped_classes().
void darknet_del_skipped_class(DarknetNetworkPtr ptr, const int class_to_include);

/// This is the @p C equivalent to @ref Darknet::set_output_stream().
void darknet_set_output_stream(const char * const filename);

/// Bounding box used with normalized coordinates (between 0.0 and 1.0).
typedef struct DarknetBox
{
	float x; // center X
	float y; // center Y
	float w;
	float h;
} DarknetBox;

/** Everything %Darknet knows about a specific detection.  This structure is used by the old API.  If using the new API,
 * use @ref Darknet::Prediction instead.
 */
typedef struct DarknetDetection
{
	DarknetBox bbox; ///< bounding boxes are normalized (between 0.0f and 1.0f)
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
} DarknetDetection;

/** The structure @ref DarknetImage is used to store a normalized RGB %Darknet image.  The format is intended to be
 * used for internal use by %Darknet, but there are some situations where it may also be used or referenced externally
 * via the %Darknet API.
 *
 * Moving forward starting with %Darknet V3 (August 2024), where possible the emphasis will be to use OpenCV @p cv::Mat
 * objects in the external-facing API instead of @ref DarknetImage.  Internally, %Darknet will continue to use @p Image.
 *
 * @warning Keep this structure as POD (plain-old-data) since there are many places in the old code where memory for
 * these image objects is calloc'd.  This structure was originally part of the old %Darknet @p C API, which is why it
 * exists this way and not as a C++ class with methods.
 *
 * Within @p data, the image is stored as 3 non-interlaced channels in RGB order.  Each channel is stored as rows, from top to bottom.
 * So a 5x3 RGB image would look like this:
 *
 * ~~~~{.txt}
 *		r r r r r
 *		r r r r r
 *		r r r r r
 *		g g g g g
 *		g g g g g
 *		g g g g g
 *		b b b b b
 *		b b b b b
 *		b b b b b
 * ~~~~
 *
 * A 5x3 pure red image, with a blueish/green square in the bottom-right corner would look like this:
 *
 * ~~~~{.txt}
 *		1.0 1.0 1.0 1.0 1.0 // red channel
 *		1.0 1.0 1.0 0.0 0.0 // note the lack of red in the bottom-right corner
 *		1.0 1.0 1.0 0.0 0.0
 *		0.0 0.0 0.0 0.0 0.0 // green channel
 *		0.0 0.0 0.0 0.5 0.5
 *		0.0 0.0 0.0 0.5 0.5
 *		0.0 0.0 0.0 0.0 0.0 // blue channel
 *		0.0 0.0 0.0 1.0 1.0 // note the blue square in the bottom-right corner
 *		0.0 0.0 0.0 1.0 1.0
 * ~~~~
 *
 * For additional information or to help debug the internals of @p DarknetImage, see @ref Darknet::image_as_debug_string().
 *
 * @see @ref Darknet::load_image()
 * @see @ref Darknet::copy_image()
 * @see @ref make_empty_image()
 * @see @ref make_image()
 * @see @ref Darknet::free_image()
 */
typedef struct DarknetImage
{
	int w;			///< width
	int h;			///< height
	int c;			///< channel
	float *data;	///< normalized floats, the number of which is determined by @p "w * h * c"
} DarknetImage;


/* ******************************* */
/* The "old" V2 C API starts here. */
/* ******************************* */


#ifdef DARKNET_INCLUDE_ORIGINAL_API

/// @see @ref diounms_sort()
typedef enum
{
	DEFAULT_NMS	,
	GREEDY_NMS	,
	DIOU_NMS	,
	CORNERS_NMS
} NMS_KIND;

/// Bounding box used with normalized coordinates (between 0.0 and 1.0).
typedef struct DarknetBox box;

/// Everything %Darknet knows about a specific detection.
typedef struct DarknetDetection detection;

/// Darknet-style image (vector of floats).
typedef struct DarknetImage image;

/** @{ This is part of the original @p C API.  Do not use in new code.
 *
 * @see @ref darknet_load_neural_network()
 * @see @ref Darknet::load_neural_network()
 */
DarknetNetworkPtr load_network(const char * cfg, const char * weights, int clear);
DarknetNetworkPtr load_network_custom(const char * cfg, const char * weights, int clear, int batch);
/// @}

/** This is part of the original @p C API.  Do not use in new code.
 *
 * @see @ref darknet_free_neural_network()
 * @see @ref Darknet::free_neural_network()
 *
 * If you were previously using @ref free_network() from within @p C code, please use @p free_network_ptr() instead by
 * passing in the @em address of the network structure (pointer to the network).
 *
 * @note See the additional comments in @ref free_network().
 */
void free_network_ptr(DarknetNetworkPtr ptr);


/** This is part of the original @p C API.  Do not use in new code.
 *
 * @see @ref darknet_load_neural_network()
 * @see @ref Darknet::load_neural_network()
 *
 * If you were previously using @p calculate_binary_weights() from within @p C code, it used to pass the network by
 * value.  Starting with %Darknet V3 in 2024-08-16, the network is now passed in as a pointer.
 */
void calculate_binary_weights(DarknetNetworkPtr ptr);

/** This is part of the original @p C API.  Do not use in new code.
 *
 * @see @ref network_predict_image()
 * @see @ref Darknet::predict()
 *
 * If you were previously using @ref network_predict() from within @p C code, please use @p network_predict_ptr() instead
 * by passing in the @em address of the network structure (pointer to the network).
 */
float * network_predict_ptr(DarknetNetworkPtr ptr, float * input);

/** This is part of the original @p C API.  Do not use in new code.
 *
 * @see @ref network_predict_ptr()
 * @see @ref Darknet::predict()
 */
float * network_predict_image(DarknetNetworkPtr ptr, const DarknetImage im);

/** This is part of the original @p C API.  Do not use in new code.  See @ref Darknet::predict() for example code that calls this function.
 *
 * You must call @ref free_detections() to free up memory once done with the detections.
 *
 * @see @ref Darknet::predict()
 * @see @ref free_detections()
 */
detection * get_network_boxes(DarknetNetworkPtr ptr, int w, int h, float thresh, float hier, int * map, int relative, int * num, int letter);

/// This is part of the original @p C API.  Do not use in new code.
void free_detections(detection * dets, int n);

/** This is part of the original @p C API.  Make an empty image with the given dimensions.  The data pointer will be
 * @p nullptr.
 *
 * @see @ref free_image()
 * @see @ref make_image()
 */
DarknetImage make_empty_image(int w, int h, int c);

/** This is part of the original @p C API.  Similar to @ref make_empty_image() but the data pointer is fully allocated.
 * @see @ref free_image()
 */
DarknetImage make_image(int w, int h, int c);

/** This is part of the original @p C API.  Free the data pointer that stores the image .  All image objects @em must
 * eventually call either this function or @ref Darknet::free_image() to prevent memory leaks.
 *
 * @see @ref Darknet::free_image()
 *
 * @note The image is passed by value, meaning that the data pointer in the caller's copy of the image will be
 * left dangling.  Be careful not to reference it once this call returns.  Where possible when using C++, call
 * @ref Darknet::free_image() instead.
 */
void free_image(DarknetImage image);

/// This is part of the original @p C API.  Non Maxima Suppression.  See @ref Darknet::predict() for example code that calls this function.
void do_nms_sort(detection * dets, int total, int classes, float thresh);

/// This is part of the original @p C API.  Non Maxima Suppression.
void do_nms_obj(detection * dets, int total, int classes, float thresh);

/// This is part of the original @p C API.  Non Maxima Suppression.
void diounms_sort(detection * dets, int total, int classes, float thresh, NMS_KIND nms_kind, float beta1);

/// This is part of the original @p C API.  Convert from data pointer to a %Darknet image.  Used by the Python API and other language wrappers.
void copy_image_from_bytes(DarknetImage im, char *pdata);

/// This is part of the original @p C API.  This function does nothing when %Darknet was built to run on the CPU.
void cuda_set_device(int n);

/// This is part of the original @p C API.  @see @ref get_network_boxes()  @see @ref make_network_boxes_v3()
detection * make_network_boxes(DarknetNetworkPtr ptr, float thresh, int * num);

/// This is part of the original @p C API.
void free_ptrs(void **ptrs, int n);

/// This is part of the original @p C API.
void reset_rnn(DarknetNetworkPtr ptr);

/// This is part of the original @p C API.  This function exists for the Python API and other language wrappers.
DarknetImage load_image_v2(const char * filename, int desired_width, int desired_height, int channels);

/// This is part of the original @p C API.  @warning Use of letterboxing is no longer a recommended technique.
float * network_predict_image_letterbox(DarknetNetworkPtr ptr, DarknetImage im);

#endif // DARKNET_INCLUDE_ORIGINAL_API


#ifdef __cplusplus
}
#endif
