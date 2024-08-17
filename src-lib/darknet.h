#pragma once

/** @file
 * Include this file to get access to the Darknet/YOLO C API.
 *
 * @li The new (V3, July 2024) Darknet/YOLO C API is at the top of the file.
 * @li The original V2 Darknet/YOLO C API is at the bottom of this file.
 * @li The new (V3, July 2024) Darknet/YOLO C++ API is in the file @ref darknet.hpp.
 */

#ifdef __cplusplus
extern "C"
{
#endif


#include "darknet_version.h"


/// An opaque pointer to a @ref Darknet::Network object, without needing to expose the internals of the network structure.
typedef void* DarknetNetworkPtr;

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




/** The old C API did not have @p "darknet" in the function names nor the structures returned.  It defined things like
 * @p network and @p image in the global namespace, which can cause problems since those are common words.  By default
 * this old API is not exposed.  If you're using some old software that expects the original "C" API in the darknet
 * library, then make sure you @p "#define DARKNET_INCLUDE_ORIGINAL_API" before you include this header file.
 *
 * Internally, %Darknet still uses the old @p C API, so @ref darknet_internal.hpp does define this macro.
 */
#ifdef DARKNET_INCLUDE_ORIGINAL_API


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
 * @see @ref Darknet::predict()
 */
float * network_predict_ptr(DarknetNetworkPtr * ptr, float * input);


#endif // DARKNET_INCLUDE_ORIGINAL_API


#ifdef __cplusplus
}
#endif
