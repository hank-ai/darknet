#pragma once

/** @file
 * Most of the Darknet code is plain old C.  This header file is the semi-public interface between that old C code and
 * the new C++ code.
 *
 * @warning This file is the @p "C" functions.  Also see the file darknet_utils.hpp for the C++ functions.
 */

#ifndef __cplusplus
#include <stdbool.h>
#else
extern "C" {
#endif

enum EColour
{
	kNormal			,
	kBlack			,
	kRed			,
	kGreen			,
	kBrown			,
	kBlue			,
	kMagenta		,
	kCyan			,
	kLightGrey		,
	kDarkGrey		,
	kBrightRed		,
	kBrightGreen	,
	kYellow			,
	kBrightBlue		,
	kBrightMagenta	,
	kBrightCyan		,
	kBrightWhite	,
	kMax
};

/// Determines if colour output will be used.  Defaults to @p true on Linux and @p false on Windows.
extern bool colour_is_enabled;

/// Array of text strings with the VT100/ANSI escape codes needed to display colour output.
extern const char * const ansi_colours[kMax];

/** Remember all of the entries in the .names file, so we don't have to keep re-loading it or passing it around.
 * @see @ref class_names
 * @see @ref class_colours
 */
void remember_class_names(char ** names, const int count);

void display_loaded_images(const int images, const double time);
void display_iteration_summary(const int iteration, const float loss, const float avg_loss, const float rate, const double time, const int images, const double avg_time);
void display_last_accuracy(const float iou_thresh, const float mean_average_precision, const float best_map);

/// Display the given message in bright red (if colour is enabled).
void display_error_msg(const char * const msg);

/// Display the given message in yellow (if colour is enabled).
void display_warning_msg(const char * const msg);

/// Use VT100/ANSI codes to update the console title during training.
void update_console_title(const int iteration, const int max_batches, const float loss, const float current_map, const float best_map, const double seconds_remaining);

/// Determine if a filename exists.
bool file_exists(const char * const filename);

/// Setup the new C++ charts.  This is called once just prior to starting training.  @see @ref Chart
void initialize_new_charts(const int max_batches, const float max_img_loss);

/// Update the new C++ charts with the given loss and mAP% accuracy value.  This is called at every iteration.  @see @ref Chart
void update_loss_in_new_charts(const int current_iteration, const float loss, const double hours_remaining, const bool dont_show);

void update_accuracy_in_new_charts(const int class_index, const float accuracy);

#ifdef __cplusplus
}
#endif
