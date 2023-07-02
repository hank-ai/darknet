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

#ifdef __cplusplus
}
#endif
