#pragma once

/** @file
 * This is the C++ side of the new Darknet utility functions.  See @p darknet_utils.h for the @p "C" interface.
 */


#ifndef __cplusplus
#error Attempting to include the C++ header file from within C code.
#endif


#include "darknet_utils.h"
#include <string>
#include <vector>

#ifdef OPENCV
#include <opencv2/opencv.hpp>
#endif


/// The names stored in the .names file.  @see @ref remember_class_names()
extern std::vector<std::string> class_names;

#ifdef OPENCV
/// The colour to use for each class.  @see @ref remember_class_names()
extern std::vector<cv::Scalar> class_colours;
#endif


std::string in_colour(const EColour colour, const int i);
std::string in_colour(const EColour colour, const float f);
std::string in_colour(const EColour colour, const double d);
std::string in_colour(const EColour colour, const std::string & msg);


/** The time used by Darknet is a double, generated from @ref what_time_is_it_now().  It is the number of seconds since
 * @p epoch with milliseconds and microseconds as decimals.  This function will format one of these @p double using the
 * most intelligent unit necessary.
 */
std::string format_time(const double & seconds_remaining);


/// Format the time remaining using simple-to-read text.  The time must be in @em seconds.
std::string format_time_remaining(const float & seconds_remaining);


/// Format the loss combined with ANSI colours.
std::string format_loss(const double & loss);


/// Format the mAP% accuracy with ANSI colours.
std::string format_map_accuracy(const float & accuracy);


/** Convert the given text to plain alphanumeric ASCII string.  Remove whitespace, keep just alphanumeric and underscore.
 * Good to use as a base for a filename.
 */
std::string text_to_simple_label(std::string txt);
