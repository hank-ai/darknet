#pragma once

/** @file
 * This is the C++ side of the new Darknet utility functions.  See @p darknet_utils.h for the @p "C" interface.
 */


#ifndef __cplusplus
#error Attempting to include the C++ header file from within C code.
#endif


#include "darknet_utils.h"
#include <string>


std::string in_colour(const EColour colour, const int i);
std::string in_colour(const EColour colour, const float f);
std::string in_colour(const EColour colour, const double d);
std::string in_colour(const EColour colour, const std::string & msg);


/** The time used by Darknet is a double, generated from @ref what_time_is_it_now().  It is the number of seconds since
 * @p epoch with milliseconds and microseconds as decimals.  This function will format one of these @p double using the
 * most intelligent unit necessary.
 */
std::string format_time(const double & t);


/// Format the time remaining using simple-to-read text.  The time must be in @em seconds.
std::string format_time_remaining(const double & t);


/// Format the loss combined with ANSI colours.
std::string format_loss(const double & loss);


/// Format the mAP% accuracy with ANSI colours.
std::string format_map_accuracy(const float & accuracy);
