#pragma once

#include "darknet_internal.hpp"


namespace Darknet
{
	enum EColour
	{
		kNormal			= 0,
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
	};

	std::string in_colour(const EColour colour, const int i);
	std::string in_colour(const EColour colour, const float f);
	std::string in_colour(const EColour colour, const double d);
	std::string in_colour(const EColour colour, const std::string & msg);
	std::string in_colour(const EColour colour);

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

	void display_loaded_images(const int images, const double time);
	void display_iteration_summary(const int iteration, const float loss, const float avg_loss, const float rate, const double time, const int images, const float seconds_remaining);
	void display_last_accuracy(const float iou_thresh, const float mean_average_precision, const float best_map);

	/// Display the given message in bright red (if colour is enabled).  The message is not linefeed terminated.
	void display_error_msg(const std::string & msg);

	/// Display the given message in yellow (if colour is enabled).  The message is not linefeed terminated.
	void display_warning_msg(const std::string & msg);
}
