/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

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

	/// Format the loss combined with ANSI colours.
	std::string format_loss(const double & loss);

	/// Format the mAP% accuracy with ANSI colours.
	std::string format_map_accuracy(const float & accuracy);

	std::string format_in_colour(const std::string & str, const EColour & colour, const int & len);
	std::string format_in_colour(const int & i, const EColour & colour, const size_t & len);
	std::string format_in_colour(const size_t & st, const EColour & colour, const size_t & len);
	std::string format_in_colour(const float & f, const EColour & colour, const size_t & len);
	std::string format_in_colour(const float & f, const size_t & len, const bool inverted = false);

	std::string format_map_confusion_matrix_values(
			const int class_id,
			std::string name, // on purpose not by reference since we can end up modifying it
			const float & average_precision,
			const int & tp,
			const int & fn,
			const int & fp,
			const int & tn,
			const float & accuracy,
			const float & error_rate,
			const float & precision,
			const float & recall,
			const float & specificity,
			const float & false_pos_rate);

	std::string format_layer_summary(
			const size_t idx,
			const Darknet::CfgSection & section,
			const Darknet::Layer & l);

	/// Display the given message in bright red (if colour is enabled).  The message is not linefeed terminated.
	void display_error_msg(const std::string & msg);

	/// Display the given message in yellow (if colour is enabled).  The message is not linefeed terminated.
	void display_warning_msg(const std::string & msg);
}
