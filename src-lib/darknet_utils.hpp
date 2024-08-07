#pragma once

/** @file
 * Collection of helper and utility functions for Darknet.
 */


#include "darknet_internal.hpp"


namespace Darknet
{
	/// The names stored in the .names file.  @see @ref remember_class_names()
	extern std::vector<std::string> xxxclass_names;

	/// The colour to use for each class.  @see @ref remember_class_names()
	extern std::vector<cv::Scalar> xxxclass_colours;

	/// Convert to lowercase and remove all but alphanumerics.
	std::string convert_to_lowercase_alphanum(const std::string & arg);

	/** Convert the given text to plain alphanumeric ASCII string.  Remove whitespace, keep just alphanumeric and underscore.
	 * Good to use as a base for a filename.
	 */
	std::string text_to_simple_label(std::string txt);

	/// Setup the new C++ charts.  This is called once just prior to starting training.  @see @ref Chart
	void initialize_new_charts(const network & net);

	/// Update the new C++ charts with the given loss and mAP% accuracy value.  This is called at every iteration.  @see @ref Chart
	void update_loss_in_new_charts(const int current_iteration, const float loss, const float seconds_remaining, const bool dont_show);

	void update_accuracy_in_new_charts(const int class_index, const float accuracy);

	std::string get_command_output(const std::string & cmd);

	void cfg_layers();
}
