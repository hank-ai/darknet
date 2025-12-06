#pragma once

/** @file
 * Collection of helper and utility functions for Darknet.
 */


#include "darknet_internal.hpp"


namespace Darknet
{
	/// Convert to lowercase and remove all but alphanumerics.
	std::string convert_to_lowercase_alphanum(const std::string & arg);

	/** Convert the given text to plain alphanumeric ASCII string.  Remove whitespace, keep just alphanumeric and underscore.
	 * Good to use as a base for a filename.
	 */
	std::string text_to_simple_label(std::string txt);

	/// Setup the new C++ charts.  This is called once just prior to starting training.  @see @ref Chart
	void initialize_new_charts(const Darknet::Network & net);

	/// Update the new C++ charts with the given loss and mAP% accuracy value.  This is called at every iteration.  @see @ref Chart
	void update_loss_in_new_charts(const int current_iteration, const float loss, const std::string & time_remaining, const bool dont_show);

	void update_f1_in_new_charts(const int class_idx, const float f1);
	void update_accuracy_in_new_charts(const int class_index, const float accuracy);

	std::string get_command_output(const std::string & cmd);

	void cfg_layers();

	/// Convert fp32 to fp16 (IEEE 754) without having to rely on CUDA (which is not available in CPU-only builds).
	std::uint16_t convert_to_fp16(const float f);
}
