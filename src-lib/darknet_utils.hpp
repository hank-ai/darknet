#pragma once

/** @file
 * Collection of helper and utility functions for Darknet.
 */


#include "darknet_internal.hpp"


namespace Darknet
{
	/// The names stored in the .names file.  @see @ref remember_class_names()
	extern std::vector<std::string> class_names;

	/// The colour to use for each class.  @see @ref remember_class_names()
	extern std::vector<cv::Scalar> class_colours;

	/** Convert the given text to plain alphanumeric ASCII string.  Remove whitespace, keep just alphanumeric and underscore.
	 * Good to use as a base for a filename.
	 */
	std::string text_to_simple_label(std::string txt);

	/** Remember all of the entries in the .names file, so we don't have to keep re-loading it or passing it around.
	 * @see @ref class_names
	 * @see @ref class_colours
	 */
	void remember_class_names(char ** names, const int count);

	/// Setup the new C++ charts.  This is called once just prior to starting training.  @see @ref Chart
	void initialize_new_charts(const int max_batches, const float max_img_loss);

	/// Update the new C++ charts with the given loss and mAP% accuracy value.  This is called at every iteration.  @see @ref Chart
	void update_loss_in_new_charts(const int current_iteration, const float loss, const float seconds_remaining, const bool dont_show);

	void update_accuracy_in_new_charts(const int class_index, const float accuracy);
}
