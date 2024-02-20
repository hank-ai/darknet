#pragma once

/** @file
 * This file contains C++ classes and functions for working with chart.png.  Most of this is called via image_opencv.cpp.
 */


#ifndef __cplusplus
#error Attempting to include the C++ header file from within C code.
#endif


#include <string>
#include <opencv2/opencv.hpp>


class Chart final
{
	public:

		/// Constructor.
		Chart();

		/// Constructor.
		Chart(const std::string & name, const size_t max_batch, const float max_loss);

		/// Destructor.
		~Chart();

		/// Determines if the chart has been initialized.
		bool empty() const;

		Chart & initialize();
		Chart & save_to_disk();

		Chart & update_save_and_display(const int current_iteration, const float loss, const float seconds_remaining, const bool dont_show);

		Chart & update_loss(const int current_iteration, const float loss);

		Chart & update_accuracy(const float accuracy);

		Chart & update_accuracy(const int current_iteration, const float accuracy);

		Chart & update_bottom_text(const float seconds_remaining);

		/** This is the full image of the chart.  It is created by @ref initialize() and kept up-to-date by
		 * @ref update_loss() and @ref update_accuracy() at the end of each iteration.
		 */
		cv::Mat mat;

		/// The RoI for the grid within the full chart mat.
		cv::Rect grid_rect;

		std::string title;		///< e.g., "Loss and Mean Average Precision"
		std::string filename;	///< e.g., "chart.png"

		/** We're going to store max_batches as a float to make the math easier.  This way we don't have to keep converting
		 * it to a float when we multiply or divide using max_batches.
		 */
		float max_batches;

		float max_chart_loss;

		/** The dimensions of the full chart.png image.  Normally, this would be @p 1000x940, and once the border areas have
		 * been removed from 3 sides this leaves us with a grid measuring 880x880, where each large cell measures 88x88 pixels.
		 */
		cv::Size dimensions;

		/// The number of pixels reserved on the left, right, and bottom sides of the image to draw text.  Normally, this is @p 60.
		int grid_offset_in_pixels;

		float previous_loss_iteration;
		float previous_loss_value;

		float previous_map_shown;
		float previous_map_iteration;
		float previous_map_value;
		float max_map_value;
		cv::Scalar map_colour;

		std::time_t started_timestamp;
		std::time_t last_update_timestamp;
		std::time_t last_save_timestamp;
};


/// The central chart, e.g., @p "chart.png".
extern Chart training_chart;

/// Additional training charts for each of the classes.
extern std::vector<Chart> more_charts;
