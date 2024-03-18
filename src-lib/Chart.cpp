#include "Chart.hpp"
#include "darknet_utils.hpp"
#include <filesystem>



Chart training_chart;

std::vector<Chart> more_charts;


Chart::Chart()
{
	max_batches = 0.0f;
	max_chart_loss = 0.0f;

	previous_loss_iteration = 0.0f;
	previous_loss_value = -1.0f;

	previous_map_shown = 0.0f;
	previous_map_iteration = 0.0f;
	previous_map_value = -1.0f;
	max_map_value = 0.0f;
	map_colour = CV_RGB(200, 0, 0);

	started_timestamp = std::time(nullptr);
	last_update_timestamp = 0;
	last_save_timestamp = 0;

	grid_offset_in_pixels = 60;
	dimensions = cv::Size(1000, 940);

	grid_rect = cv::Rect(grid_offset_in_pixels, 0, dimensions.width - grid_offset_in_pixels * 2, dimensions.height - grid_offset_in_pixels);

	return;
}


Chart::Chart(const std::string & name, const size_t max_batch, const float max_loss) :
	Chart()
{
	max_batches = max_batch;
	max_chart_loss = max_loss;

	title = (name.empty() ? "Loss and Mean Average Precision" : name);

	if (title == "Loss and Mean Average Precision")
	{
		// this is the "original" chart, which we want to keep named as "chart.png"
		filename = "chart.png";
	}
	else
	{
		filename = "chart_" + Darknet::text_to_simple_label(title) + ".png";
	}

	initialize();
	update_bottom_text(-1.0);
	save_to_disk();

	return;
}


Chart::~Chart()
{
	return;
}


bool Chart::empty() const
{
	return mat.empty();
}


Chart & Chart::initialize()
{
	mat = cv::Mat(dimensions, CV_8UC3, CV_RGB(255, 255, 255));

	cv::Mat grid = mat(grid_rect);
	const cv::Size grid_size = grid.size();
	const float number_of_lines = 100.0f;
	cv::Size text_size;
	std::string txt;

	// If we have the previous chart.png file, we should import the grid from that file.
	if (std::filesystem::exists(filename))
	{
		cv::Mat tmp = cv::imread(filename);
		if (tmp.size() == dimensions)
		{
			// the image was read, and it is the correct size!

			// convert the previous image to greyscale, and then back to 3-channel BGR
			cv::cvtColor(tmp, tmp, cv::COLOR_BGR2GRAY);
			cv::cvtColor(tmp, tmp, cv::COLOR_GRAY2BGR);

			// lighten (fade) the old grid information
			cv::Mat lighter;
			const double percent = 0.25 * 255.0; // how much lighter to make the image (0.25 = 25%)
			tmp.convertTo(lighter, -1, 1, percent);

			lighter(grid_rect).copyTo(mat(grid_rect));
		}
	}

	// we're running out of space to put text in the chart!

	// the version number needs to be rotated 90 degrees and then inserted into the edge of the chart
	txt = "Darknet/YOLO v" DARKNET_VERSION_SHORT;
	const int border = 5;
	text_size = cv::Size(border * 2, border * 2) + cv::getTextSize(txt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1, nullptr);
	cv::Mat tmp(text_size, CV_8UC3, CV_RGB(255, 255, 255));
	cv::putText(tmp, txt, cv::Point(border, tmp.rows - border), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, CV_RGB(128, 128, 128), 1, cv::LINE_AA);
	cv::rotate(tmp, tmp, cv::ROTATE_90_COUNTERCLOCKWISE);
	tmp.copyTo(mat(cv::Rect(0, grid_rect.height - tmp.rows, tmp.cols, tmp.rows)));

	// draw the chart lines and axis values

	// vertical lines starting on the left with zero and working our way to the right side of the grid
	for (int i = 0; i <= number_of_lines; i++)
	{
		const cv::Point p1(std::round(i * grid_size.width / number_of_lines), 0);
		const cv::Point p2(p1.x, grid_size.height);

		if (i % 10)
		{
			cv::line(grid, p1, p2, CV_RGB(224, 224, 224));
		}
		else
		{
			cv::line(grid, p1, p2, CV_RGB(128, 128, 128));

			// the text is on the main image, not on the grid
			txt = std::to_string(static_cast<int>(std::round(i * max_batches / number_of_lines)));
			text_size = cv::getTextSize(txt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, 1, nullptr);
			cv::Point p3 = p2;
			p3.x += grid_offset_in_pixels - text_size.width / 2.0f;
			p3.y += 15;
			cv::putText(mat, txt, p3, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, cv::LINE_AA);
		}
	}

	// horizontal lines starting at the top and working our way down to zero
	for (int i = 0; i <= number_of_lines; i++)
	{
		const cv::Point p1(0, i * grid_size.height / number_of_lines);
		const cv::Point p2(grid_size.width, p1.y);

		if (i % 10)
		{
			cv::line(grid, p1, p2, CV_RGB(224, 224, 224));
		}
		else
		{
			cv::line(grid, p1, p2, CV_RGB(128, 128, 128)); // every 10th line is slightly darker

			// blue "loss" text on the LEFT axis
			cv::Point p3(30, p1.y + 3);
			if (p3.y < 12)
			{
				p3.y = 12;
			}
			std::stringstream ss;
			ss << std::fixed << std::setprecision(1) << max_chart_loss * (number_of_lines - i) / number_of_lines;
			cv::putText(mat, ss.str(), p3, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 200), 1, cv::LINE_AA);

			// red "mAP%" text on the RIGHT axis
			p3.x = grid_offset_in_pixels + grid_size.width + 2;
			txt = std::to_string(static_cast<int>(std::round(100 * (number_of_lines - i) / number_of_lines))) + "%";
			cv::putText(mat, txt, p3, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, cv::LINE_AA);
		}
	}

	// black rectangle around 3 sides of the grid (not on the top)
	grid_rect.y = -1;
	cv::rectangle(mat, grid_rect, CV_RGB(0, 0, 0));
	grid_rect.y = 0;

	// LEFT axis
	cv::putText(mat, "Loss", cv::Point(10, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 200), 1, cv::LINE_AA);

	// RIGHT axis
	cv::putText(mat, "mAP%", cv::Point(grid_offset_in_pixels + grid_size.width + 5, 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, cv::LINE_AA);

	txt = "Iteration";
	text_size = cv::getTextSize(txt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, 1, nullptr);
	cv::putText(mat, txt, cv::Point((mat.cols - text_size.width) / 2, mat.rows - 25), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 0), 1, cv::LINE_AA);

	// filename (or title?) in the bottom center
	txt = title;
	text_size = cv::getTextSize(txt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1, nullptr);
	cv::putText(mat, txt, cv::Point((mat.cols - text_size.width) / 2, mat.rows - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, CV_RGB(128, 128, 128), 1, cv::LINE_AA);

	return *this;
}


Chart & Chart::save_to_disk()
{
	TAT(TATPARMS);

	cv::imwrite(filename, mat, {cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION, 3});

	return *this;
}


Chart & Chart::update_loss(const int current_iteration, const float loss)
{
	/// @note This is called at @em every iteration to update the chart.

	TAT(TATPARMS);

	cv::Mat grid = mat(grid_rect);
	const cv::Size grid_size = grid.size();

	// draw the blue "loss" point
	cv::Point p;
	p.x = std::max(1.0f, std::round(grid_size.width * current_iteration / max_batches));
	p.y = std::max(0.0f, std::round(grid_size.height * (1.0f - loss / max_chart_loss)));
	cv::circle(grid, p, 1, CV_RGB(0, 0, 255), 1, cv::LINE_AA);

	previous_loss_iteration = current_iteration;
	previous_loss_value = loss;

	return *this;
}


Chart & Chart::update_accuracy(const float accuracy)
{
	// this is called prior to update_loss() where the iteration is recorded, so we must add 1
	const int iteration_guess = 1 + static_cast<int>(std::max(previous_loss_iteration, previous_map_iteration));

	return update_accuracy(iteration_guess, accuracy);
}


Chart & Chart::update_accuracy(const int current_iteration, const float accuracy)
{
	/// @note This is called only when a new mAP% value has been calculated.

	TAT(TATPARMS);

	cv::Mat grid = mat(grid_rect);
	const cv::Size grid_size = grid.size();

	// draw the red mAP% line
	if (current_iteration > 0.0f && accuracy >= 0.0f)
	{
		if (previous_map_iteration <= 0.0f && previous_map_value <= 0.0f)
		{
			// this is the very first mAP% entry, so "inherit" the current mAP% values
			previous_map_iteration = current_iteration;
			previous_map_value = accuracy;
		}

		cv::Point p1(std::round(grid_size.width * previous_map_iteration / max_batches), std::round(grid_size.height * (1.0f - previous_map_value)));
		cv::Point p2(std::max(1.0f, std::round(grid_size.width * current_iteration / max_batches)), std::round(grid_size.height * (1.0f - accuracy)));
		cv::line(grid, p1, p2, map_colour, 2, cv::LINE_AA);
		cv::circle(grid, p2, 2, map_colour, 2, cv::LINE_AA);

		// decide if the mAP% value has changed enough that we need to re-label it on the chart
		if ((std::fabs(previous_map_value - accuracy) > 0.1) ||
			(accuracy > max_map_value) ||
			(current_iteration - previous_map_shown) >= max_batches / 10.0f)
		{
			previous_map_shown = current_iteration;
			max_map_value = std::max(max_map_value, accuracy);

			const std::string txt = std::to_string(static_cast<int>(std::round(100.0f * accuracy))) + "%";
			p2.x -= 30;
			p2.y += 15;
			cv::putText(grid, txt, p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(255, 255, 255), 5, cv::LINE_AA);
			cv::putText(grid, txt, p2, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, map_colour, 1, cv::LINE_AA);
		}
	}

	previous_map_iteration = current_iteration;
	previous_map_value = accuracy;

	return *this;
}


Chart & Chart::update_bottom_text(const float seconds_remaining)
{
	// draw the text at the bottom of the chart

	TAT(TATPARMS);

	const cv::Size grid_size = grid_rect.size();
	cv::Size text_size;

	// blue LOSS=...
	std::stringstream ss;
	ss << "loss=";
	if (previous_loss_value < 0.0f)
	{
		ss << "unknown";
	}
	else
	{
		ss << std::fixed << std::setprecision(4) << previous_loss_value;
	}
	cv::Point p1(15, grid_size.height + 20);
	cv::Point p2(p1.x + 150, p1.y + 20);
	cv::rectangle(mat, p1, p2, CV_RGB(255, 255, 255), cv::FILLED); // clear out previous text
	p1.y += 15;
	cv::putText(mat, ss.str(), p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(0, 0, 200), 1, cv::LINE_AA);

	// red MAP%=...
	ss = std::stringstream();
	ss << "mAP%=";
	if (previous_map_value < 0.0f)
	{
		ss << "unknown";
	}
	else
	{
		ss << std::fixed << std::setprecision(4) << previous_map_value;

		if (max_map_value > 0.0f)
		{
			ss << ", best=" << max_map_value;
		}
	}
	p1 = cv::Point(150, grid_size.height + 20);
	p2 = cv::Point(p1.x + 250, p1.y + 20);
	cv::rectangle(mat, p1, p2, CV_RGB(255, 255, 255), cv::FILLED); // clear out previous text
	p1.y += 15;
	cv::putText(mat, ss.str(), p1, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.7, CV_RGB(200, 0, 0), 1, cv::LINE_AA);

	// grey TIME REMAINING=...
	const std::time_t current_time = std::time(nullptr);
	std::string txt = "time remaining=" + Darknet::format_time_remaining(seconds_remaining);
	if (seconds_remaining <= 5.0f)
	{
		// instead of the time remaining, show the amount of time that has elapsed
		txt = "time elapsed=" + Darknet::format_time_remaining(current_time - started_timestamp);
	}
	p1 = cv::Point(mat.cols - 250, grid_size.height + 20);
	p2 = cv::Point(mat.cols, p1.y + 20);
	cv::rectangle(mat, p1, p2, CV_RGB(255, 255, 255), cv::FILLED); // clear out previous text
	text_size = cv::getTextSize(txt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1, nullptr);
	cv::putText(mat, txt, cv::Point(mat.cols - text_size.width - 5, p1.y + 15), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, CV_RGB(128, 128, 128), 1, cv::LINE_AA);

	// grey ITERATION #...
	const int current_iteration = static_cast<int>(std::max(previous_loss_iteration, previous_map_iteration));
	txt = "iteration #"
		+ std::to_string(current_iteration)
		+ " of "
		+ std::to_string(static_cast<int>(max_batches))
		+ " ("
		+ std::to_string(static_cast<int>(std::round(100.0f * current_iteration / max_batches)))
		+ "%)";
	p1 = cv::Point(mat.cols - 250, mat.rows - 20);
	p2 = cv::Point(mat.cols, p1.y + 20);
	cv::rectangle(mat, p1, p2, CV_RGB(255, 255, 255), cv::FILLED); // clear out previous text
	text_size = cv::getTextSize(txt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, 1, nullptr);
	cv::putText(mat, txt, cv::Point(mat.cols - text_size.width - 5, mat.rows - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, CV_RGB(128, 128, 128), 1, cv::LINE_AA);

	// draw the timestamp in the lower left
	std::tm * tm = std::localtime(&current_time);
	char buffer[100];
	std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %z", tm);
	p1 = cv::Point(15, mat.rows - 20);
	p2 = cv::Point(p1.x + 210, p1.y + 20);
	cv::rectangle(mat, p1, p2, CV_RGB(255, 255, 255), cv::FILLED); // fill it with white to clear out the previous text
	cv::putText(mat, buffer, cv::Point(15, mat.rows - 10), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.6, CV_RGB(128, 128, 128), 1, cv::LINE_AA);

	return *this;
}


Chart & Chart::update_save_and_display(const int current_iteration, const float loss, const float seconds_remaining, const bool dont_show)
{
	TAT(TATPARMS);

	update_loss(current_iteration, loss);

	bool need_to_update = false;
	bool need_to_save = false;

	const std::time_t current_time = std::time(nullptr);
	if (current_time >= last_update_timestamp + 15)
	{
		need_to_update = true;
	}

	if ((current_iteration == 5) or		// update soon after training has started so the user has an idea of how long it will take
		(current_iteration % 100) == 0)	// update every 100th iteration
	{
		need_to_update = true;
		need_to_save = true;
	}

	if (need_to_update)
	{
		update_bottom_text(seconds_remaining);
		last_update_timestamp = current_time;
	}

	if (need_to_save)
	{
		save_to_disk();
		last_save_timestamp = current_time;
	}

	if (need_to_update && dont_show == false)
	{
		cv::imshow(title, mat);
		cv::waitKey(10);
	}

	return *this;
}
