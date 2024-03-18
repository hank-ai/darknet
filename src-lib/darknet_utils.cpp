#include "Chart.hpp"
#include "darknet_internal.hpp"


std::vector<std::string> Darknet::class_names;
std::vector<cv::Scalar> Darknet::class_colours;


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


void Darknet::remember_class_names(char ** names, const int count)
{
	TAT(TATPARMS);

	if (static_cast<size_t>(count) == class_names.size())
	{
		// assume this is a redundant call and we already know all of the class names and colours
		return;
	}

	printf("\nRemembering %d class%s:\n", count, (count == 1 ? "" : "es"));

	class_names.clear();
	class_names.reserve(count);

	class_colours.clear();
	class_colours.reserve(count);

	for (int idx = 0; idx < count; idx ++)
	{
		const std::string name = names[idx];
		if (name.find_first_not_of(" \t\r\n") == std::string::npos)
		{
			display_error_msg("The .names file appears to contain a blank line.\n");
		}

		class_names.push_back(name);

		const int offset = idx * 123457 % count;
		const int r = std::min(255.0f, std::round(256.0f * get_color(2, offset, count)));
		const int g = std::min(255.0f, std::round(256.0f * get_color(1, offset, count)));
		const int b = std::min(255.0f, std::round(256.0f * get_color(0, offset, count)));

		class_colours.push_back(CV_RGB(r, g, b));

		printf("-> class #%d (%s) will use colour #%02X%02X%02X\n", idx, names[idx], r, g, b);
	}

	printf("\n");

	return;
}


std::string Darknet::format_time(const double & seconds_remaing)
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss << std::fixed << std::setprecision(3);

	if (seconds_remaing >= 0.5)
	{
		ss << seconds_remaing << " seconds";
	}
	else if (seconds_remaing >= 0.005)
	{
		ss << (seconds_remaing * 1000.0) << " milliseconds";
	}
	else
	{
		ss << (seconds_remaing * 1000000.0) << " microseconds";
	}

	return ss.str();
}


std::string Darknet::format_time_remaining(const float & seconds_remaining)
{
	TAT(TATPARMS);

	const float seconds	= 1.0;
	const float minutes	= 60.0 * seconds;
	const float hours	= 60.0 * minutes;
	const float days	= 24.0 * hours;
	const float weeks	= 7.0 * days;

	std::stringstream ss;
	ss << std::fixed << std::setprecision(0);

	if (seconds_remaining < 0.0f or seconds_remaining >= 4.0f * weeks)
	{
		ss << "unknown";
	}
	else if (seconds_remaining >= 2.0f * weeks)
	{
		ss << (seconds_remaining / weeks) << " weeks";
	}
	else if (seconds_remaining >= 2.0f * days)
	{
		ss << (seconds_remaining / days) << " days";
	}
	else if (seconds_remaining >= 2.0f * hours)
	{
		ss << (seconds_remaining / hours) << " hours";
	}
	else if (seconds_remaining >= 2.0f * minutes)
	{
		ss << (seconds_remaining / minutes) << " minutes";
	}
	else
	{
		const int secs = std::round(seconds_remaining);
		ss << secs << " second";
		if (secs != 1)
		{
			ss << "s";
		}
	}

	return ss.str();
}


std::string Darknet::format_loss(const double & loss)
{
	TAT(TATPARMS);

	EColour colour = EColour::kNormal;

	if (loss < 0.0		||
		loss >= 1000.0	||
		std::isfinite(loss) == false)
	{
		colour = EColour::kBrightRed;
	}
	else if (loss >= 20)
	{
		colour = EColour::kRed;
	}
	else if (loss >= 4)
	{
		colour = EColour::kCyan;
	}
	else
	{
		// else loss is somewhere between 0 and 4
		colour = EColour::kBrightCyan;
	}

	std::stringstream ss;
	ss << std::fixed << std::setprecision(3) << loss;

	return in_colour(colour, ss.str());
}


std::string Darknet::format_map_accuracy(const float & accuracy)
{
	TAT(TATPARMS);

	EColour colour = EColour::kNormal;

	if (accuracy < 0.5f || std::isfinite(accuracy) == false)
	{
		colour = EColour::kBrightRed;
	}
	else if (accuracy < 0.6f)
	{
		colour = EColour::kRed;
	}
	else if (accuracy < 0.7f)
	{
		colour = EColour::kBlue;
	}
	else
	{
		// else accuracy is >= 70%
		colour = EColour::kBrightBlue;
	}

	std::stringstream ss;
	ss << std::fixed << std::setprecision(2) << (100.0f * accuracy) << "%";

	return in_colour(colour, ss.str());
}


void Darknet::display_loaded_images(const int images, const double time)
{
	TAT(TATPARMS);

	printf("loaded %d images in %s\n", images, format_time(time).c_str());

	return;
}


void Darknet::display_iteration_summary(const int iteration, const float loss, const float avg_loss, const float rate, const double time, const int images, const float seconds_remaining)
{
	TAT(TATPARMS);

	printf("%s: loss=%s, avg loss=%s, rate=%f, %s, %d images, time remaining=%s\n",
			in_colour(EColour::kBrightWhite, iteration)	.c_str(),
			format_loss(loss)							.c_str(),
			format_loss(avg_loss)						.c_str(),
			rate,
			format_time(time)							.c_str(),
			images,
			format_time_remaining(seconds_remaining)	.c_str()
			);

	return;
}


void Darknet::display_last_accuracy(const float iou_thresh, const float mean_average_precision, const float best_map)
{
	TAT(TATPARMS);

	printf("-> last accuracy mAP@%0.2f=%s, best=%s\n",
			iou_thresh,
			format_map_accuracy(mean_average_precision).c_str(),
			format_map_accuracy(best_map).c_str());

	return;
}


void Darknet::display_error_msg(const std::string & msg)
{
	TAT(TATPARMS);

	if (not msg.empty())
	{
		std::cout << in_colour(EColour::kBrightRed, msg);
	}

	return;
}


void Darknet::display_warning_msg(const std::string & msg)
{
	TAT(TATPARMS);

	if (not msg.empty())
	{
		std::cout << in_colour(EColour::kYellow, msg);
	}

	return;
}


void Darknet::update_console_title(const int iteration, const int max_batches, const float loss, const float current_map, const float best_map, const float seconds_remaining)
{
	TAT(TATPARMS);

	// doing this requires some ANSI/VT100 escape codes, so only do this if colour is also enabled
	if (cfg_and_state.colour_is_enabled)
	{
		if (std::isfinite(current_map) && current_map > 0.0f)
		{
			printf("\033]2;%d/%d: loss=%0.1f map=%0.2f best=%0.2f time=%s\007", iteration, max_batches, loss, current_map, best_map, format_time_remaining(seconds_remaining).c_str());
		}
		else
		{
			printf("\033]2;%d/%d: loss=%0.1f time=%s\007", iteration, max_batches, loss, format_time_remaining(seconds_remaining).c_str());
		}
	}

	return;
}


std::string Darknet::text_to_simple_label(std::string txt)
{
	TAT(TATPARMS);

	// first we convert unknown characters to whitespace
	size_t pos = 0;
	while (true)
	{
		pos = txt.find_first_not_of(
			" "
			"0123456789"
			"abcdefghijklmnopqrstuvwxyz"
			"ABCDEFGHIJKLMNOPQRSTUVWXYZ", pos);
		if (pos == std::string::npos)
		{
			break;
		}
		txt[pos] = ' ';
	}

	// then we merge consecutive spaces together
	pos = 0;
	while (true)
	{
		pos = txt.find("  ", pos);
		if (pos == std::string::npos)
		{
			break;
		}
		txt.erase(pos, 1);
	}

	// finally we convert any remaining whitespace to underscore so it can be used to create filenames without spaces
	pos = 0;
	while (true)
	{
		pos = txt.find(' ', pos);
		if (pos == std::string::npos)
		{
			break;
		}
		txt[pos] = '_';
	}

	return txt;
}


void Darknet::initialize_new_charts(const int max_batches, const float max_img_loss)
{
	TAT(TATPARMS);

	training_chart = Chart("", max_batches, max_img_loss);

	more_charts.clear();

	for (size_t idx = 0; idx < class_names.size(); idx ++)
	{
		Chart chart(class_names[idx], max_batches, max_img_loss);
		chart.map_colour = class_colours[idx];

		more_charts.push_back(chart);
	}

	return;
}


void Darknet::update_loss_in_new_charts(const int current_iteration, const float loss, const float seconds_remaining, const bool dont_show)
{
	TAT(TATPARMS);

	if (training_chart.empty() == false)
	{
		training_chart.update_save_and_display(current_iteration, loss, seconds_remaining, dont_show);

		for (auto & chart : more_charts)
		{
			chart.update_save_and_display(current_iteration, loss, seconds_remaining, true);
		}
	}

	return;
}


void Darknet::update_accuracy_in_new_charts(const int class_index, const float accuracy)
{
	TAT(TATPARMS);

	if (training_chart.empty() == false)
	{
		if (class_index < 0)
		{
			training_chart.update_accuracy(accuracy);
		}
		else if (static_cast<size_t>(class_index) < more_charts.size())
		{
			more_charts[class_index].update_accuracy(accuracy);
		}
	}

	return;
}
