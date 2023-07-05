#include "darknet_utils.hpp"
#include "Chart.hpp"
#include "image.h"

#include <iomanip>
#include <sstream>
#include <string>
#include <cstdio>
#include <cmath>


std::vector<std::string> class_names;

#ifdef OPENCV
std::vector<cv::Scalar> class_colours;
#endif


/** @todo Colour not yet supported on Windows.  On recent editions of Windows see @p SetConsoleMode() and
 * @p ENABLE_VIRTUAL_TERMINAL_PROCESSING, though I think this is only with newer versions of Windows 10.
 * Looks like this can also be enabled globally in the registry.  In @p [HKEY_CURRENT_USER/Console]
 * create a key called @p "VirtualTerminalLevel" and set it to @p 1.  Open a new console for the change
 * to take effect.
 */
#if defined(_MSC_VER) || defined(WIN32)
bool colour_is_enabled = false;
#else
bool colour_is_enabled = true;
#endif


const char * const ansi_colours[kMax] =
{
	"\033[0m",		// EColour::kNormal
	"\033[0;30m",	// EColour::kBlack
	"\033[0;31m",	// EColour::kRed
	"\033[0;32m",	// EColour::kGreen
	"\033[0;33m",	// EColour::kBrown
	"\033[0;34m",	// EColour::kBlue
	"\033[0;35m",	// EColour::kMagenta
	"\033[0;36m",	// EColour::kCyan
	"\033[0;37m",	// EColour::kLightGrey
	"\033[1;30m",	// EColour::kDarkGrey
	"\033[1;31m",	// EColour::kBrightRed
	"\033[1;32m",	// EColour::kBrightGreen
	"\033[1;33m",	// EColour::kYellow
	"\033[1;34m",	// EColour::kBrightBlue
	"\033[1;35m",	// EColour::kBrightMagenta
	"\033[1;36m",	// EColour::kBrightCyan
	"\033[1;37m"	// EColour::kBrightWhite
};


std::string in_colour(const EColour colour, const int i)
{
	return in_colour(colour, std::to_string(i));
}


std::string in_colour(const EColour colour, const float f)
{
	return in_colour(colour, std::to_string(f));
}


std::string in_colour(const EColour colour, const double d)
{
	return in_colour(colour, std::to_string(d));
}


std::string in_colour(const EColour colour, const std::string & msg)
{
	if (colour_is_enabled)
	{
		return ansi_colours[colour] + msg + ansi_colours[EColour::kNormal];
	}

	return msg;
}


void remember_class_names(char ** names, const int count)
{
	if (static_cast<size_t>(count) == class_names.size())
	{
		// assume this is a redundant call and we already know all of the class names and colours
		return;
	}

	printf("\nRemembering %d class%s:\n", count, (count == 1 ? "" : "es"));

	class_names.clear();
	class_names.reserve(count);

	#ifdef OPENCV
	class_colours.clear();
	class_colours.reserve(count);
	#endif

	for (int idx = 0; idx < count; idx ++)
	{
		const std::string name = names[idx];
		if (name.find_first_not_of(" \t\r\n") == std::string::npos)
		{
			display_error_msg("The .names file appears to contain a blank line.\n");
		}

		class_names.push_back(name);

		#ifdef OPENCV
		const int offset = idx * 123457 % count;
		const int r = std::min(255.0f, std::round(256.0f * get_color(2, offset, count)));
		const int g = std::min(255.0f, std::round(256.0f * get_color(1, offset, count)));
		const int b = std::min(255.0f, std::round(256.0f * get_color(0, offset, count)));

		class_colours.push_back(CV_RGB(r, g, b));

		printf("-> class #%d (%s) will use colour #%02X%02X%02X\n", idx, names[idx], r, g, b);
		#endif
	}

	printf("\n");

	return;
}


std::string format_time(const double & t)
{
	std::stringstream ss;
	ss << std::fixed << std::setprecision(3);

	if (t >= 0.5)
	{
		ss << t << " seconds";
	}
	else if (t >= 0.005)
	{
		ss << (t * 1000.0) << " milliseconds";
	}
	else
	{
		ss << (t * 1000000.0) << " microseconds";
	}

	return ss.str();
}


std::string format_time_remaining(const double & t)
{
	const double seconds	= 1.0;
	const double minutes	= 60.0 * seconds;
	const double hours		= 60.0 * minutes;
	const double days		= 24.0 * hours;
	const double weeks		= 7.0 * days;

	std::stringstream ss;
	ss << std::fixed << std::setprecision(1);

	if (t < 0)
	{
		ss << "unknown";
	}
	else if (t >= 2 * weeks)
	{
		ss << (t / weeks) << " weeks";
	}
	else if (t >= 1.5 * days)
	{
		ss << (t / days) << " days";
	}
	else if (t >= 1.5 * hours)
	{
		ss << (t / hours) << " hours";
	}
	else if (t >= 1.5 * minutes)
	{
		ss << (t / minutes) << " minutes";
	}
	else
	{
		const int secs = static_cast<int>(round(t));
		ss << secs << " second" << (secs == 1 ? "" : "s");
	}

	return ss.str();
}


std::string format_loss(const double & loss)
{
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


std::string format_map_accuracy(const float & accuracy)
{
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


void display_loaded_images(const int images, const double time)
{
	printf("loaded %d images in %s\n", images, format_time(time).c_str());

	return;
}


void display_iteration_summary(const int iteration, const float loss, const float avg_loss, const float rate, const double time, const int images, const double avg_time)
{
	printf("%s: loss=%s, avg loss=%s, rate=%f, %s, %d images, time remaining=%s\n",
			in_colour(EColour::kBrightWhite, iteration)	.c_str(),
			format_loss(loss)							.c_str(),
			format_loss(avg_loss)						.c_str(),
			rate,
			format_time(time)							.c_str(),
			images,
			format_time_remaining(avg_time)				.c_str()
			);

	return;
}


void display_last_accuracy(const float iou_thresh, const float mean_average_precision, const float best_map)
{
	printf("-> last accuracy mAP@%0.2f=%s, best=%s\n",
			iou_thresh,
			format_map_accuracy(mean_average_precision).c_str(),
			format_map_accuracy(best_map).c_str());

	return;
}


void display_error_msg(const char * const msg)
{
	if (msg != nullptr)
	{
		printf("%s", in_colour(EColour::kBrightRed, msg).c_str());
	}

	return;
}


void display_warning_msg(const char * const msg)
{
	if (msg != nullptr)
	{
		printf("%s", in_colour(EColour::kYellow, msg).c_str());
	}

	return;
}


void update_console_title(const int iteration, const int max_batches, const float loss, const float current_map, const float best_map, const double seconds_remaining)
{
	// doing this requires some ANSI/VT100 escape codes, so only do this if colour is also enabled
	if (colour_is_enabled)
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


bool file_exists(const char * const filename)
{
	bool result = false;

	if (filename != nullptr)
	{
		FILE * file = std::fopen(filename, "rb");
		if (file)
		{
			std::fclose(file);
			result = true;
		}
	}

	return result;
}


std::string text_to_simple_label(std::string txt)
{
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


void initialize_new_charts(const int max_batches, const float max_img_loss)
{
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


void update_loss_in_new_charts(const int current_iteration, const float loss, const double hours_remaining, const bool dont_show)
{
	training_chart.update_save_and_display(current_iteration, loss, hours_remaining, dont_show);

	for (auto & chart : more_charts)
	{
		chart.update_save_and_display(current_iteration, loss, hours_remaining, true);
	}

	return;
}


void update_accuracy_in_new_charts(const int class_index, const float accuracy)
{
	if (class_index < 0)
	{
		training_chart.update_accuracy(accuracy);
	}
	else if (static_cast<size_t>(class_index) < more_charts.size())
	{
		more_charts[class_index].update_accuracy(accuracy);
	}

	return;
}
