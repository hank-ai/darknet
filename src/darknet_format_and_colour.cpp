#include "darknet_format_and_colour.hpp"


namespace
{
	/// Text strings with the VT100/ANSI escape codes needed to display colour output.
	const Darknet::VStr ansi_colours =
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
}


std::string Darknet::in_colour(const Darknet::EColour colour, const int i)
{
	return in_colour(colour, std::to_string(i));
}


std::string Darknet::in_colour(const EColour colour, const float f)
{
	return in_colour(colour, std::to_string(f));
}


std::string Darknet::in_colour(const EColour colour, const double d)
{
	return in_colour(colour, std::to_string(d));
}


std::string Darknet::in_colour(const EColour colour, const std::string & msg)
{
	if (cfg_and_state.colour_is_enabled)
	{
		return ansi_colours[colour] + msg + ansi_colours[EColour::kNormal];
	}

	return msg;
}


std::string Darknet::in_colour(const EColour colour)
{
	if (cfg_and_state.colour_is_enabled)
	{
		return ansi_colours[colour];
	}

	return "";
}
