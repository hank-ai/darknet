#include "darknet_format_and_colour.hpp"


namespace
{
	/// Text strings with the VT100/ANSI escape codes needed to display colour output.
	static const Darknet::VStr ansi_colours =
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

	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


std::string Darknet::in_colour(const Darknet::EColour colour, const int i)
{
	TAT(TATPARMS);

	return in_colour(colour, std::to_string(i));
}


std::string Darknet::in_colour(const EColour colour, const float f)
{
	TAT(TATPARMS);

	return in_colour(colour, std::to_string(f));
}


std::string Darknet::in_colour(const EColour colour, const double d)
{
	TAT(TATPARMS);

	return in_colour(colour, std::to_string(d));
}


std::string Darknet::in_colour(const EColour colour, const std::string & msg)
{
	TAT(TATPARMS);

	if (cfg_and_state.colour_is_enabled)
	{
		return ansi_colours[colour] + msg + ansi_colours[EColour::kNormal];
	}

	return msg;
}


std::string Darknet::in_colour(const EColour colour)
{
	TAT(TATPARMS);

	if (cfg_and_state.colour_is_enabled)
	{
		return ansi_colours[colour];
	}

	return "";
}


std::string Darknet::format_in_colour(const std::string & str, const EColour & colour, const size_t & len)
{
	TAT(TATPARMS);

	std::string padding;
	if (str.length() < len)
	{
		padding = std::string(len - str.length(), ' ');
	}

	return in_colour(colour, str) + padding;
}


std::string Darknet::format_in_colour(const int & i, const EColour & colour, const size_t & len)
{
	TAT(TATPARMS);

	std::string str = std::to_string(i);
	std::string padding;
	if (str.length() < len)
	{
		padding = std::string(len - str.length(), ' ');
	}

	return padding + in_colour(colour, str);
}


std::string Darknet::format_in_colour(const float & f, const EColour & colour, const size_t & len)
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss << std::fixed << std::setprecision(4);
	ss << (std::isnormal(f) ? f : 0.0f);

	const std::string str = ss.str();
	std::string padding;
	if (str.length() < len)
	{
		padding = std::string(len - str.length(), ' ');
	}

	return padding + in_colour(colour, str);
}


std::string Darknet::format_in_colour(const float & f, const size_t & len, const bool inverted)
{
	TAT(TATPARMS);

	// Are we dealing with 0...1, or 0...100?  If the value is > 1.0,
	// then assume we have a number between 0...100 and divide by 100
	// to scale it back to 0...1.
	float scale = f;
	if (scale > 1.0f)
	{
		scale /= 100.0f;
	}

	if (inverted and scale >= 0.0f and scale <= 1.0f)
	{
		// if we're showing an error rate for example, numbers near zero are good and numbers near 1 are bad,
		// so we invert the scale to ensure the colours maintain the same meaning (green=good, red=bad)
		scale = 1.0f - scale;
	}

	EColour colour = EColour::kNormal;
	if (scale > 1.0f)			colour = EColour::kNormal;
	else if (scale >= 0.85f)	colour = EColour::kBrightGreen;
	else if (scale >= 0.70f)	colour = EColour::kBrightCyan;
	else if (scale >= 0.55f)	colour = EColour::kBrightBlue;
	else if (scale >= 0.40f)	colour = EColour::kBrightMagenta;
	else						colour = EColour::kBrightRed;

	return format_in_colour(f, colour, len);
}


std::string Darknet::format_map_confusion_matrix_values(
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
	const float & false_pos_rate)
{
	TAT(TATPARMS);

	if (name.length() > 16)
	{
		name.erase(15);
		name += "+";
	}

	const std::string output =
		"  " +
		format_in_colour(class_id, EColour::kNormal	, 2	) + " " +
		format_in_colour(name, EColour::kBrightWhite, 16) + " " +
		format_in_colour(100.0f * average_precision	, 12) + " " +
		format_in_colour(tp, EColour::kNormal		, 6	) + " " +
		format_in_colour(fn, EColour::kNormal		, 6	) + " " +
		format_in_colour(fp, EColour::kNormal		, 6	) + " " +
		format_in_colour(tn, EColour::kNormal		, 6	) + " " +
		format_in_colour(accuracy					, 8	) + " " +
		format_in_colour(error_rate					, 9, true) + " " +
		format_in_colour(precision					, 9	) + " " +
		format_in_colour(recall						, 6	) + " " +
		format_in_colour(specificity				, 11) + " " +
		format_in_colour(false_pos_rate				, 12, true);

	return output;
}
