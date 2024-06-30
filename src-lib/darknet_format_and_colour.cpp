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


std::string Darknet::format_in_colour(const std::string & str, const EColour & colour, const int & len)
{
	TAT(TATPARMS);

	// The text string will be left-aligned.  If the length is negative, then it will be right-aligned.
	const int l = (len < 0 ? -len : len);

	std::string padding;
	if (str.length() < l)
	{
		padding = std::string(l - str.length(), ' ');
	}

	if (len < 0)
	{
		return padding + in_colour(colour, str);
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


std::string Darknet::format_in_colour(const size_t & st, const EColour & colour, const size_t & len)
{
	TAT(TATPARMS);

	std::string str = std::to_string(st);
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


std::string Darknet::format_layer_summary(const size_t idx, const Darknet::CfgSection & section, const layer & l)
{
	TAT(TATPARMS);

	const std::string name = [&]() -> std::string
	{
		if (l.type == LAYER_TYPE::UPSAMPLE and l.reverse)
		{
			return "downsample";
		}
		if (l.type == LAYER_TYPE::MAXPOOL and l.avgpool)
		{
			return "avg";
		}
		if (l.type == LAYER_TYPE::MAXPOOL and l.maxpool_depth)
		{
			return "max-depth";
		}
		return section.name;
	}();
	const std::string filters_and_groups = [&]()
	{
		std::string str;
		if (l.n != 0)
		{
			str = std::to_string(l.n);

			if (l.type == LAYER_TYPE::ROUTE)
			{
				str = std::to_string(l.groups);
			}
			else if (l.groups > 1)
			{
				str += " / " + std::to_string(l.groups);
			}
		}
		return str;
	}();
	const std::string size = [&]()
	{
		std::stringstream ss;
		if (l.type == LAYER_TYPE::UPSAMPLE)
		{
			ss << l.stride << "X";
		}
		else if (l.type == LAYER_TYPE::ROUTE or l.type == LAYER_TYPE::SHORTCUT)
		{
			for (size_t i = 0; i < l.n; i++)
			{
				ss << (i == 0 ? "#" : ", ") << l.input_layers[i];
			}
		}
		else if (l.type == LAYER_TYPE::YOLO)
		{
			ss << l.total << " anchors";
		}
		else
		{
			if (l.size > 0)
			{
				ss << l.size << " x " << l.size << " / ";
				if (l.stride_x != l.stride_y)
				{
					ss << l.stride_x << " x " << l.stride_y;

					if (l.dilation > 1)
					{
						ss << " / " << l.dilation;
					}
				}
				else
				{
					ss << l.stride;
				}
			}
			else
			{
				ss << "  x";
			}
		}
		return ss.str();
	}();
	const std::string input_size = [&]()
	{
		std::stringstream ss;
		if (l.type == LAYER_TYPE::ROUTE)
		{
			if (l.groups > 1)
			{
				ss << l.group_id << "/" << l.groups;
			}
		}
		else if (l.type == LAYER_TYPE::SHORTCUT)
		{
			if (l.weights_normalization != WEIGHTS_NORMALIZATION_T::NO_NORMALIZATION)
			{
				ss << "n=" << Darknet::to_string(static_cast<Darknet::EWeightsNormalization>(l.weights_normalization));
			}
			else if (l.weights_type != WEIGHTS_TYPE_T::NO_WEIGHTS)
			{
				ss << "t=" << Darknet::to_string(static_cast<Darknet::EWeightsType>(l.weights_type));
			}
		}
		else
		{
			ss << l.w << " x " << l.h << " x " << l.c;
		}
		return ss.str();
	}();
	const std::string output_size = [&]()
	{
		std::stringstream ss;
		ss << l.out_w << " x " << l.out_h << " x " << l.out_c;
		return ss.str();
	}();
	const std::string bflops = [&]()
	{
		std::stringstream ss;
		if (l.bflops > 0.0f)
		{
			ss << std::fixed << std::setprecision(3) << l.bflops << " BF";
		}
		return ss.str();
	}();

	const std::string output =
		format_in_colour(idx					, EColour::kNormal			, 3	) + " " +
		format_in_colour(section.line_number	, EColour::kBrightBlue		, 4	) + " " +
		format_in_colour(name					, EColour::kBrightWhite		, 15) + " " +
		format_in_colour(filters_and_groups		, EColour::kNormal			, 7	) + " " +
		format_in_colour(size					, EColour::kBrightGreen		, 10) + " " +
		format_in_colour(input_size				, EColour::kBrightCyan		, 15) +
		format_in_colour(" -> "					, EColour::kNormal			, 4	) +
		format_in_colour(output_size			, EColour::kBrightCyan		, 15) + " " +
		format_in_colour(bflops					, EColour::kBrightMagenta	, 8);

	return output;
}
