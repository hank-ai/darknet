#include "Cfg.hpp"


Darknet::CfgLine::CfgLine(const std::string & l, const size_t ln, const std::string & lhs, const std::string & rhs) :
	line_number(ln),
	line(l),
	key(lhs),
	val(rhs),
	used(false)
{
	TAT(TATPARMS);

	if (val.size() > 0 and val.find_first_not_of("-.0123456789") == std::string::npos)
	{
		// looks like this might be a number!
		try
		{
			f = std::stof(val);
		}
		catch (...)
		{
			darknet_fatal_error(DARKNET_LOC, "line #%d: failed to parse numeric value from \"%s\"", line_number, l.c_str());
		}
	}

//	std::cout << "#" << line_number << ": line=\"" << line << "\" key=" << key << " val=" << val << std::endl;

	return;
};


Darknet::CfgLine::~CfgLine()
{
	TAT(TATPARMS);

	return;
}


std::string Darknet::CfgLine::debug() const
{
	std::stringstream ss;
	ss << line_number << ": key=" << key << " val=" << val;

	if (f.has_value())
	{
		ss << " f=" << f.value();
	}

	return ss.str();
}


Darknet::CfgSection::CfgSection(const std::string & l, const size_t ln) :
	type(Darknet::get_layer_from_name(l)),
	line_number(ln)
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgSection::~CfgSection()
{
	TAT(TATPARMS);

	return;
}


std::string Darknet::CfgSection::debug() const
{
	std::stringstream ss;
	ss << line_number << ": [" << get_name_from_layer(type) << "]" << std::endl;

	for (const auto & line : lines)
	{
		ss << line.debug() << std::endl;
	}

	return ss.str();
}


Darknet::CfgFile::CfgFile() :
	total_lines(0)
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgFile::CfgFile(const std::filesystem::path & fn) :
	CfgFile()
{
	TAT(TATPARMS);

	parse(fn);

	return;
}


Darknet::CfgFile::~CfgFile()
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgFile & Darknet::CfgFile::parse(const std::filesystem::path & fn)
{
	TAT(TATPARMS);

	filename = fn;

	return parse();
}


#if 0
Darknet::CfgOptions foo(const Darknet::ELayerType type)
{
	switch (type)
	{
		case ELayerType::LOCAL:
		{
			CfgOptions options;
			options.
				add("filters"	, true, true, 1).	// int n = option_find_int(options, "filters",1);
				add("size"		, true, true, 1).	// int size = option_find_int(options, "size",1);
				add("stride"	, true, true, 1).	// int stride = option_find_int(options, "stride",1);
				add("pad"		, true, true, 0).	// int pad = option_find_int(options, "pad",0);
				add("activation", true, true, "logistic");	// char *activation_s = option_find_str(options, "activation", "logistic");
			//ACTIVATION activation = get_activation(activation_s);
			break;
		}
		case ELayerType::NETWORK:
		{
			CfgOptions options;
			options.
				add(
		}
	}

	return options;
}
#endif


Darknet::CfgFile & Darknet::CfgFile::parse()
{
	TAT(TATPARMS);

	total_lines = 0;
	sections.clear();

	if (not std::filesystem::exists(filename))
	{
		/// @throw std::invalid_argument if the given filename doesn't exist
		throw std::invalid_argument("config file does not exist: " + filename.string());
	}

	filename = std::filesystem::canonical(filename);

	/* find lines such as these:
	 *
	 *		[net]
	 *		width=224
	 *		height=160 # this is a test
	 *		channels = 3
	 *		momentum=0.9
	 */
	const std::regex rx(
		"^"				// start of line
		"\\["			// [
		"(.+)"			// group #1:  section name
		"\\]"			// ]
		"|"				// ...or...
		"^"				// start of line
		"[ \t]*"		// optional whitespace
		"("				// group #2
		"[^#= \t]+"		// key (everything up to #, =, or whitespace)
		")"				// end of group #2
		"[ \t]*"		// optional whitespace
		"="				// =
		"[ \t]*"		// optional whitespace
		"("				// group #3
		"[^#]+"			// value (everything up to #)
		")"				// end of group #3
		);

	std::ifstream ifs(filename);

	std::string line;
	while (std::getline(ifs, line))
	{
		total_lines ++;

		std::smatch matches;
		if (std::regex_match(line, matches, rx))
		{
			// line is either a section name, or a key-value pair -- check for a section name first

			const std::string section_name = matches.str(1);
			if (section_name.size() > 0)
			{
				CfgSection s(section_name, total_lines);
				sections.push_back(s);
			}
			else
			{
				// a section must exist prior to reading a key=val pair
				if (sections.empty())
				{
					/// @throw std::runtime_error if a key=val pair is found in the config file prior to [section]
					throw std::runtime_error("cannot add a configuration line without a section at line #" + std::to_string(total_lines) + " in " + filename.string());
				}

				const std::string key = matches.str(2);
				const std::string val = matches.str(3);

				CfgLine l(line, total_lines, key, val);

				auto & s = sections.back();
				s.lines.push_back(l);
			}
		}
	}

	return *this;
}


std::string Darknet::CfgFile::debug() const
{
	std::stringstream ss;
	ss	<< "Filename: " << filename.string() << std::endl
		<< "Lines parsed: " << total_lines << std::endl;

	for (const auto & section : sections)
	{
		ss << section.debug() << std::endl;
	}

	return ss.str();
}
