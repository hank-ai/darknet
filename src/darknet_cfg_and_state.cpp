#include "darknet_cfg_and_state.hpp"
#include <iostream>

#ifdef WIN32
#include <Windows.h>
#endif


Darknet::CfgAndState Darknet::cfg_and_state;


Darknet::CfgAndState::CfgAndState()
{
	reset();

	return;
}


Darknet::CfgAndState::~CfgAndState()
{
	return;
}


Darknet::CfgAndState & Darknet::CfgAndState::reset()
{
	must_immediately_exit	= false;
	is_shown				= true;
	colour_is_enabled		= true;
	argv					.clear();
	args					.clear();
	command					.clear();
	function				.clear();
	filenames				.clear();
	cfg_filename			.clear();
	data_filename			.clear();
	names_filename			.clear();
	weights_filename		.clear();
	additional_arguments	.clear();

	return *this;
}


std::string convert_to_lowercase_alphanum(const std::string & arg)
{
	std::string str;
	str.reserve(arg.length());
	for (auto & c : arg)
	{
		if (std::isalnum(c))
		{
			str.push_back(std::tolower(c));
		}
	}

	return str;
}


Darknet::CfgAndState & Darknet::CfgAndState::process_arguments(int argc, char ** argp)
{
	argv.clear();
	args.clear();

	argv.reserve(argc);

	const auto & all_known_args = Darknet::get_all_possible_arguments();

	for (int idx = 0; idx < argc; idx ++)
	{
		errno = 0;

		const std::string original_arg = argp[idx];
		argv.push_back(original_arg);

		if (idx == 0)
		{
			// don't expect argv[0] to be in the list of supported commands
			continue;
		}

		const std::string str = convert_to_lowercase_alphanum(original_arg);

		auto iter = all_known_args.find(str);
		if (iter == all_known_args.end())
		{
			// before we decide this is an unknown command, look through the alternate spellings
			for (iter = all_known_args.begin(); iter != all_known_args.end(); iter ++)
			{
				if (iter->name_alternate == str)
				{
					// found an item with a matching alternate spelling!
					break;
				}
			}
		}

		if (iter == all_known_args.end())
		{
			// see if the argument is a valid filename
			std::filesystem::path path(original_arg);
			if (std::filesystem::exists(path))
			{
				filenames.push_back(path.string());

				const auto extension = path.extension();
				if (extension == ".cfg"		and cfg_filename	.empty())	{ cfg_filename		= path; }
				if (extension == ".data"	and data_filename	.empty())	{ data_filename		= path; }
				if (extension == ".names"	and names_filename	.empty())	{ names_filename	= path; }
				if (extension == ".weights"	and weights_filename.empty())	{ weights_filename	= path; }

				// we don't have an "iter" for filenames, so loop back up to the top of the for() loop
				continue;
			}
		}

		if (iter == all_known_args.end())
		{
			// this argument is unknown to Darknet (we even looked through the alternate spellings)
			if (idx <= 1)
			{
				throw std::invalid_argument("unknown argument \"" + original_arg + "\" (argument #" + std::to_string(idx) + ")");
			}
			else
			{
				additional_arguments.push_back(original_arg);
				display_warning_msg("skipped validating of argument #" + std::to_string(idx) + " (" + original_arg + ")");
				std::cout << std::endl;
				continue;
			}
		}

		if (args.count(iter->name) > 0)
		{
			// why was this parameter specified more than once?
			throw std::invalid_argument("argument \"" + original_arg + "\" specified more than once (argument #" + std::to_string(args[iter->name].arg_index) + " and #" + std::to_string(idx) + ")");
		}

		ArgsAndParms args_and_parms	= *iter;
		args_and_parms.arg_index	= idx;
		args[iter->name]			= args_and_parms;

		if (args_and_parms.type == ArgsAndParms::EType::kCommand)
		{
			if (not cfg_and_state.command.empty())
			{
				if (cfg_and_state.function.empty())
				{
					// we already have a command, so assume this is a function (such as "darknet test" and "darknet detector test")
					cfg_and_state.function = iter->name;
				}
				else
				{
//					throw std::invalid_argument("command \"" + cfg_and_state.command + "\" is already set while processing new command argument \"" + original_arg + "\"");
					display_warning_msg("command \"" + cfg_and_state.command + "\" is already set while processing new command argument \"" + original_arg + "\"\n");
				}
			}
			else
			{
				cfg_and_state.command = iter->name;
			}
		}

		if (args_and_parms.type == ArgsAndParms::EType::kFunction and original_arg[0] != '-') // don't mix up func "map" with parm "-map"
		{
			if (not cfg_and_state.function.empty())
			{
				/// @todo need to fix things like the function "map" and the optional training parameter "-map" which would trigger this code
//				throw std::invalid_argument("function \"" + cfg_and_state.function + "\" is already set while processing new function argument \"" + original_arg + "\"");
//				display_warning_msg("function \"" + cfg_and_state.function + "\" is already set while processing new function argument \"" + original_arg + "\"\n");
			}
			else
			{
				cfg_and_state.function = iter->name;
			}
		}
	}

	if (args.count("dontshow") > 0)
	{
		is_shown = false;
	}

	if (args.count("colour") > 0)
	{
		colour_is_enabled = true;
	}

	if (args.count("nocolour") > 0)
	{
		colour_is_enabled = false;
	}

#ifdef WIN32
	if (colour_is_enabled)
	{
		// enable VT100 and ANSI escape handling in Windows
		bool success = false;
		for (auto handle : {STD_OUTPUT_HANDLE, STD_ERROR_HANDLE})
		{
			HANDLE std_handle = GetStdHandle(handle);
			if (std_handle != INVALID_HANDLE_VALUE)
			{
				DWORD mode = 0;
				if (GetConsoleMode(std_handle, &mode))
				{
					mode |= ENABLE_PROCESSED_OUTPUT;
					mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
					SetConsoleMode(std_handle, mode);
					success = true;
				}
			}
		}
		if (not success)
		{
			colour_is_enabled = false;
		}
	}
#endif

	return *this;
}


bool Darknet::CfgAndState::is_set(const std::string arg, const bool default_value)
{
	const std::string name = convert_to_lowercase_alphanum(arg);

	if (args.count(name) > 0)
	{
		return true;
	}

	return default_value;
}
