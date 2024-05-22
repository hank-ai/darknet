#include "darknet_cfg_and_state.hpp"
#include <iostream>

#ifdef WIN32
#include <Windows.h>
#endif


Darknet::CfgAndState::CfgAndState()
{
	TAT(TATPARMS);

	reset();

	return;
}


Darknet::CfgAndState::~CfgAndState()
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgAndState & Darknet::CfgAndState::get()
{
	TAT(TATPARMS);

	static CfgAndState cfg_and_state;

	return cfg_and_state;
}


Darknet::CfgAndState & Darknet::CfgAndState::reset()
{
	TAT(TATPARMS);

	// Seeding the old C-style rand() is peppered all over the codebase for some reason.
	// I'm hoping we can do it once, and then not worry about it again.  Eventually we
	// can move to the new C++11 objects and functions for dealing with random numbers.
	std::srand(std::time(nullptr));
	// ... also see the seeding that happens in get_rnd_engine()

	// prefer using 500 over 5e+02 when outputting floats
	std::cout << std::fixed;

	must_immediately_exit	= false;
	is_shown				= true;
	colour_is_enabled		= true;
	is_verbose				= false;

#ifdef GPU
	gpu_index				= 0;
#else
	gpu_index				= -1;
#endif

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
	TAT(TATPARMS);

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
	TAT(TATPARMS);

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
			if (not command.empty())
			{
				if (function.empty())
				{
					// we already have a command, so assume this is a function (such as "darknet test" and "darknet detector test")
					function = iter->name;
				}
				else
				{
//					throw std::invalid_argument("command \"" + command + "\" is already set while processing new command argument \"" + original_arg + "\"");
					display_warning_msg("command \"" + command + "\" is already set while processing new command argument \"" + original_arg + "\"\n");
				}
			}
			else
			{
				command = iter->name;
			}
		}

		if (args_and_parms.type == ArgsAndParms::EType::kFunction and original_arg[0] != '-') // don't mix up func "map" with parm "-map"
		{
			if (not function.empty())
			{
				/// @todo need to fix things like the function "map" and the optional training parameter "-map" which would trigger this code
//				throw std::invalid_argument("function \"" + function + "\" is already set while processing new function argument \"" + original_arg + "\"");
//				display_warning_msg("function \"" + function + "\" is already set while processing new function argument \"" + original_arg + "\"\n");
			}
			else
			{
				function = iter->name;
			}
		}
	}

	if (args.count("verbose"		) > 0 or
		args.count("show_details"	) > 0) // old Darknet had "-show_details", which I didn't know about when I created "--verbose"
	{
		is_verbose = true;
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

#if 0 // for debug purposes, display all arguments
	std::cout
		<< "--------------------------------" << std::endl
		<< "CMD=" << command << std::endl
		<< "FUN=" << function << std::endl
		<< "ARG=" << args.size() << std::endl;
	for (const auto & [key, val] : args)
	{
		std::cout
			<< "IDX=" << val.arg_index
			<< " NUM=" << val.value
			<< " KEY=" << key
			<< " VAL=" << val.name;
		if (val.name_alternate.empty() == false)
		{
			std::cout << " ALT=" << val.name_alternate;
		}
		std::cout << std::endl;
	}
	std::cout << "--------------------------------" << std::endl;
#endif

	return *this;
}


bool Darknet::CfgAndState::is_set(const std::string & arg, const bool default_value) const
{
	TAT(TATPARMS);

	const std::string name = convert_to_lowercase_alphanum(arg);

	if (args.count(name) > 0)
	{
		return true;
	}

	return default_value;
}


const Darknet::ArgsAndParms & Darknet::CfgAndState::get(const std::string & arg) const
{
	TAT(TATPARMS);

	const std::string name = convert_to_lowercase_alphanum(arg);

	if (args.count(name))
	{
		return args.at(name);
	}

	// if we get here, this argument wasn't specified on the CLI so get the default value from the known list of arguments

	const auto & all_known_args = Darknet::get_all_possible_arguments();
	auto iter = all_known_args.find(name);
	if (iter != all_known_args.end())
	{
		return *iter;
	}

	// if we get here, we have no idea what this argument might be

	throw std::invalid_argument("cannot find argument \"" + arg + "\"");
}


float Darknet::CfgAndState::get(const std::string & arg, const float f) const
{
	TAT(TATPARMS);

	const std::string name = convert_to_lowercase_alphanum(arg);

	if (args.count(name))
	{
		return args.at(name).value;
	}

	// arg was not found, so see if we have a "known" default we can return

	const auto & all_known_args = Darknet::get_all_possible_arguments();
	auto iter = all_known_args.find(name);
	if (iter != all_known_args.end())
	{
		return iter->value;
	}

	// no clue what this might be so use the default value that was passed in
	return f;
}


int Darknet::CfgAndState::get(const std::string & arg, const int i) const
{
	const float f = get(arg, static_cast<float>(i));

	return static_cast<int>(f);
}


void Darknet::CfgAndState::set_thread_name(const std::thread::id & tid, const std::string & name)
{
	TAT(TATPARMS);

	if (name.empty() == false and thread_names.count(tid) == 0)
	{
		std::scoped_lock lock(thread_names_mutex);
		thread_names[tid] = name;
	}

	return;
}


std::string Darknet::CfgAndState::get_thread_name()
{
	TAT(TATPARMS);

	std::string name = "unknown thread";

	const auto id = std::this_thread::get_id();
	if (thread_names.count(id) != 0)
	{
		std::scoped_lock lock(thread_names_mutex);
		name = thread_names.at(id);
	}

	return name;
}


void Darknet::CfgAndState::del_thread_name(const std::thread::id & tid)
{
	TAT(TATPARMS);

	if (thread_names.count(tid) != 0)
	{
		std::scoped_lock lock(thread_names_mutex);
		thread_names.erase(tid);
	}

	return;
}
