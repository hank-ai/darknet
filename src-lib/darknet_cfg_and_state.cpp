#include "darknet_internal.hpp"

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif


namespace
{
	void parse_skip_classes(Darknet::SInt & classes_to_ignore, std::string str)
	{
		TAT(TATPARMS);

		// take a string like this:   2,5-7,9
		// ...and populate the std::set with the individual values 2, 5, 6, 7, and 9

		classes_to_ignore.clear();

		const std::regex rx(
				"(\\d+)-(\\d+)"	// range
				"|"
				"(\\d+)"		// single value
				);

		while (not str.empty())
		{
			std::smatch matches;

			if (not std::regex_search(str, matches, rx))
			{
				break;
			}

			if (matches.length() == 1)
			{
				// we have a single value
				const int idx = std::stod(matches[3].str());
				classes_to_ignore.insert(idx);
			}
			else if (matches.length() > 2)
			{
				// we have a range
				const int i1 = std::stod(matches[1].str());
				const int i2 = std::stod(matches[2].str());
				for (auto idx = i1; idx <= i2; idx ++)
				{
					classes_to_ignore.insert(idx);
				}
			}

			const size_t len = matches.position() + matches.length();
			str.erase(0, len);
		}

		return;
	}
}


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

	/* Default is to use std::cout for console output.  Do *NOT* call set_output_stream() from here,
	 * since it will cause infinite recursion when it attempts to call CfgAndState::get().
	 */
	output = &std::cout;
	*output << std::fixed; // if this is changed, see set_output_stream()

	must_immediately_exit	= false;
	is_shown				= true;
	colour_is_enabled		= true;
	is_verbose				= false;
	is_trace				= false;

#ifdef DARKNET_GPU
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

	#ifdef DARKNET_USE_OPENBLAS
	// "If your application is already multi-threaded, it will conflict with OpenBLAS
	// multi-threading. Thus, you must set OpenBLAS to use single thread as following."
	//
	// 2025-07-10
	// Source:  http://www.openmathlib.org/OpenBLAS/docs/faq/#how-can-i-use-openblas-in-multi-threaded-applications
	openblas_set_num_threads(1);
	#endif

	return *this;
}


Darknet::CfgAndState & Darknet::CfgAndState::process_arguments(int argc, char ** argp)
{
	TAT(TATPARMS);

	#ifdef DEBUG
	#if 0 // force Darknet to run specific code in the debugger
	if (argc == 1)
	{
		// if no argument has been specified _AND_ we're in debug mode, then run a "fake" command.
		// This makes it easier to run darknet in a debugger without having to remember all the args.

		static std::vector<char*> cmd =
		{
			// process a single image file
//			"darknet", "detector", "test", "LegoGears.cfg", "LegoGears.data", "LegoGears_best.weights", "set_01/DSCN1580_frame_000000.jpg"

			// train the network
//			"darknet", "detector", "train", "-map", "-dont_show", "LegoGears.data", "LegoGears.cfg"

			// calculate mAP%
//			"darknet", "detector", "map", "LegoGears.cfg", "LegoGears.data", "LegoGears_best.weights"

			// recalculate anchors
			"darknet", "detector", "calcanchors", "LegoGears.data", "-show", "-num_of_clusters", "6", "-width", "224", "-height", "160"
		};

		int c = cmd.size();
		char ** v = &cmd[0];

		*output << "Inserting " << c << " fake arguments because we're running darknet in DEBUG mode!" << std::endl;
		for (int i = 0; i < c; i ++)
		{
			*output << "argv[" << i << "] = \"" << v[i] << "\"" << std::endl;
		}

		return process_arguments(c, v);
	}
	#endif
	#endif

	argv.clear();
	args.clear();

	argv.reserve(argc);

	for (int idx = 1; idx < argc; idx ++) // ignore argv[0]
	{
		argv.push_back(argp[idx]);
	}

	return process_arguments(argv);
}


Darknet::CfgAndState & Darknet::CfgAndState::process_arguments(const VStr & v, Darknet::NetworkPtr ptr)
{
	TAT(TATPARMS);

	args.clear();
	const auto & all_known_args = Darknet::get_all_possible_arguments();

	// ********************************************************************************
	// WARNING:  it is perfectly valid for this pointer to be NULL if a network
	// has not yet been loaded, so check before attempting to dereference it!
	//
	// For example, this is called with a NULL pointer from Darknet::parse_arguments().
	// ********************************************************************************
	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);

	for (int idx = 0; idx < v.size(); idx ++)
	{
		errno = 0;
		const std::string & original_arg = v.at(idx);
		const std::string str = convert_to_lowercase_alphanum(original_arg);

		// see if this parameter exists, either as primary name or an alternate spelling
		const auto iter = [&]()
		{
			for (auto i = all_known_args.begin(); i != all_known_args.end(); i++)
			{
				if (i->name == str or i->name_alternate == str)
				{
					return i;
				}
			}

			// name was not found
			return all_known_args.end();
		}();

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

			// When this is first called by Darknet::parse_arguments() prior to parameter expansion, we need to skip
			// the parms that are used to find and load the neural network.  For example, when attempting to run
			//
			//		darknet_03_display_videos LegoGears
			//
			// We don't want to warn about "LegoGears" which is eventually used to find the neural network.  When "net" is
			// NULL the assumption is made that this is early in the loading process and unknown parms should be skipped.
			if (net)
			{
				additional_arguments.push_back(original_arg);
				display_warning_msg("skipped validating of argument #" + std::to_string(idx) + " \"" + original_arg + "\" (does not appear to be a known parameter, file, or directory)");
				*output << std::endl;
			}

			continue;
		}

		if (args.count(iter->name) > 0)
		{
			// why was this parameter specified more than once?
			throw std::invalid_argument("argument \"" + original_arg + "\" specified more than once (argument #" + std::to_string(args[iter->name].arg_index) + " and #" + std::to_string(idx) + ")");
		}

		ArgsAndParms args_and_parms	= *iter;
		args_and_parms.arg_index	= idx;

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

		if (args_and_parms.type == ArgsAndParms::EType::kParameter and args_and_parms.expect_parm)
		{
			const int next_arg_idx = idx + 1;
			if (next_arg_idx < v.size())
			{
				if (args_and_parms.str.empty())
				{
					// the next parm should be a numeric value
					size_t pos = 0;
					try
					{
						args_and_parms.value = std::stof(v.at(next_arg_idx), &pos);
					}
					catch (...)
					{
						// do nothing, this is not a number
					}

					if (pos == v.at(next_arg_idx).size())
					{
						// consume the next argument
						idx ++;
					}
					else
					{
						darknet_fatal_error(DARKNET_LOC, "expected a numeric parameter after %s, not %s", v.at(idx).c_str(), v.at(next_arg_idx).c_str());
					}
				}
				else
				{
					// this is a "text" argument
					args_and_parms.str = v.at(next_arg_idx);
					// consume the next argument
					idx ++;
				}
			}
			else
			{
				darknet_fatal_error(DARKNET_LOC, "expected an additional parameter after %s", v.at(idx).c_str());
			}
		}

		args[iter->name] = args_and_parms;
	}

	if (args.count("verbose"		) > 0 or
		args.count("show_details"	) > 0) // old Darknet had "-show_details", which I didn't know about when I created "--verbose"
	{
		is_verbose = true;
	}

	if (args.count("trace") > 0)
	{
		is_verbose	= true;
		is_trace	= true;
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

	if (args.count("log") > 0)
	{
		const ArgsAndParms & log = get("log");
		set_output_stream(log.str);
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

	if (net and args.count("skipclasses"))
	{
		const ArgsAndParms & arg = get("skipclasses");

		parse_skip_classes(net->details->classes_to_ignore, arg.str);
	}

	// for debug purposes, display all arguments
	if (args.count("trace") > 0)
	{
		*output
			<< "--------------------------------" << std::endl
			<< "CMD=" << command << std::endl
			<< "FUN=" << function << std::endl
			<< "ARG=" << args.size() << std::endl;
		for (const auto & [key, val] : args)
		{
			*output
				<< "IDX=" << val.arg_index
				<< " NUM=" << val.value
				<< " EXPECT=" << val.expect_parm
				<< " KEY=" << key
				<< " VAL=" << val.name
				<< " STR=" << val.str;
			if (val.name_alternate.empty() == false)
			{
				*output << " ALT=" << val.name_alternate;
			}
			*output << std::endl;
		}
		*output << "--------------------------------" << std::endl;
	}

	static bool need_to_show_version_info = true;
	if (need_to_show_version_info)
	{
		need_to_show_version_info = false;
		Darknet::show_version_info();
	}

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

	// if we get here we haven't yet found a match, so look through all the "alternate" names as well

	for (const auto & [key, val] : args)
	{
		if (val.name_alternate == name)
		{
			return true;
		}
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

	// if we get here we don't have a perfect match, so go ahead and check the alternate names

	for (const auto & [key, val] : args)
	{
		if (val.name_alternate == name)
		{
			return val;
		}
	}

	// if we get here, this argument wasn't specified on the CLI so get the default value from the known list of arguments

	const auto & all_known_args = Darknet::get_all_possible_arguments();
	auto iter = all_known_args.find(name);
	if (iter != all_known_args.end())
	{
		return *iter;
	}

	// now check the defaults to see if we have a matching alternate name
	for (const auto & known_arg : all_known_args)
	{
		if (known_arg.name_alternate == name)
		{
			return known_arg;
		}
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

	// maybe this is an "alternate name"?

	for (const auto & [key, val] : args)
	{
		if (val.name_alternate == name)
		{
			return val.value;
		}
	}

	// arg was not found, so see if we have a "known" default we can return

	const auto & all_known_args = Darknet::get_all_possible_arguments();
	auto iter = all_known_args.find(name);
	if (iter != all_known_args.end())
	{
		return iter->value;
	}

	// and what about a matching alternate name?
	for (const auto & known_arg : all_known_args)
	{
		if (known_arg.name_alternate == name)
		{
			return known_arg.value;
		}
	}

	// no clue what this might be so use the default value that was passed in
	return f;
}


int Darknet::CfgAndState::get(const std::string & arg, const int i) const
{
	TAT(TATPARMS);

	const float f = get(arg, static_cast<float>(i));

	return static_cast<int>(f);
}


float Darknet::CfgAndState::get_float(const std::string & arg) const
{
	TAT(TATPARMS);

	const std::string name = convert_to_lowercase_alphanum(arg);

	if (args.count(name) == 1)
	{
		return args.at(name).value;
	}

	// also check the alternate names

	for (const auto & [key, val] : args)
	{
		if (val.name_alternate == name)
		{
			return val.value;
		}
	}

	darknet_fatal_error(DARKNET_LOC, "failed to find a parameter named \"%s\"", arg.c_str());
}


int Darknet::CfgAndState::get_int(const std::string & arg) const
{
	TAT(TATPARMS);

	return static_cast<int>(get_float(arg));
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
