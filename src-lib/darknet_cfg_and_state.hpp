#pragma once

#include "darknet_internal.hpp"


namespace Darknet
{
	class CfgAndState final
	{
		public:

			/// Constructor.
			CfgAndState();

			/// Destructor.
			~CfgAndState();

			/// Get a reference to the singleton used by Darknet.
			static CfgAndState & get();

			/// Clear out all settings and state to a known initial state.
			CfgAndState & reset();

			/// Process @p argv[] from @p main() and store the results in @p argv and @p args.
			CfgAndState & process_arguments(int argc, char ** argp);

			/** Determine if the user specified the given option, or if unspecified then use the default value.
			 *
			 * For example:
			 * ~~~~
			 * if (cfg.is_set("map"))
			 * {
			 *     // do something when the user specified -map
			 * }
			 * ~~~~
			 */
			bool is_set(const std::string arg, const bool default_value = false);

			/** This bool gets set by @ref darknet_fatal_error() when a thread terminates and Darknet must exit.  This causes
			 * training to finish early, and also prevents Darknet from logging any more (misleading) errors that happen on
			 * additional threads.
			 */
			std::atomic<bool> must_immediately_exit;

			/// When @p -dont_show has been set, this value will be set to @p false.
			bool is_shown;

			/// Determines if ANSI colour output will be used with the console output.  Defaults to @p true on Linux and @p false on Windows.
			bool colour_is_enabled;

			/// Every argument starting with @p argv[0], unmodified, and in the exact order they were specified.
			VStr argv;

			/// A map of all arguments starting with @p argv[1].
			MArgsAndParms args;

			std::string command;
			std::string function;

			VStr filenames;

			std::filesystem::path cfg_filename;
			std::filesystem::path data_filename;
			std::filesystem::path names_filename;
			std::filesystem::path weights_filename;

			/// Parameters that were unrecognized.
			VStr additional_arguments;

			/// The index of the GPU to use.  @p -1 means no GPU is selected.
			int gpu_index;
	};
}
