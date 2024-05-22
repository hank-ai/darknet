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
			bool is_set(const std::string & arg, const bool default_value = false) const;

			/** Get a CLI argument based on the name.  For example, if you call it with @p "thresh" you'd get the threshold
			 * argument with the default value of 0.24 or whatever the user typed on the CLI.
			 */
			const ArgsAndParms & get(const std::string & arg) const;

			float get(const std::string & arg, const float f) const;

			int get(const std::string & arg, const int i) const;

			/// Return a name for the current thread.  Thread name must previous have been added using @ref set_thread_name().
			std::string get_thread_name();


			/// Set a name to use for the given thread.  @see @ref del_thread_name()
			void set_thread_name(const std::thread::id & tid, const std::string & name);


			/// Alias for @ref set_thread_name().
			void set_thread_name(const std::thread & t, const std::string & name)
			{
				set_thread_name(t.get_id(), name);
			}

			/// Alias for @ref set_thread_name().  Uses the ID of the current running thread.
			void set_thread_name(const std::string & name)
			{
				set_thread_name(std::this_thread::get_id(), name);
			}

			/// Delete the thread name from the map.  Does nothing if this thread wasn't given a name.
			void del_thread_name(const std::thread::id & tid);

			/// Alias for @ref del_thread_name().
			void del_thread_name(const std::thread & t)
			{
				del_thread_name(t.get_id());
			}

			/// Alias for @ref del_thread_name().  Uses the ID of the current running thread.
			void del_thread_name()
			{
				del_thread_name(std::this_thread::get_id());
			}

			/** This bool gets set by @ref darknet_fatal_error() when a thread terminates and Darknet must exit.  This causes
			 * training to finish early, and also prevents Darknet from logging any more (misleading) errors that happen on
			 * additional threads.
			 */
			std::atomic<bool> must_immediately_exit;

			/// When @p -dont_show has been set, this value will be set to @p false.
			bool is_shown;

			/// Determines if ANSI colour output will be used with the console output.  Defaults to @p true on Linux and @p false on Windows.
			bool colour_is_enabled;

			/// Whether Darknet was started with the --verbose flag.  Default is @p false.
			bool is_verbose;

			/// Every argument starting with @p argv[0], unmodified, and in the exact order they were specified.
			VStr argv;

			/** A map of all arguments starting with @p argv[1].  Note this only has the arguments the user specified, not a
			 * collection of "all" possible arguments.  @see @ref Darknet::get_all_possible_arguments()
			 */
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

			/// @{ Name the threads that we create in case we have to report an error.
			std::mutex thread_names_mutex;
			std::map<std::thread::id, std::string> thread_names;
			/// @}
	};
}
