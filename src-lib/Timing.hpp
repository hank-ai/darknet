#pragma once

/** @file
 * This file contains C++ classes and functions for working with timing and tracking results.
 */

#include <chrono>
#include <map>
#include <string>


namespace Darknet
{
	/** The timing and tracking functionality is used to find places in the code where optimizations should be made.  Since
	 * the original authors are no longer active in the Darknet/YOLO project, there is a lot of unknown code.  This class
	 * is used to time each function, and the results are stored in the @ref TimingRecords object.  When Darknet exits, the
	 * results are shown in a table.
	 *
	 * Running with this enabled will slow down Darknet!  By a significant amount.  It is not meant to be used by "normal"
	 * users, but instead by developers.  When building Darknet, you have to give it an extra parameter when you run the
	 * cmake command.  For example:
	 *
	 * ~~~~
	 * cd build
	 * cmake -DENABLE_TIMING_AND_TRACKING=ON -DCMAKE_BUILD_TYPE=Debug ..
	 * ~~~~
	 *
	 * Note that @p "Debug" is not needed for this functionality.  It will work just the same in @p "Release" mode.
	 */
	class TimingAndTracking final
	{
		public:

			TimingAndTracking(const std::string & n, const bool r = false, const std::string & c = "");
			~TimingAndTracking();

			bool reviewed;
			std::string name;
			std::string comment;
			std::chrono::high_resolution_clock::time_point start_time;
			std::chrono::high_resolution_clock::time_point end_time;
	};

	/** An object of this type is statically instantiated in Timing.cpp.  It is used to store all the results from the
	 * various @ref TimingAndTracking objects.  Upon destruction, this object will format all of the entries and display
	 * then on the console.  Note this is expensive, and is only enabled when the necessary build flag has been set.
	 * See the documentation in @ref TimingAndTracking.
	 */
	class TimingRecords final
	{
		public:

			TimingRecords();
			~TimingRecords();

			TimingRecords & add(const TimingAndTracking & tat);

			std::map<std::string, uint64_t>		min_elapsed_time_per_function;		///< in nanoseconds
			std::map<std::string, uint64_t>		max_elapsed_time_per_function;		///< in nanoseconds
			std::map<std::string, uint64_t>		total_elapsed_time_per_function;	///< in nanoseconds
			std::map<std::string, uint64_t>		number_of_calls_per_function;
			std::map<std::string, bool>			reviewed_per_function;
			std::map<std::string, std::string>	comment_per_function;
	};
}

#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED

	/// Create a @ref Darknet::TimingAndTracking object on the stack to generate some information allowing us to debug which parts of the code takes a long time to run.
	#define TAT(n) Darknet::TimingAndTracking tat(n)

	/// Similar to @ref TAT() but indicate this function or method was reviewed, as well as the date when it was last reviewed.
	#define TAT_REVIEWED(n, d) Darknet::TimingAndTracking tat(n, true, d)

	/// Similar to @ref TAT() but with a comment.
	#define TAT_COMMENT(n, c) Darknet::TimingAndTracking tat(n, false, c)

	#ifdef WIN32
		#define TATPARMS __FUNCTION__
	#else
		#define TATPARMS __PRETTY_FUNCTION__
	#endif

#else

	#define TAT(...)
	#define TAT_REVIEWED(...)
	#define TAT_COMMENT(...)
	#define TATPARMS ""

#endif
