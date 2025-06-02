/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#pragma once

/** @file
 * This file contains C++ classes and functions for working with timing and tracking results.
 */

#ifndef DARKNET_TIMING_AND_TRACKING_USE_CTRACK

#include <chrono>
#include <map>
#include <string>

namespace Darknet
{

	/** The timing and tracking functionality is used to find places in the code where optimizations should be made.  Since
	 * the original authors are no longer active in the Darknet/YOLO project, there is a lot of unknown code.  This class
	 * is used to time each function, and the results are stored in the @ref TimingRecords object.  When %Darknet exits, the
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
		TimingAndTracking(const std::string &n, const bool r = false, const std::string &c = "");
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

		TimingRecords &add(const TimingAndTracking &tat);

		std::map<std::string, uint64_t> min_elapsed_time_per_function;	 ///< in nanoseconds
		std::map<std::string, uint64_t> max_elapsed_time_per_function;	 ///< in nanoseconds
		std::map<std::string, uint64_t> total_elapsed_time_per_function; ///< in nanoseconds
		std::map<std::string, uint64_t> number_of_calls_per_function;
		std::map<std::string, bool> reviewed_per_function;
		std::map<std::string, std::string> comment_per_function;
	};

}
#endif // DARKNET_TIMING_AND_TRACKING_USE_CTRACK

// This define DARKNET_TIMING_AND_TRACKING_ENABLED might be set by cmake/command line.
// It can still be used to globally enable/disable any timing, even if ctrack is selected.
#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED

#ifdef DARKNET_TIMING_AND_TRACKING_USE_CTRACK

#include "ctrack.hpp" // Assuming ctrack.hpp is in the include path
#include <iostream>	  // For printing ctrack results
#include <string>	  // For ctrack::result_as_string

inline void darknet_ctrack_print_results()
{
	// Check if any ctrack macros were actually used and ctrack is not globally disabled.
	// store::thread_cnt is initialized to -1 and incremented for each thread that uses ctrack.
#ifndef CTRACK_DISABLE
	ctrack::ctrack_result_settings settings;
	settings.min_percent_active_exclusive = 0.5;
	std::cout << "\n===== ctrack Performance Analysis =====\n";
	auto start = std::chrono::high_resolution_clock::now();
	// ctrack::result_print() is a void function that prints to std::cout
	ctrack::result_print(settings);
	std::cout << "=====================================\n"
			  << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "Elapsed time for ctrack calculations " << elapsed << " s" << std::endl;
#else
	std::cout << "\n===== ctrack is disabled (CTRACK_DISABLE is defined) =====\n"
			  << std::endl;
#endif
}

/// Create a ctrack event using CTRACK_NAME.
#define TAT(n) CTRACK_NAME(n)
#define TAT_REVIEWED(n, d) CTRACK_NAME(n)
#define TAT_COMMENT(n, c) CTRACK_NAME(n)
#define TAT_SKIP(...)
/// Similar to @ref TAT() but indicate this function or method was reviewed, as well as the date when it was last reviewed.
// #define TAT_REVIEWED(n, d) CTRACK_NAME((std::string(n) + " -> " +" [Reviewed: " + d + "]").c_str())

/// Similar to @ref TAT() but with a comment.
// #define TAT_COMMENT(n, c) CTRACK_NAME((std::string(n) + " -> " +  " [Comment: " + c + "]").c_str())

#define TATPARMS __builtin_FUNCTION()
// #define TT_PRINT darknet_ctrack_print_results() // This is now handled by CTrackLifetimeManager

#else // Original Darknet Timing (DARKNET_TIMING_AND_TRACKING_USE_CTRACK is not defined)

inline void darknet_ctrack_print_results()
{
}
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
#define TAT_SKIP(n, d) TAT_REVIEWED(n, d)
#define TT_PRINT /* Do nothing if not using ctrack and original timing */

#endif // DARKNET_TIMING_AND_TRACKING_USE_CTRACK

#else // DARKNET_TIMING_AND_TRACKING_ENABLED is not defined

#define TAT(...)
#define TAT_REVIEWED(...)
#define TAT_COMMENT(...)
#define TAT_SKIP(...)
#define TATPARMS ""
#define TT_PRINT /* Do nothing if timing is not enabled */

#endif // DARKNET_TIMING_AND_TRACKING_ENABLED
