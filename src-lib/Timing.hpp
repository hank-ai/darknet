/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#pragma once

/** @file
 * This file used to contain C++ classes, functions, and macros for working with timing and tracking results.
 *
 * The timing and tracking functionality is used to find places in the code where optimizations should be made.  Since
 * the original authors are no longer active in the Darknet/YOLO project, there is a lot of unknown code.  This class
 * is used to time each function.  When %Darknet exits, the results are shown in a table.
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
 *
 * In June of 2025, Darknet's "Timing and Tracking" functionality was replaced by CTrack:  https://github.com/Compaile/ctrack
 * See this pull request for additional details:  https://github.com/hank-ai/darknet/pull/126/
 * Merged into Darknet V5.
 */

// This define DARKNET_TIMING_AND_TRACKING_ENABLED might be set by cmake/command line.
// It can still be used to globally enable/disable any timing, even if ctrack is selected.
#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED

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
	std::cout << std::endl << "===== ctrack Performance Analysis =====" << std::endl;
	auto start = std::chrono::high_resolution_clock::now();
	// ctrack::result_print() is a void function that prints to std::cout
	ctrack::result_print(settings);
	std::cout << "=====================================" << std::endl << std::endl;
	auto end = std::chrono::high_resolution_clock::now();
	auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
	std::cout << "Elapsed time for ctrack calculations " << elapsed << " s" << std::endl;
#else
	std::cout << std::endl << "===== ctrack is disabled (CTRACK_DISABLE is defined) =====" << std::endl << std::endl;
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

#else // DARKNET_TIMING_AND_TRACKING_ENABLED is not defined

#define TAT(...)
#define TAT_REVIEWED(...)
#define TAT_COMMENT(...)
#define TAT_SKIP(...)
#define TATPARMS ""
#define TT_PRINT /* Do nothing if timing is not enabled */

#endif // DARKNET_TIMING_AND_TRACKING_ENABLED
