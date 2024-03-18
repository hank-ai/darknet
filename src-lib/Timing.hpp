#pragma once

/** @file
 * This file contains C++ classes and functions for working with timing and tracking results.
 */

#include <chrono>
#include <map>
#include <string>


namespace Darknet
{
	class TimingAndTracking final
	{
		public:

			TimingAndTracking(const std::string & n);
			~TimingAndTracking();

			std::string name;
			std::chrono::high_resolution_clock::time_point start_time;
			std::chrono::high_resolution_clock::time_point end_time;
	};

	class TimingRecords final
	{
		public:

			TimingRecords();
			~TimingRecords();

			TimingRecords & add(const TimingAndTracking & tat);

			std::map<std::string, size_t> min_elapsed_time_per_function;
			std::map<std::string, size_t> max_elapsed_time_per_function;
			std::map<std::string, size_t> total_elapsed_time_per_function;
			std::map<std::string, size_t> number_of_calls_per_function;
	};
}

#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED

/// Create a @ref Darknet::TimingAndTracking object on the stack to generate some information allowing us to debug which parts of the code takes a long time to run.
#define TAT(n) Darknet::TimingAndTracking tat(n)

#define TATPARMS __PRETTY_FUNCTION__

#else

#define TAT(...)
#define TATPARMS ""

#endif
