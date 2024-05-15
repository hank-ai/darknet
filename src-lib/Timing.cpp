#include "darknet_internal.hpp"


namespace
{
	Darknet::TimingRecords & get_tr()
	{
		/// There is only 1 of these objects.  All the the tracking/timing information is stored in this object.
		static Darknet::TimingRecords tr;

		return tr;
	}

	/// Mutex used to lock access to @ref tr, to ensure we're not modifying the STL containers at the same time across multiple threads.
	std::timed_mutex timing_and_tracking_container_mutex;
}


Darknet::TimingAndTracking::TimingAndTracking(const std::string& n, const bool r, const std::string & c)
{
	#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED

	name		= n;
	reviewed	= r;
	comment		= c;
	start_time	= std::chrono::high_resolution_clock::now();

	#endif

	return;
}


Darknet::TimingAndTracking::~TimingAndTracking()
{
	#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED

	end_time = std::chrono::high_resolution_clock::now();

	get_tr().add(*this);

	#endif

	return;
}


Darknet::TimingRecords::TimingRecords()
{
	return;
}


Darknet::TimingRecords::~TimingRecords()
{
	#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED

	// Remember this is the destruction of a *static* object.  By the time we get here,
	// everything has been taken down, main() has stopped running, and no other static
	// object can be relied upon.
	//
	// Do not attempt to use the colour codes in this method.  The colour table has already
	// been destructed which leads to strange segfaults which are very difficult to debug.

	std::scoped_lock lock(timing_and_tracking_container_mutex);

	// sort the calls by total time
	VStr sorted_names;
	sorted_names.reserve(total_elapsed_time_per_function.size());
	for (const auto & [k, v] : total_elapsed_time_per_function)
	{
		sorted_names.push_back(k);
	}
	std::sort(sorted_names.begin(), sorted_names.end(),
			[&](const std::string & lhs, const std::string & rhs)
			{
				// sort by total time

				const auto & lhs_nanoseconds = total_elapsed_time_per_function[lhs];
				const auto & rhs_nanoseconds = total_elapsed_time_per_function[rhs];

				if (lhs_nanoseconds != rhs_nanoseconds)
				{
					return lhs_nanoseconds > rhs_nanoseconds;
				}

				// ...unless the total time is exactly the same, in which case sort by the number of calls
				return number_of_calls_per_function[lhs] > number_of_calls_per_function[rhs];
			});

	const VStr cols =
	{
		"calls",
		"min",
		"max",
		"total",
		"average",
		"reviewed",
		"comment",
		"function"
	};
	const MStrInt m =
	{
		{"calls"	, 12},
		{"min"		, 8},
		{"max"		, 8},
		{"total"	, 12},
		{"average"	, 12},
		{"reviewed"	, 8},
		{"comment"	, 18},
		{"function"	, 8},
	};

	std::cout
		<< "               +---------------------------------------------------+" << std::endl
		<< "               | min, max, total, and average are in milliseconds  |" << std::endl;

	std::string seperator;
	for (const auto & name : cols)
	{
		const int len = m.at(name);
		seperator += "+-" + std::string(len, '-') + "-";
	}
	std::cout << seperator << std::endl;
	for (const auto & name : cols)
	{
		std::cout << "| " << std::setw(m.at(name)) << name << " ";
	}
	std::cout << std::endl << seperator << std::endl;

	const double nanoseconds_to_milliseconds = 1000000.0;

	size_t skipped = 0;
	for (const auto & name : sorted_names)
	{
		const uint64_t calls				= number_of_calls_per_function.at(name);
		const uint64_t total_milliseconds	= std::round(total_elapsed_time_per_function.at(name)	/ nanoseconds_to_milliseconds);
		const uint64_t min_milliseconds		= std::round(min_elapsed_time_per_function.at(name)		/ nanoseconds_to_milliseconds);
		const uint64_t max_milliseconds		= std::round(max_elapsed_time_per_function.at(name)		/ nanoseconds_to_milliseconds);
		const double average_milliseconds	= total_elapsed_time_per_function.at(name)				/ nanoseconds_to_milliseconds / calls;
		const std::string reviewed			= (reviewed_per_function.at(name) ? "yes" : "");
		const std::string comment			= comment_per_function.at(name);

		if (total_milliseconds < 10.0f)
		{
			skipped ++;
			continue;
		}

		auto display_name = name.substr(0, 100);
		if (name.size() > 100)
		{
			display_name += "...";
		}

		std::cout
			<< "| " << std::setw(m.at("calls"	)) << calls															<< " "
			<< "| " << std::setw(m.at("min"		)) << min_milliseconds												<< " "
			<< "| " << std::setw(m.at("max"		)) << max_milliseconds												<< " "
			<< "| " << std::setw(m.at("total"	)) << total_milliseconds											<< " "
			<< "| " << std::setw(m.at("average"	)) << std::fixed << std::setprecision(1) << average_milliseconds	<< " "
			<< "| " << std::setw(m.at("reviewed")) << reviewed														<< " "
			<< "| " << std::setw(m.at("comment"	)) << std::left << comment << std::right							<< " "
			<< "| " << display_name
			<< std::endl;
	}

	std::cout
		<< seperator << std::endl
		<< "Entries skipped:  " << skipped << std::endl;

	#endif

	return;
}


Darknet::TimingRecords & Darknet::TimingRecords::add(const Darknet::TimingAndTracking & tat)
{
	#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED

	const auto duration = tat.end_time - tat.start_time;
	const auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

	std::scoped_lock lock(timing_and_tracking_container_mutex);

	number_of_calls_per_function[tat.name] ++;
	total_elapsed_time_per_function[tat.name] += nanoseconds;

	reviewed_per_function[tat.name] = tat.reviewed;
	comment_per_function[tat.name] = tat.comment;

	if (min_elapsed_time_per_function.count(tat.name) == 0 or nanoseconds < min_elapsed_time_per_function[tat.name])
	{
		min_elapsed_time_per_function[tat.name] = nanoseconds;
	}
	if (max_elapsed_time_per_function.count(tat.name) == 0 or nanoseconds > max_elapsed_time_per_function[tat.name])
	{
		max_elapsed_time_per_function[tat.name] = nanoseconds;
	}

	#endif

	return *this;
}
