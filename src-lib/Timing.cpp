#include "darknet_internal.hpp"


namespace
{
	static Darknet::TimingRecords tr;

	static std::timed_mutex timing_and_tracking_container_mutex;
}


Darknet::TimingAndTracking::TimingAndTracking(const std::string& n, const bool r, const std::string & c)
{
	name		= n;
	reviewed	= r;
	comment		= c;
	start_time	= std::chrono::high_resolution_clock::now();

	return;
}


Darknet::TimingAndTracking::~TimingAndTracking()
{
	end_time = std::chrono::high_resolution_clock::now();

	tr.add(*this);

	return;
}


Darknet::TimingRecords::TimingRecords()
{
	return;
}


Darknet::TimingRecords::~TimingRecords()
{
	#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED

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
				return total_elapsed_time_per_function[lhs] > total_elapsed_time_per_function[rhs];
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
		{"calls"	, 8},
		{"min"		, 8},
		{"max"		, 8},
		{"total"	, 12},
		{"average"	, 8},
		{"reviewed"	, 8},
		{"comment"	, 10},
		{"function"	, 8},
	};

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

	size_t skipped = 0;
	for (const auto & name : sorted_names)
	{
		const auto & calls				= number_of_calls_per_function.at(name);
		const auto & total_milliseconds	= total_elapsed_time_per_function.at(name);
		const auto & min_milliseconds	= min_elapsed_time_per_function.at(name);
		const auto & max_milliseconds	= max_elapsed_time_per_function.at(name);
		const auto average_milliseconds	= float(total_milliseconds) / float(calls);
		const std::string reviewed		= (reviewed_per_function.at(name) ? "yes" : "");
		const std::string comment		= comment_per_function.at(name);

		if (total_milliseconds < 100.0f)
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
			<< "| " << std::setw(m.at("comment"	)) << comment														<< " "
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
	const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

	const bool is_locked = timing_and_tracking_container_mutex.try_lock_for(std::chrono::seconds(3));

	number_of_calls_per_function[tat.name] ++;
	total_elapsed_time_per_function[tat.name] += milliseconds;

	reviewed_per_function[tat.name] = tat.reviewed;
	comment_per_function[tat.name] = tat.comment;

	if (min_elapsed_time_per_function.count(tat.name) == 0 or milliseconds < min_elapsed_time_per_function[tat.name])
	{
		min_elapsed_time_per_function[tat.name] = milliseconds;
	}
	if (max_elapsed_time_per_function.count(tat.name) == 0 or milliseconds > max_elapsed_time_per_function[tat.name])
	{
		max_elapsed_time_per_function[tat.name] = milliseconds;
	}

	if (is_locked)
	{
		timing_and_tracking_container_mutex.unlock();
	}

	#endif

	return *this;
}
