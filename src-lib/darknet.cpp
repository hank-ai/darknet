#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


extern "C"
{
	void darknet_set_verbose(const bool flag)
	{
		Darknet::set_verbose(flag);
	}


	void darknet_set_trace(const bool flag)
	{
		Darknet::set_trace(flag);
	}
}


void Darknet::set_verbose(const bool flag)
{
	cfg_and_state.is_verbose = flag;

	// when verbose is disabled, then disable trace as well
	if (not flag)
	{
		set_trace(flag);
	}
}


void Darknet::set_trace(const bool flag)
{
	cfg_and_state.is_trace = flag;

	// when trace is enabled, then enable verbose as well
	if (flag)
	{
		set_verbose(flag);
	}
}
