#pragma once

#ifndef __cplusplus
#error "The darknet project requires the use of a C++ compiler."
#endif


#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>


/// The namespace for the C++ Darknet API.
namespace Darknet
{
	/// @{ Convenient simple types for the C++ API.
	using MStr		= std::map<std::string, std::string>;
	using MStrInt	= std::map<std::string, int>;
	using SStr		= std::set<std::string>;
	using VInt		= std::vector<int>;
	using VStr		= std::vector<std::string>;
	using VFloat	= std::vector<float>;
	/// @}

	/** Set the @ref Darknet::CfgAndState::is_verbose flag.  When enabled, extra information will be sent to @p STDOUT.
	 * Default value is @p false.
	 * @see @ref Darknet::set_trace()
	 * @since 2024-07-14
	 */
	void set_verbose(const bool flag);

	/** Set the @ref Darknet::CfgAndState::is_trace flag.  When enabled, debug information will be sent to @p STDOUT.
	 * Default value is @p false.
	 * @see @ref Darknet::set_verbose()
	 * @since 2024-07-14
	 */
	void set_trace(const bool flag);
}


#include "darknet.h"
#include "darknet_enums.hpp"
#include "darknet_cfg.hpp"
