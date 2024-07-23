#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

/** @file
 * Include this file to get access to the Darknet/YOLO C++ API.  Note there are additional Darknet/YOLO header files
 * included at the bottom of this file.
 */

#include <filesystem>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>


/// The namespace for the C++ Darknet API.
namespace Darknet
{
	/// @{ Convenient simple types used in the Darknet/YOLO C++ API.
	using MStr		= std::map<std::string, std::string>;
	using MStrInt	= std::map<std::string, int>;
	using SStr		= std::set<std::string>;
	using VFloat	= std::vector<float>;
	using VInt		= std::vector<int>;
	using VStr		= std::vector<std::string>;
	/// @}

	/** Set the @ref Darknet::CfgAndState::is_verbose flag.  When enabled, extra information will be sent to @p STDOUT.
	 * Default value is @p false.
	 *
	 * @note Disabling @p verbose will also disable @p trace.
	 *
	 * @see @ref Darknet::set_trace()
	 * @see @ref darknet_set_verbose()
	 * @since 2024-07-14
	 */
	void set_verbose(const bool flag);

	/** Set the @ref Darknet::CfgAndState::is_trace flag.  When enabled, debug information will be sent to @p STDOUT.
	 * Default value is @p false.
	 *
	 * @note Enabling @p trace will also enable @p verbose.
	 *
	 * @see @ref Darknet::set_verbose()
	 * @see @ref darknet_set_trace()
	 * @since 2024-07-14
	 */
	void set_trace(const bool flag);
}


#include "darknet.h"
#include "darknet_enums.hpp"
#include "darknet_cfg.hpp"
