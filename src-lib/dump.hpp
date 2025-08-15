/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#pragma once

#include "darknet_internal.hpp"


namespace Darknet
{
	/// @{ Dump some information on the given network and layers to a text file for debugging.
	void dump(Darknet::Network * net, const Darknet::CfgFile::CommonParms & parms);
	void dump(Darknet::CfgFile & cfg);
	/// @}

	/// Dump to @p std::cout the given block of memory.
	void dump(const float * ptr, const size_t count, const size_t row_len = 20);

	/// Dump the given layer's output buffer to @p std::cout.
	void dump(const Darknet::Layer & l);
}
