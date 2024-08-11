#pragma once

#include "darknet_internal.hpp"


namespace Darknet
{
	/// @{ Dump some information on the given network and layers to a text file for debugging.
	void dump(Darknet::Network * net, const Darknet::CfgFile::CommonParms & parms);
	void dump(Darknet::CfgFile & cfg);
	/// @}
}
