#pragma once

#ifndef __cplusplus
#error "The darknet project requires the use of a C++ compiler."
#endif

#if __cplusplus < 201703L
#error "The darknet project requires C++17 or newer."
#endif

#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <ciso646>

namespace Darknet
{
	using MStr = std::map<std::string, std::string>;
	using SStr = std::set<std::string>;
	using VStr = std::vector<std::string>;
}

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <opencv2/opencv.hpp>

#include "darknet.h"

#include "box.hpp"
#include "blas.hpp"
#include "utils.hpp"

#include "darknet_version.h"
#include "darknet_layers.hpp"
#include "darknet_format_and_colour.hpp"
#include "darknet_utils.hpp"
#include "darknet_args_and_parms.hpp"
#include "darknet_cfg_and_state.hpp"
