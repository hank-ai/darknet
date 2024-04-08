#pragma once

#ifndef __cplusplus
#error "The darknet project requires the use of a C++ compiler."
#endif

#include <atomic>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <ciso646>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <chrono>
#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>

namespace Darknet
{
	using MStr = std::map<std::string, std::string>;
	using MStrInt = std::map<std::string, int>;
	using SStr = std::set<std::string>;
	using VInt = std::vector<int>;
	using VStr = std::vector<std::string>;
	using VThreads = std::vector<std::thread>;
}

#include "darknet.h"
#include "darknet.hpp"
#include "darknet_version.h"

#include "darknet_args_and_parms.hpp"
#include "darknet_cfg_and_state.hpp"
#include "darknet_layers.hpp"
#include "darknet_format_and_colour.hpp"
#include "darknet_utils.hpp"
#include "Timing.hpp"

#include "box.hpp"
#include "blas.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "data.hpp"
