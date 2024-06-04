#pragma once

#ifndef __cplusplus
#error "The darknet project requires the use of a C++ compiler."
#endif

#include <atomic>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <deque>
#include <list>
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
#include <regex>
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

	/** This is used to help keep some state between calls to functions fill_network_boxes(), get_yolo_detections(), etc.
	 * We use the cache to track objects within the output array, so we don't have to walk over the entire array every
	 * time we need to find all the objects and bounding boxes.
	 */
	struct Output_Object
	{
		int layer_index;	///< The layer index where this was found.
		int n;				///< What is "n"...the mask number?
		int i;				///< The index into the float output array for the given layer.
		int obj_index;		///< The object index.
	};
	using Output_Object_Cache = std::list<Output_Object>;
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
#include "demo.hpp"
