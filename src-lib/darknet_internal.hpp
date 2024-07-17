#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

// C headers
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

// C++ headers
#include <atomic>
#include <chrono>
#include <ciso646>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <vector>

// 3rd-party lib headers
#include <opencv2/opencv.hpp>

namespace Darknet
{
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

	class CfgLine;
	class CfgSection;
	class CfgFile;
}


#include "darknet.h"			// the old C header
#include "darknet.hpp"			// the new C++ header
#include "darknet_version.h"	// version macros

int yolo_num_detections_v3(network * net, const int index, const float thresh, Darknet::Output_Object_Cache & cache);
int get_yolo_detections_v3(network * net, int w, int h, int netw, int neth, float thresh, int *map, int relative, detection *dets, int letter, Darknet::Output_Object_Cache & cache);

#include "darknet_args_and_parms.hpp"
#include "darknet_cfg_and_state.hpp"
#include "darknet_enums.hpp"
#include "darknet_layers.hpp"
#include "darknet_format_and_colour.hpp"
#include "darknet_utils.hpp"
#include "Timing.hpp"
#include "darknet_cfg.hpp"
#include "box.hpp"
#include "blas.hpp"
#include "utils.hpp"
#include "parser.hpp"
#include "data.hpp"
#include "demo.hpp"
#include "network.hpp"
#include "option_list.hpp"
#include "classifier.hpp"
#include "image.hpp"
#include "dark_cuda.hpp"
#include "tree.hpp"
