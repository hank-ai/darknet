/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */
#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

/** @file
 * Class used to create and manipulate ONNX nodes.
 */

#include "darknet_internal.hpp"


namespace Darknet
{
	/// Class used to create and manipulate ONNX nodes.
	class Node final
	{
		public:

			/// Constructor.
			Node(const std::string & n);

			/// Constructor.
			Node(const Darknet::CfgSection & section, const std::string & append = "");

			/// Constructor for a single @p float constant.
			Node(const Darknet::CfgSection & section, const float f);

			/// Constructor for many @p int constants.
			Node(const Darknet::CfgSection & section, const Darknet::VInt & v);

			/// Destructor.
			~Node();

			Node & init(const std::string & n);
			Node & doc(const std::string & str);
			Node & doc(const Darknet::CfgSection & section);
			Node & doc_append(const std::string & str);
			Node & type(const std::string & type);

			/// Look up the node's name from the given index and use that as input.
			Node & add_input(int idx);

			Node & add_input(const std::string & input);

			Node & set_output(const std::string & out = "");

			Node & add_attribute_INT(const std::string & key, const int val);
			Node & add_attribute_INTS(const std::string & key, const Darknet::VInt & val);
			Node & add_attribute_STR(const std::string & key, const std::string & val);
			Node & add_attribute_FLOAT(const std::string & key, const float val);

			onnx::NodeProto * node;
			size_t layer_index;
			size_t counter;
			std::string name;
			std::string output;

			static onnx::GraphProto * graph;
			static std::string cfg_filename;
			static size_t node_counter;

			/// Keep track of the most recent output name for each of the layers.
			static Darknet::MIdStr output_per_layer_index;
			static std::string get_output_for_layer_index(const int idx);
	};
}
