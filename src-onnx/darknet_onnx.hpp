/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */
#pragma once

#ifndef __cplusplus
#error "The Darknet/YOLO project requires a C++ compiler."
#endif

/** @file
 * Class used to convert the Darknet .cfg and .weights file to a .onnx file.
 */

#include "darknet_internal.hpp"


namespace Darknet
{
	/// Everthing we need to convert .cfg and .weights to .onnx is contained within this class.
	class ONNXExport final
	{
		public:

			/// Constructor.
			ONNXExport(const std::filesystem::path & cfg_filename, const std::filesystem::path & weights_filename, const std::filesystem::path & onnx_filename);

			/// Destructor.
			~ONNXExport();

			/// Use Darknet to load the neural network.
			ONNXExport & load_network();

			/// Display some general information about the protocol buffer model.
			ONNXExport & display_summary();

			/// Initialize some of the simple protobuffer model fields.
			ONNXExport & initialize_model();

			ONNXExport & populate_input_output_dimensions(onnx::ValueInfoProto * proto, const std::string & name, const int v1, const int v2=-1, const int v3=-1, const int v4=-1, const size_t line_number=0);
			ONNXExport & populate_graph_input_frame();
			ONNXExport & populate_graph_YOLO_output();
			ONNXExport & populate_graph_nodes();
			ONNXExport & populate_graph_postprocess_boxes();

			ONNXExport & add_node_conv			(const size_t index, Darknet::CfgSection & section);
			ONNXExport & add_node_shortcut		(const size_t index, Darknet::CfgSection & section);
			ONNXExport & check_activation		(const size_t index, Darknet::CfgSection & section);
			ONNXExport & add_node_activation	(const size_t index, Darknet::CfgSection & section);
			ONNXExport & add_node_route_identity(const size_t index, Darknet::CfgSection & section);
			ONNXExport & add_node_route_slice	(const size_t index, Darknet::CfgSection & section);
			ONNXExport & add_node_route_concat	(const size_t index, Darknet::CfgSection & section);
			ONNXExport & add_node_maxpool		(const size_t index, Darknet::CfgSection & section);
			ONNXExport & add_node_yolo			(const size_t index, Darknet::CfgSection & section);
			ONNXExport & add_node_resize		(const size_t index, Darknet::CfgSection & section);
			ONNXExport & add_node_bn			(const size_t index, Darknet::CfgSection & section);

			ONNXExport & populate_graph_initializer(const float * f, const size_t n, const size_t idx, const Darknet::Layer & l, const std::string & name, const bool simple = false);
			ONNXExport & build_model();

			// post-processing boxes

			std::string add_const_float_tensor	(const std::string & stem, const float & f);
			std::string add_const_ints_tensor	(const std::string & stem, const std::vector<int> & v);
			ONNXExport & postprocess_yolo_split_and_concat(const size_t index, Darknet::CfgSection & section);
			ONNXExport & postprocess_yolo_tx_ty	(const size_t index, Darknet::CfgSection & section);
			ONNXExport & postprocess_yolo_tw_th	(const size_t index, Darknet::CfgSection & section);
			ONNXExport & postprocess_yolo_to	(const size_t index, Darknet::CfgSection & section);

			/// Save the entire model as an .onnx file.
			ONNXExport & save_output_file();

			std::filesystem::path cfg_fn;
			std::filesystem::path weights_fn;
			std::filesystem::path onnx_fn;

			Darknet::CfgFile cfg;

			/// Which opset version to use (10, 18, ...)?
			int opset_version;

			onnx::ModelProto	model;
			onnx::GraphProto	* graph;

			/// Whether or not we need to fuse batchnorm (`fuse` and `dontfuse` on the CLI).
			bool fuse_batchnorm;

			/// Whether or not we need to output the post-processing nodes to deal with boxes.
			bool postprocess_boxes;

			/// The dimensions used in @ref populate_graph_input_frame().
			std::string input_string;

			/// The names of the output nodes for this neural network.
			std::string output_string;

			/// Keep track of the single most recent output name for each of the layers.
			std::map<int, std::string> most_recent_output_per_index;

			/** The key is the last part of the string, and the value is the number of floats.
			 * For example, for "000_conv_bias", we store the key as "bias".
			 */
			std::map<std::string, size_t> number_of_floats_exported;
	};
}
