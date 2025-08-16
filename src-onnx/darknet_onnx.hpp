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

			/// Callback function that Protocol Buffers calls to log messages.
			static void log_handler(google::protobuf::LogLevel level, const char * filename, int line, const std::string & message);

			/// Use Darknet to load the neural network.
			ONNXExport & load_network();

			/// Display some general information about the protocol buffer model.
			ONNXExport & display_summary();

			/// Initialize some of the simple protobuffer model fields.
			ONNXExport & initialize_model();

			/// Insert a standard doc string indicating which line in the .cfg corresponds to this section
			ONNXExport & set_doc_string(onnx::ValueInfoProto * proto, const size_t line_number);

			ONNXExport & populate_input_output_dimensions(onnx::ValueInfoProto * proto, const std::string & name, const int v1, const int v2=-1, const int v3=-1, const int v4=-1, const size_t line_number=0);
			ONNXExport & populate_graph_input_000_net();
			ONNXExport & populate_graph_input();
			ONNXExport & populate_graph_output();
			ONNXExport & populate_graph_nodes();
			ONNXExport & populate_graph_initializer(const float * f, const size_t n, const size_t idx, const Darknet::Layer & l, const std::string & name);
			ONNXExport & populate_graph_initializers();
			ONNXExport & build_model();

			/// Save the entire model as an .onnx file.
			ONNXExport & save_output_file();

			std::filesystem::path cfg_fn;
			std::filesystem::path weights_fn;
			std::filesystem::path onnx_fn;

			Darknet::CfgFile cfg;

			onnx::ModelProto	model;
			onnx::GraphProto	* graph;

			/** The key is the last part of the string, and the value is the number of floats.
			 * For example, for "000_conv_bias", we store the key as "bias".
			 */
			std::map<std::string, size_t> number_of_floats_exported;
	};
}
