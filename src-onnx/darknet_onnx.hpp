/* Darknet/YOLO:  https://github.com/hank-ai/darknet
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

			/// Save the entire model as an .onnx file.
			ONNXExport & save_output_file();

			std::filesystem::path cfg_fn;
			std::filesystem::path weights_fn;
			std::filesystem::path onnx_fn;

			Darknet::CfgFile cfg;

			onnx::ModelProto model;
	};
}
