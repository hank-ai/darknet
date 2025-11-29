/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#include "darknet_internal.hpp"
#include "darknet_onnx.hpp"


/** @file
 * Convert Darknet/YOLO .cfg and .weights files to .onnx files.
 */


// Example command to dump the content of a .onnx file as plain text:
//
//			protoc --decode=onnx.ModelProto onnx.proto3 < LegoGears.onnx
//
// ...where the onnx.proto3 file is the one downloaded from the ONNX project.
// (See onnx.proto3.pb.txt for details.)


int main(int argc, char * argv[])
{
	TAT(TATPARMS);

	int rc = 1;

	try
	{
		// make sure the network loads correctly
		Darknet::Parms parms	= Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr ptr	= Darknet::load_neural_network(parms);
		const auto cfg_fn		= Darknet::get_config_filename(ptr);
		const auto weights_fn	= Darknet::get_weights_filename(ptr);
		const auto onnx_fn		= std::filesystem::path(cfg_fn).replace_extension(".onnx");
		Darknet::free_neural_network(ptr);

		// once we get here we know we have all the right parms, and the network successfully loads

		Darknet::ONNXExport onnx_export(cfg_fn, weights_fn, onnx_fn);
		onnx_export.load_network();
		onnx_export.initialize_model();
		onnx_export.build_model();
		onnx_export.display_summary();
		onnx_export.save_output_file();

		rc = 0;
	}
	catch (const std::exception & e)
	{
		rc = 2;
		std::cerr															<< std::endl
			<< "A fatal exception was detected:"							<< std::endl
			<< Darknet::in_colour(Darknet::EColour::kBrightRed, e.what())	<< std::endl;
	}

	return rc;
}
