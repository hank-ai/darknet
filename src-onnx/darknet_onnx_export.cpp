#include "onnx.pb.h"
#include "darknet_internal.hpp"


int main(int argc, char * argv[])
{
	try
	{
		Darknet::Parms parms	= Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net	= Darknet::load_neural_network(parms);
		const auto cfg_fn		= Darknet::get_config_filename(net);
		const auto weights_fn	= Darknet::get_weights_filename(net);
		const auto onnx_fn		= std::filesystem::path(cfg_fn).replace_extension(".onnx");

		std::cout << std::endl << "Darknet/YOLO ONNX Export"	<< std::endl
			<< "-> configuration ... " << cfg_fn	.string()	<< std::endl
			<< "-> weights ......... " << weights_fn.string()	<< std::endl
			<< "-> onnx output ..... " << onnx_fn	.string()	<< std::endl
			;

		if (std::filesystem::exists(onnx_fn))
		{
			bool success = std::filesystem::remove(onnx_fn);
			if (not success)
			{
				throw std::runtime_error("failed to delete existing file " + onnx_fn.string());
			}
		}

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
