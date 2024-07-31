#include "darknet.hpp"

/** @file
 * This application will print to @p STDOUT details on each object detected in an image.
 * Call it like this:
 *
 *     darknet_01_inference_images LegoGears DSCN1580_frame_000034.jpg
 */


int main(int argc, char * argv[])
{
	try
	{
		std::cout << "Darknet v" << DARKNET_VERSION_SHORT << std::endl;

		Darknet::set_verbose(true);
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		for (const auto & parm : parms)
		{
			if (parm.type == Darknet::EParmType::kFilename)
			{
				std::cout << "predicting " << parm.string << ":" << std::endl;
				const auto results = Darknet::predict(net, parm.string);

				// output all of the predictions on the console as plain text
				std::cout << results << std::endl;
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
