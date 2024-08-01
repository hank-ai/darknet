#include "darknet.hpp"

/** @file
 * This application will print to @p STDOUT details on each object detected in an image.
 * Call it like this:
 *
 *     darknet_01_inference_images LegoGears DSCN1580_frame_000034.jpg
 *
 * The output should be similar to this:
 *
 *     predicting DSCN1580_frame_000034.jpg:
 *     prediction results: 5
 *     -> 1/5: #4 prob=0.999925 x=420 y=47 w=216 h=210 entries=1
 *     -> 2/5: #3 prob=0.999904 x=286 y=92 w=147 h=124 entries=1
 *     -> 3/5: #2 prob=0.991701 x=529 y=133 w=28 h=28 entries=1
 *     -> 4/5: #1 prob=0.998882 x=460 y=142 w=25 h=27 entries=1
 *     -> 5/5: #0 prob=0.995901 x=43 y=133 w=47 h=40 entries=1
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
