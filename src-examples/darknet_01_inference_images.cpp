/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#include "darknet.hpp"

/** @file
 * This application will print to @p STDOUT details on each object detected in an image.
 * Output images are saved to disk.  Call it like this:
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
		Darknet::show_version_info();

		Darknet::set_verbose(true);
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		for (const auto & parm : parms)
		{
			if (parm.type == Darknet::EParmType::kFilename)
			{
				const std::filesystem::path input_filename(parm.string);

				std::cout << "loading " << input_filename << std::endl;
				cv::Mat mat = cv::imread(input_filename);
				if (mat.empty())
				{
					std::cout << "...invalid image?" << std::endl;
					continue;
				}

				// output all of the predictions on the console as plain text
				const auto results = Darknet::predict(net, input_filename);
				std::cout << results << std::endl;

				// save the annotated image to disk
				cv::Mat output = Darknet::annotate(net, results, mat);
				std::string output_filename = input_filename.stem().string() + "_output";
#if 1
				output_filename += ".jpg";
				const bool successful = cv::imwrite(output_filename, output, {cv::ImwriteFlags::IMWRITE_JPEG_QUALITY, 70});
#else
				output_filename += ".png";
				const bool successful = cv::imwrite(output_filename, output, {cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION, 5});
#endif
				if (not successful)
				{
					std::cout << "failed to save the output to " << output_filename << std::endl;
				}
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
