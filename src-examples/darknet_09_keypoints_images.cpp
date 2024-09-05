/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#include "darknet.hpp"
#include "darknet_keypoints.hpp"

/** @file
 * This application will use the MSCOCO keypoints neural network to display images in a GUI window.  You must have the
 * MSCOCO Keypoints 17-class neural network for this to work.  Call it like this:
 *
 *     darknet_09_keypoints_images people_*.jpg
 *
 * If you have a specific network you want to load, call it like this:
 *
 *     darknet_09_keypoints_images mynetwork.cfg mynetwork.weights people_*.jpg
 */


void set_default_neural_network_files(Darknet::Parms & parms)
{
	// if we know which neural network to load, then there is nothing for us to change,
	// but if the user didn't specify files to load then we'll default to the MSCOCO "keypoints"

	bool found_cfg		= false;
	bool found_weights	= false;
	for (const auto & parm : parms)
	{
		if (parm.type == Darknet::EParmType::kCfgFilename)
		{
			found_cfg = true;
		}
		else if (parm.type == Darknet::EParmType::kWeightsFilename)
		{
			found_weights = true;
		}
	}

	if (not found_cfg)
	{
		Darknet::Parm parm;
		parm.idx		= 0;
		parm.type		= Darknet::EParmType::kCfgFilename;
		parm.original	= "mscoco-keypoints.cfg";
		parm.string		= parm.original;
		parms.push_back(parm);
	}

	if (not found_weights)
	{
		Darknet::Parm parm;
		parm.idx		= 0;
		parm.type		= Darknet::EParmType::kWeightsFilename;
		parm.original	= "mscoco-keypoints.weights";
		parm.string		= parm.original;
		parms.push_back(parm);
	}

	return;
}


int main(int argc, char * argv[])
{
	try
	{
		Darknet::show_version_info();

		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		set_default_neural_network_files(parms);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);
		Darknet::Keypoints keypoints(net);
		Darknet::set_trace(true); // this will force the skeleton details to display on STDOUT
		Darknet::set_detection_threshold(net, 0.2f);
		Darknet::set_annotation_line_type(net, cv::LineTypes::LINE_AA);

		for (const auto & parm : parms)
		{
			if (parm.type == Darknet::EParmType::kFilename)
			{
				std::cout << "processing " << parm.string << std::endl;

				cv::Mat mat = cv::imread(parm.string);

				const auto results = Darknet::predict(net, mat);
				const auto skeletons = keypoints.create_skeletons(results);

				/* We have both "results" and "skeletons".  The skeletons are vectors of 17 indexes into "results", where each
				 * entry is "nose", "eye", "ear", etc.  So at this point if you want to do something with the skeleton, you
				 * need to do something with *both* "results" and "skeletons".
				 *
				 * For this example application, we'll call annotate() to get Darknet to draw the skeleton(s) for us.
				 */
				keypoints.annotate(results, skeletons, mat);
				cv::imshow("annotated", mat);

				const char c = cv::waitKey(-1);
				if (c == 27) // ESC
				{
					break;
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
