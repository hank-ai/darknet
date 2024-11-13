/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#include "darknet.hpp"
#include "darknet_keypoints.hpp"
#include "darknet_cfg_and_state.hpp"


/** @file
 * This application uses a specific neural network that follows the MSCOCO-style keypoints to show "skeletons" over
 * people detected in images.  You must have the Darknet/YOLO Keypoints 18-class neural network for this to work.
 * (See the Darknet/YOLO readme for links to download the network config and weights.)
 *
 * The 18 classes are:
 *
 * @li 0:	nose
 * @li 1:	left eye
 * @li 2:	right eye
 * @li 3:	left ear
 * @li 4:	right ear
 * @li 5:	left shoulder
 * @li 6:	right shoulder
 * @li 7:	left elbow
 * @li 8:	right elbow
 * @li 9:	left wrist
 * @li 10:	right wrist
 * @li 11:	left hip
 * @li 12:	right hip
 * @li 13:	left knee
 * @li 14:	right knee
 * @li 15:	left ankle
 * @li 16:	right ankle
 * @li 17:	person
 *
 * Call it like this:
 *
 *     darknet_09_keypoints_images people_*.jpg
 *
 * If you have a specific network you want to load, call it like this:
 *
 *     darknet_09_keypoints_images mynetwork.cfg mynetwork.weights people_*.jpg
 */


int main(int argc, char * argv[])
{
	try
	{
		Darknet::show_version_info();

		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::set_default_keypoints_files(parms);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);
		Darknet::Keypoints keypoints(net);
		Darknet::set_trace(true); // this will force the skeleton details to display on STDOUT
		Darknet::set_detection_threshold(net, 0.2f);
		Darknet::set_annotation_line_type(net, cv::LineTypes::LINE_AA);

		cv::namedWindow("output", cv::WindowFlags::WINDOW_GUI_NORMAL);

		for (const auto & parm : parms)
		{
			if (parm.type != Darknet::EParmType::kFilename)
			{
				continue;
			}

			std::cout << "processing " << parm.string << std::endl;

			cv::Mat mat = cv::imread(parm.string);

			Darknet::resize_keeping_aspect_ratio(mat, cv::Size(1024, 768));

			const auto results = Darknet::predict(net, mat);
			const auto skeletons = keypoints.create_skeletons(results);

			/* We have both "results" and "skeletons".  The skeletons are vectors of 17 indexes into "results", where each
			 * entry is "nose", "eye", "ear", etc.  So at this point if you want to do something with the skeleton, you'll
			 * need to pass both "results" and "skeletons".
			 *
			 * For this example application, we'll call annotate() to get Darknet to draw the skeleton(s) for us.
			 */
			keypoints.annotate(results, skeletons, mat);

			if (Darknet::CfgAndState::get().is_set("heatmaps"))
			{
				auto maps = Darknet::create_yolo_heatmaps(net);
				auto & heatmap = maps[-1]; // only grab the heatmap with all the classes
				cv::namedWindow("heatmap", cv::WindowFlags::WINDOW_GUI_NORMAL);
				cv::resizeWindow("heatmap", heatmap.size());
				cv::imshow("heatmap", Darknet::visualize_heatmap(heatmap));
			}

			const std::string title = "Darknet/YOLO Keypoints & Skeletons - " + std::filesystem::path(parm.string).filename().string();
			cv::imshow("output", mat);
			cv::resizeWindow("output", mat.size());
			cv::setWindowTitle("output", title);

			const char c = cv::waitKey(-1);
			if (c == 27) // ESC
			{
				break;
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
