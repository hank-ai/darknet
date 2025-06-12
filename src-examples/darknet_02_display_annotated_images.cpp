/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#include "darknet.hpp"
#include "darknet_cfg_and_state.hpp"


/** @file
 * This application will display processed images in a GUI window.  Call it like this:
 *
 *     darknet_02_display_annotated_images LegoGears DSCN1580_frame_000034.jpg
 */


int main(int argc, char * argv[])
{
	try
	{
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);
#if 0
		Darknet::set_annotation_font(net, cv::LineTypes::LINE_4, cv::HersheyFonts::FONT_HERSHEY_PLAIN, 2, 2.00);
		Darknet::set_rounded_corner_bounding_boxes(net, true, 1.0f);
#endif

		int network_w = 0;
		int network_h = 0;
		int network_c = 0;
		Darknet::network_dimensions(net, network_w, network_h, network_c);

		for (const auto & parm : parms)
		{
			if (parm.type != Darknet::EParmType::kFilename)
			{
				continue;
			}

			std::cout << "processing " << parm.string << std::endl;

			const std::string title = "Darknet/YOLO - " + std::filesystem::path(parm.string).filename().string();

			const auto t1 = std::chrono::high_resolution_clock::now();
			cv::Mat mat = cv::imread(parm.string);
			const auto t2 = std::chrono::high_resolution_clock::now();
			Darknet::resize_keeping_aspect_ratio(mat, cv::Size(1024, 768));
			const auto t3 = std::chrono::high_resolution_clock::now();

			auto duration = (t3 - t1);

			cv::namedWindow("original", cv::WindowFlags::WINDOW_GUI_NORMAL);
			cv::setWindowTitle("original", title + " [original]");
			cv::resizeWindow("original", mat.size());
			cv::imshow("original", mat);

			// We could call the following:
			//
			//		Darknet::predict_and_annotate(net, mat);
			//
			// but to get the timing details of each function, we'll
			// call Darknet::predict() followed by Darknet::annotate().

			const auto t4 = std::chrono::high_resolution_clock::now();
			const auto results = Darknet::predict(net, mat);
			const auto t5 = std::chrono::high_resolution_clock::now();
			Darknet::annotate(net, results, mat);
			const auto t6 = std::chrono::high_resolution_clock::now();

			duration += (t6 - t4);

			cv::namedWindow("output", cv::WindowFlags::WINDOW_GUI_NORMAL);
			cv::setWindowTitle("output", title + " [annotated]");
			cv::resizeWindow("output", mat.size());
			cv::imshow("output", mat);

			std::cout
				<< "-> reading image from disk ........... " << Darknet::format_duration_string(t2 - t1, 3, Darknet::EFormatDuration::kPad) << " [" << mat.cols << " x " << mat.rows << " x " << mat.channels() << "]" << std::endl
				<< "-> resizing image to match network ... " << Darknet::format_duration_string(t3 - t2, 3, Darknet::EFormatDuration::kPad) << " [" << network_w << " x " << network_h << " x " << network_c << "]" << std::endl
				<< "-> using Darknet to predict .......... " << Darknet::format_duration_string(t5 - t4, 3, Darknet::EFormatDuration::kPad) << " [" << results.size() << " object" << (results.size() == 1 ? "" : "s") << "]" << std::endl
				<< "-> using Darknet to annotate image ... " << Darknet::format_duration_string(t6 - t5, 3, Darknet::EFormatDuration::kPad) << std::endl;

			if (Darknet::CfgAndState::get().is_set("heatmaps"))
			{
				const auto t7 = std::chrono::high_resolution_clock::now();
				const auto maps = Darknet::create_yolo_heatmaps(net);
				const auto t8 = std::chrono::high_resolution_clock::now();

				for (const auto & [k, v] : maps)
				{
					const std::string name = std::to_string(k);
					cv::namedWindow(name, cv::WindowFlags::WINDOW_GUI_NORMAL);
					cv::resizeWindow(name, v.size());
					cv::imshow(name, Darknet::visualize_heatmap(v));
				}
				const auto t9 = std::chrono::high_resolution_clock::now();

				duration += (t9 - t7);

				std::cout
					<< "-> create Darknet/YOLO heatmaps ...... " << Darknet::format_duration_string(t8 - t7, 3, Darknet::EFormatDuration::kPad) << " [" << maps.size() << "X]" << std::endl
					<< "-> visualize heatmaps ................ " << Darknet::format_duration_string(t9 - t8, 3, Darknet::EFormatDuration::kPad) << std::endl;
			}

			const int fps = std::round(1000000000.0f / std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
			std::cout << "-> total time elapsed ................ " << Darknet::format_duration_string(duration, 3, Darknet::EFormatDuration::kPad) << " [" << fps << " FPS]" << std::endl << std::endl;

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
