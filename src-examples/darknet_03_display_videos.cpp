/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2026 Stephane Charette
 */

#include "darknet.hpp"
#include "darknet_cfg_and_state.hpp"


/** @file
 * This application will display one or more videos at the frame rate stored in the video file.  Each individual video
 * frame is processed by the given neural network.  The output video is displayed on the screen, and @em not saved to
 * disk.  Call it like this:
 *
 *     darknet_03_display_videos LegoGears DSCN1582A.MOV
 */


int main(int argc, char * argv[])
{
	try
	{
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		int network_width = 0;
		int network_height = 0;
		int network_channels = 0;
		Darknet::network_dimensions(net, network_width, network_height, network_channels);

		const bool show_heatmaps = Darknet::CfgAndState::get().is_set("heatmaps");

		bool escape_detected = false;
		for (const auto & parm : parms)
		{
			if (escape_detected)
			{
				break;
			}

			if (parm.type != Darknet::EParmType::kFilename)
			{
				continue;
			}

			std::cout << "processing " << parm.string << ":" << std::endl;

			cv::VideoCapture cap(parm.string);
			if (not cap.isOpened())
			{
				std::cout << "Failed to open the video file " << parm.string << std::endl;
				continue;
			}

			cv::Mat mat;
			cap >> mat;
			cap.set(cv::CAP_PROP_POS_FRAMES, 0.0);

			const size_t video_width				= mat.cols;
			const size_t video_height				= mat.rows;
			const size_t video_channels				= mat.channels();
			const size_t video_frames_count			= cap.get(cv::CAP_PROP_FRAME_COUNT);
			const double fps						= cap.get(cv::CAP_PROP_FPS);
			const size_t fps_rounded				= std::round(fps);
			const size_t nanoseconds_per_frame		= std::round(1000000000.0 / fps);
			const size_t video_length_milliseconds	= std::round(nanoseconds_per_frame / 1000000.0 * video_frames_count);
			const auto frame_duration				= std::chrono::nanoseconds(nanoseconds_per_frame);

			std::cout
				<< "-> neural network size ...... " << network_width	<< " x " << network_height	<< " x " << network_channels	<< std::endl
				<< "-> video dimensions ......... " << video_width		<< " x " << video_height	<< " x " << video_channels		<< std::endl
				<< "-> frame count .............. " << video_frames_count						<< std::endl
				<< "-> frame rate ............... " << fps << " FPS"							<< std::endl
				<< "-> each frame lasts ......... " << nanoseconds_per_frame << " nanoseconds"	<< std::endl
				<< "-> estimated video length ... " << Darknet::format_duration_string(std::chrono::milliseconds(video_length_milliseconds)) << std::endl;

			if (show_heatmaps)
			{
				const auto maps = Darknet::create_yolo_heatmaps(net);
				for (const auto & [k, v] : maps)
				{
					const std::string name = std::to_string(k);
					cv::namedWindow(name, cv::WindowFlags::WINDOW_GUI_NORMAL);
					cv::resizeWindow(name, v.size());
					cv::imshow(name, v);
				}
			}

			const std::string title = "Darknet/YOLO - " + std::filesystem::path(parm.string).filename().string();
			mat = cv::Mat(video_height, video_width, CV_8UC3, {0, 0, 0});
			cv::namedWindow("output", cv::WindowFlags::WINDOW_GUI_NORMAL);
			cv::setWindowTitle("output", title);
			cv::resizeWindow("output", mat.size());
			cv::imshow("output", mat);
			cv::waitKey(5);

			size_t frame_counter		= 0;
			size_t fell_behind			= 0;
			size_t total_objects_found	= 0;
			auto total_sleep_duration	= std::chrono::high_resolution_clock::duration();

			const auto timestamp_when_video_started = std::chrono::high_resolution_clock::now();
			auto timestamp_next_frame = timestamp_when_video_started + frame_duration;

			while (cap.isOpened())
			{
				cap >> mat;
				if (mat.empty())
				{
					break;
				}

				const auto results = Darknet::predict_and_annotate(net, mat);
				cv::imshow("output", mat);
				frame_counter ++;
				total_objects_found += results.size();

				if (show_heatmaps)
				{
					const auto maps = Darknet::create_yolo_heatmaps(net);
					for (const auto & [k, v] : maps)
					{
						cv::imshow(std::to_string(k), Darknet::visualize_heatmap(v));
					}
				}

				if (frame_counter % fps_rounded == 0)
				{
					const int percentage = std::round(100.0f * frame_counter / video_frames_count);
					std::cout
						<< "-> frame #" << frame_counter << "/" << video_frames_count
						<< " (" << percentage << "%)\r"
						<< std::flush;
				}

				// see how much time we should sleep based on the length of time between each frame
				const auto now				= std::chrono::high_resolution_clock::now();
				const auto time_remaining	= timestamp_next_frame - now;
				const int milliseconds		= std::chrono::duration_cast<std::chrono::milliseconds>(time_remaining).count();
				if (milliseconds >= 0)
				{
					total_sleep_duration += time_remaining;
				}
				else if (frame_counter > 1) // not unusual for the very first frame to fall behind
				{
					fell_behind ++;
					if (fell_behind == 10)
					{
						std::cout << "WARNING: cannot maintain " << fps << " FPS" << std::endl;
					}
				}

				const char c = cv::waitKey(std::max(5, milliseconds));
				if (c == 27) // ESC
				{
					escape_detected = true;
					std::cout << std::endl << "ESC!" << std::endl;
					break;
				}
				timestamp_next_frame += frame_duration;
			}

			const auto timestamp_when_video_ended = std::chrono::high_resolution_clock::now();
			const auto video_duration = timestamp_when_video_ended - timestamp_when_video_started;
			const size_t video_length_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(video_duration).count();
			const double final_fps = 1000.0 * frame_counter / video_length_in_milliseconds;
			const double total_sleep_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(total_sleep_duration).count();
			const size_t average_sleep_nanoseconds = std::round(total_sleep_nanoseconds / frame_counter);

			std::cout
				<< "-> number of frames shown ... " << frame_counter													<< std::endl
				<< "-> total sleep time ......... " << Darknet::format_duration_string(total_sleep_duration)			<< std::endl
				<< "-> average sleep per frame .. " << Darknet::format_duration_string(std::chrono::nanoseconds(average_sleep_nanoseconds)) << std::endl
				<< "-> video display time ....... " << Darknet::format_duration_string(video_duration)					<< std::endl
				<< "-> processed frame rate ..... " << final_fps << " FPS"												<< std::endl;
			if (fell_behind)
			{
				std::cout << "-> failed to maintain FPS ... " << fell_behind << " frame" << (fell_behind == 1 ? "" : "s") << std::endl;
			}
			std::cout
				<< "-> total objects found ...... " << total_objects_found												<< std::endl
				<< "-> average objects/frame .... " << static_cast<float>(total_objects_found) / frame_counter			<< std::endl;
		}

		cv::destroyAllWindows();

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
