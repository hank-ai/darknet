/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#include "darknet.hpp"
#include "darknet_keypoints.hpp"
#include "darknet_cfg_and_state.hpp"


/** @file
 * This application combines the functionality of darknet_03_display_videos and darknet_09_keypoints_images.
 * Videos with keypoints and skeletons will be shown in realtime.  Call it like this:
 *
 *     darknet_10_keypoints_videos example.avi
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
		Darknet::set_trace(false);
		Darknet::set_detection_threshold(net, 0.2f);
		Darknet::set_annotation_line_type(net, cv::LineTypes::LINE_AA);

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

			const std::string output_filename		= std::filesystem::path(parm.string).stem().string() + "_keypoints.m4v";
			const size_t video_width				= cap.get(cv::CAP_PROP_FRAME_WIDTH);
			const size_t video_height				= cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			const size_t video_frames_count			= cap.get(cv::CAP_PROP_FRAME_COUNT);
			const double fps						= cap.get(cv::CAP_PROP_FPS);
			const size_t show_stats_frequency		= std::round(fps * 1.5); // stats will be shown about every 1.5 seconds
			const size_t nanoseconds_per_frame		= std::round(1000000000.0 / fps);
			const size_t video_length_milliseconds	= std::round(nanoseconds_per_frame / 1000000.0 * video_frames_count);
			const auto frame_duration				= std::chrono::nanoseconds(nanoseconds_per_frame);

			std::cout
				<< std::fixed << std::setprecision(3)
				<< "-> frame dimensions ......... " << video_width << " x " << video_height			<< std::endl
				<< "-> frame count .............. " << video_frames_count							<< std::endl
				<< "-> frame rate ............... " << fps << " FPS"								<< std::endl
				<< "-> each frame lasts ......... " << nanoseconds_per_frame << " nanoseconds"		<< std::endl
				<< "-> input video length ....... " << video_length_milliseconds << " milliseconds"	<< std::endl
				<< "-> output filename .......... " << output_filename								<< std::endl;

			cv::VideoWriter out(output_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(video_width, video_height));
			if (not out.isOpened())
			{
				std::cout << "Failed to open the output video file " << output_filename << std::endl;
				continue;
			}

			if (Darknet::CfgAndState::get().is_set("heatmaps"))
			{
				auto maps = Darknet::create_yolo_heatmaps(net);
				auto & mat = maps[-1]; // only grab the heatmap with all the classes
				cv::namedWindow("heatmap", cv::WindowFlags::WINDOW_GUI_NORMAL);
				cv::resizeWindow("heatmap", mat.size());
				cv::imshow("heatmap", Darknet::visualize_heatmap(mat));
			}

			const std::string title = "Darknet/YOLO - " + std::filesystem::path(parm.string).filename().string();
			cv::Mat mat(video_height, video_width, CV_8UC3, {0, 0, 0});
			cv::namedWindow("output", cv::WindowFlags::WINDOW_GUI_NORMAL);
			cv::resizeWindow("output", mat.size());
			cv::setWindowTitle("output", title);
			cv::imshow("output", mat);
			cv::waitKey(5);

			size_t fell_behind = 0;
			size_t frame_counter = 0;
			size_t recent_frame_counter	= 0;
			double total_sleep_in_milliseconds = 0.0;
			const auto timestamp_when_video_started = std::chrono::high_resolution_clock::now();
			auto timestamp_recent = timestamp_when_video_started;
			auto timestamp_next_frame = timestamp_when_video_started + frame_duration;

			while (cap.isOpened())
			{
				cap >> mat;
				if (mat.empty())
				{
					break;
				}

				const auto results = Darknet::predict(net, mat);
				const auto skeletons = keypoints.create_skeletons(results);
				keypoints.annotate(results, skeletons, mat);

				out.write(mat);
				cv::imshow("output", mat);
				frame_counter ++;

				if (Darknet::CfgAndState::get().is_set("heatmaps"))
				{
					auto maps = Darknet::create_yolo_heatmaps(net);
					auto & heatmap = maps[-1]; // only grab the heatmap with all the classes
					cv::imshow("heatmap", Darknet::visualize_heatmap(heatmap));
				}

				if (frame_counter == video_frames_count or frame_counter % show_stats_frequency == 0)
				{
					const int percentage = std::round(100.0f * frame_counter / video_frames_count);
					const auto now = std::chrono::high_resolution_clock::now();
					const double nanoseconds_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - timestamp_recent).count();
					const double recent_fps = (frame_counter - recent_frame_counter) / nanoseconds_elapsed * 1000000000.0;

					std::cout
						<< "\r-> frame #" << frame_counter << "/" << video_frames_count
						<< " (" << percentage << "%)"
						<< ", " << recent_fps << " FPS "
						<< std::flush;

					timestamp_recent = now;
					recent_frame_counter = frame_counter;
				}

				// see how much time we should sleep based on the length of time between each frame
				const auto now				= std::chrono::high_resolution_clock::now();
				const auto time_remaining	= timestamp_next_frame - now;
				const int milliseconds		= std::chrono::duration_cast<std::chrono::milliseconds>(time_remaining).count();
				if (milliseconds >= 0)
				{
					total_sleep_in_milliseconds	+= milliseconds;
				}
				else if (frame_counter > 1) // not unusual for the very first frame to fall behind
				{
					fell_behind ++;
					if (fell_behind == 10)
					{
						std::cout << std::endl << "WARNING: cannot maintain " << fps << " FPS" << std::endl;
					}
				}

				const char c = cv::waitKey(std::max(5, milliseconds));
				if (c == 27) // ESC
				{
					escape_detected = true;
					std::cout << std::endl << "ESC!";
					break;
				}
				timestamp_next_frame += frame_duration;
			}

			const auto timestamp_when_video_ended = std::chrono::high_resolution_clock::now();
			const auto video_duration = timestamp_when_video_ended - timestamp_when_video_started;
			const size_t video_length_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(video_duration).count();
			const double final_fps = 1000.0 * frame_counter / video_length_in_milliseconds;

			std::cout
				<< "-> number of frames shown ... " << frame_counter													<< std::endl
				<< "-> average sleep per frame .. " << total_sleep_in_milliseconds / frame_counter << " milliseconds"	<< std::endl
				<< "-> total length of video .... " << video_length_in_milliseconds << " milliseconds"					<< std::endl
				<< "-> processed frame rate ..... " << final_fps << " FPS"												<< std::endl;
			if (fell_behind)
			{
				std::cout << "-> failed to maintain FPS ... " << fell_behind << " frame" << (fell_behind == 1 ? "" : "s") << std::endl;
			}
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
