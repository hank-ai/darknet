/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#include "darknet.hpp"

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
		Darknet::show_version_info();

		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

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

			const size_t video_width				= cap.get(cv::CAP_PROP_FRAME_WIDTH);
			const size_t video_height				= cap.get(cv::CAP_PROP_FRAME_HEIGHT);
			const size_t video_frames_count			= cap.get(cv::CAP_PROP_FRAME_COUNT);
			const double fps						= cap.get(cv::CAP_PROP_FPS);
			const size_t fps_rounded				= std::round(fps);
			const size_t nanoseconds_per_frame		= std::round(1000000000.0 / fps);
			const size_t video_length_milliseconds	= std::round(nanoseconds_per_frame / 1000000.0 * video_frames_count);
			const auto frame_duration				= std::chrono::nanoseconds(nanoseconds_per_frame);

			std::cout
				<< "-> frame dimensions ......... " << video_width << " x " << video_height		<< std::endl
				<< "-> frame count .............. " << video_frames_count						<< std::endl
				<< "-> frame rate ............... " << fps << " FPS"							<< std::endl
				<< "-> each frame lasts ......... " << nanoseconds_per_frame << " nanoseconds"	<< std::endl
				<< "-> estimated video length ... " << video_length_milliseconds << " milliseconds" << std::endl;

			const std::string title = "Darknet/YOLO - " + std::filesystem::path(parm.string).filename().string();
			cv::Mat mat(video_height, video_width, CV_8UC3, {0, 0, 0});
			cv::imshow("output", mat);
			cv::resizeWindow("output", mat.size());
			cv::setWindowTitle("output", title);

			size_t frame_counter = 0;
			size_t total_objects_found = 0;
			double total_sleep_in_milliseconds = 0.0;
			const auto timestamp_when_video_started = std::chrono::high_resolution_clock::now();
			auto timestamp_next_frame = timestamp_when_video_started + frame_duration;

			while (true)
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
				if (milliseconds >= 1)
				{
					total_sleep_in_milliseconds	+= milliseconds;
					const char c = cv::waitKey(milliseconds);
					if (c == 27) // ESC
					{
						escape_detected = true;
						std::cout << std::endl << "ESC!" << std::endl;
						break;
					}
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
				<< "-> processed frame rate ..... " << final_fps << " FPS"												<< std::endl
				<< "-> total objects founds ..... " << total_objects_found												<< std::endl
				<< "-> average objects/frame .... " << static_cast<float>(total_objects_found) / frame_counter			<< std::endl;
		}

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
