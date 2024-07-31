#include "darknet.hpp"


int main(int argc, char * argv[])
{
	try
	{
		std::cout << "Darknet v" << DARKNET_VERSION_SHORT << std::endl;

		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		for (const auto & parm : parms)
		{
			if (parm.type == Darknet::EParmType::kFilename)
			{
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
				const size_t frame_in_nanoseconds		= std::round(1000000000.0 / fps);
				const size_t video_length_milliseconds	= std::round(frame_in_nanoseconds / 1000000.0 * video_frames_count);
				const auto frame_duration				= std::chrono::nanoseconds(frame_in_nanoseconds);
				std::cout
					<< "-> frame dimensions ......... " << video_width << " x " << video_height		<< std::endl
					<< "-> frame count .............. " << video_frames_count						<< std::endl
					<< "-> frame rate ............... " << fps << " FPS"							<< std::endl
					<< "-> each frame lasts ......... " << frame_in_nanoseconds << " nanoseconds"	<< std::endl
					<< "-> estimated video length ... " << video_length_milliseconds << " milliseconds" << std::endl;

				size_t frame_counter = 0;
				double total_sleep_in_milliseconds = 0.0;
				const auto timestamp_when_video_started = std::chrono::high_resolution_clock::now();
				auto timestamp_next_frame = timestamp_when_video_started + frame_duration;

				while (true)
				{
					cv::Mat mat;
					cap >> mat;
					if (mat.empty())
					{
						break;
					}

					Darknet::predict_and_annotate(net, mat);
					cv::imshow("annotated", mat);
					frame_counter ++;

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
							std::cout << "ESC!" << std::endl;
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
					<< "-> final frame rate ......... " << final_fps << " FPS"												<< std::endl;
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
