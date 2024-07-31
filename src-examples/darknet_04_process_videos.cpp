#include "darknet.hpp"

/** @file
 * This application will process one or more videos as fast as possible.  The results are not immediately shown to the
 * user, but the output video is saved to disk for review.  Call it like this:
 *
 *     darknet_04_process_videos LegoGears DSCN1582A.MOV
 */


int main(int argc, char * argv[])
{
	try
	{
		std::cout << "Darknet v" << DARKNET_VERSION_SHORT << std::endl;

		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		int network_width = 0;
		int network_height = 0;
		int network_channels = 0;
		Darknet::network_dimensions(net, network_width, network_height, network_channels);

		for (const auto & parm : parms)
		{
			if (parm.type == Darknet::EParmType::kFilename)
			{
				std::cout << "processing " << parm.string << ":" << std::endl;

				cv::VideoCapture cap(parm.string);
				if (not cap.isOpened())
				{
					std::cout << "Failed to open the input video file " << parm.string << std::endl;
					continue;
				}

				const std::string output_filename		= std::filesystem::path(parm.string).stem().string() + "_output.m4v";
				const size_t video_width				= cap.get(cv::CAP_PROP_FRAME_WIDTH);
				const size_t video_height				= cap.get(cv::CAP_PROP_FRAME_HEIGHT);
				const size_t video_frames_count			= cap.get(cv::CAP_PROP_FRAME_COUNT);
				const double fps						= cap.get(cv::CAP_PROP_FPS);
				const size_t frame_in_nanoseconds		= std::round(1000000000.0 / fps);
				const size_t video_length_milliseconds	= std::round(frame_in_nanoseconds / 1000000.0 * video_frames_count);

				std::cout
					<< "-> neural network size ...... " << network_width << " x " << network_height << " x " << network_channels << std::endl
					<< "-> input video dimensions ... " << video_width << " x " << video_height			<< std::endl
					<< "-> input video frame count .. " << video_frames_count							<< std::endl
					<< "-> input video frame rate ... " << fps << " FPS"								<< std::endl
					<< "-> input video length ....... " << video_length_milliseconds << " milliseconds"	<< std::endl
					<< "-> output filename .......... " << output_filename								<< std::endl;

				cv::VideoWriter out(output_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(video_width, video_height));
				if (not out.isOpened())
				{
					std::cout << "Failed to open the output video file " << output_filename << std::endl;
					continue;
				}

				size_t frame_counter = 0;
				size_t total_objects_found = 0;
				const auto timestamp_when_video_started = std::chrono::high_resolution_clock::now();

				while (true)
				{
					cv::Mat mat;
					cap >> mat;
					if (mat.empty())
					{
						break;
					}

					const auto results = Darknet::predict_and_annotate(net, mat);
					out.write(mat);
					frame_counter ++;
					total_objects_found += results.size();
				}

				const auto timestamp_when_video_ended = std::chrono::high_resolution_clock::now();
				const auto processing_duration = timestamp_when_video_ended - timestamp_when_video_started;
				const size_t processing_time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(processing_duration).count();
				const double final_fps = 1000.0 * frame_counter / processing_time_in_milliseconds;

				std::cout
					<< "-> total frames processed ... " << frame_counter											<< std::endl
					<< "-> time to process video .... " << processing_time_in_milliseconds << " milliseconds"		<< std::endl
					<< "-> final frame rate ......... " << final_fps << " FPS"										<< std::endl
					<< "-> total objects founds ..... " << total_objects_found										<< std::endl
					<< "-> average objects/frame .... " << static_cast<float>(total_objects_found) / frame_counter	<< std::endl;
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
