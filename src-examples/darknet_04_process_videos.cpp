/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#include "darknet.hpp"

/** @file
 * This application will process one or more videos as fast as possible on a single thread and save a new output video
 * to disk.  The results are not shown to the user.  Call it like this:
 *
 *     darknet_04_process_videos LegoGears DSCN1582A.MOV
 *
 * The output should be similar to this:
 *
 *     processing DSCN1582A.MOV:
 *     -> neural network size ...... 224 x 160 x 3
 *     -> input video dimensions ... 640 x 480
 *     -> input video frame count .. 1230
 *     -> input video frame rate ... 29.970030 FPS
 *     -> input video length ....... 41041 milliseconds
 *     -> output filename .......... DSCN1582A_output.m4v
 *     -> total frames processed ... 1230
 *     -> time to process video .... 3207 milliseconds
 *     -> processed frame rate ..... 383.536015 FPS
 *     -> total objects found ...... 6189
 *     -> average objects/frame .... 5.031707
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

				cv::Mat mat;
				cap >> mat;
				cap.set(cv::CAP_PROP_POS_FRAMES, 0.0);

				const std::string output_filename		= std::filesystem::path(parm.string).stem().string() + "_output.m4v";
				const size_t video_width				= mat.cols;
				const size_t video_height				= mat.rows;
				const size_t video_channels				= mat.channels();
				const size_t video_frames_count			= cap.get(cv::CAP_PROP_FRAME_COUNT);
				const double fps						= cap.get(cv::CAP_PROP_FPS);
				const size_t fps_rounded				= std::round(fps);
				const size_t nanoseconds_per_frame		= std::round(1000000000.0 / fps);
				const size_t video_length_milliseconds	= std::round(nanoseconds_per_frame / 1000000.0 * video_frames_count);

				std::cout
					<< "-> neural network size ...... " << network_width	<< " x " << network_height	<< " x " << network_channels	<< std::endl
					<< "-> input video dimensions ... " << video_width		<< " x " << video_height	<< " x " << video_channels		<< std::endl
					<< "-> input video frame count .. " << video_frames_count							<< std::endl
					<< "-> input video frame rate ... " << fps << " FPS"								<< std::endl
					<< "-> input video length ....... " << Darknet::format_duration_string(std::chrono::milliseconds(video_length_milliseconds)) << std::endl
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
					cap >> mat;
					if (mat.empty())
					{
						break;
					}

					const auto results = Darknet::predict_and_annotate(net, mat);
					out.write(mat);
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
				}

				const auto timestamp_when_video_ended = std::chrono::high_resolution_clock::now();
				const auto processing_duration = timestamp_when_video_ended - timestamp_when_video_started;
				const size_t processing_time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(processing_duration).count();
				const double final_fps = 1000.0 * frame_counter / processing_time_in_milliseconds;

				std::cout
					<< "-> total frames processed ... " << frame_counter											<< std::endl
					<< "-> time to process video .... " << Darknet::format_duration_string(processing_duration)		<< std::endl
					<< "-> processed frame rate ..... " << final_fps << " FPS"										<< std::endl
					<< "-> total objects found ...... " << total_objects_found										<< std::endl
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
