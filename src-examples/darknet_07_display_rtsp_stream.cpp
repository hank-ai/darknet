/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#include "darknet.hpp"

/** @file
 * This application will read from a RTP stream, run the video through Darknet/YOLO, and display the results.
 *
 * If you don't have a device that generates a RTP stream, you can use VLC and a computer with a webcam such as a laptop.
 *
 * For example, from my laptop I run one of these commands to start a stream on the multicast address 239.0.0.1:
 *
 * ~~~~{.sh}
 *		cvlc -vvv v4l2:///dev/video0 :v4l2-width=640 :v4l2-height=480 :v4l2-fps=10 --sout '#transcode{vcodec=mp2v,vb=256,width=640,height=480,acodec=none}:rtp{dst=239.0.0.1,port=43210,mux=ts}'
 * ~~~~
 * or:
 * ~~~~{.sh}
 *		cvlc -vvv v4l2:///dev/video0 :v4l2-width=1280 :v4l2-height=720 :v4l2-fps=30 --sout '#transcode{vcodec=mp2v,width=1280,height=720,acodec=none}:rtp{dst=239.0.0.1,port=43210,mux=ts}'
 * ~~~~
 */


int main(int argc, char * argv[])
{
	try
	{
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);

		std::string stream = "rtp://239.0.0.1:43210";

		// see if the user gave a specific RTP or RTSP stream to use
		for (const auto & parm : parms)
		{
			if (parm.type == Darknet::EParmType::kOther and
				(parm.string.find("rtp") == 0 or parm.string.find("rtsp") == 0))
			{
				stream = parm.string;
				break;
			}
		}

#ifdef WIN32
		// There are *many* possible backends we can use.  Over 30.  Many of which are Windows-specific.  But it appears
		// that CAP_ANY might be a good choice to make.  See issue #97:  https://github.com/hank-ai/darknet/issues/97
		const cv::VideoCaptureAPIs backend = cv::VideoCaptureAPIs::CAP_ANY; // let OpenCV choose a back-end to use
#else
//		const cv::VideoCaptureAPIs backend = cv::VideoCaptureAPIs::CAP_ANY;
//		const cv::VideoCaptureAPIs backend = cv::VideoCaptureAPIs::CAP_V4L2;
//		const cv::VideoCaptureAPIs backend = cv::VideoCaptureAPIs::CAP_FFMPEG;
		const cv::VideoCaptureAPIs backend = cv::VideoCaptureAPIs::CAP_GSTREAMER;
#endif

		std::cout << "Connecting to stream " << stream << "..." << std::endl;
		cv::VideoCapture cap(stream, backend);
		if (not cap.isOpened())
		{
			throw std::runtime_error("failed to open the stream \"" + stream + "\"");
		}

		// attempt to read a frame so we have accurate information on the stream
		std::cout << "Attempting to read from stream " << stream << "..";
		for (int idx = 0; cap.isOpened() and idx < 20; idx ++)
		{
			std::cout << "." << std::flush;
			cv::Mat mat;
			cap >> mat;
			if (mat.empty() == false)
			{
				break;
			}
		}
		std::cout << std::endl;

		const size_t video_width				= cap.get(cv::CAP_PROP_FRAME_WIDTH);
		const size_t video_height				= cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		const double fps						= cap.get(cv::CAP_PROP_FPS);
		const size_t fps_rounded				= std::round(std::max(1.0, fps));
		const size_t nanoseconds_per_frame		= std::round(1000000000.0 / fps);
		const size_t milliseconds_per_frame		= std::round(1000.0 / fps);
		std::cout
			<< "-> frame dimensions ......... " << video_width << " x " << video_height			<< std::endl
			<< "-> frame rate ............... " << fps << " FPS"								<< std::endl
			<< "-> each frame lasts ......... " << nanoseconds_per_frame << " nanoseconds"		<< std::endl
			<< "-> each frame lasts ......... " << milliseconds_per_frame << " milliseconds"	<< std::endl;

		size_t frame_counter		= 0;
		size_t total_objects_found	= 0;
		size_t recent_error_counter	= 0;
		auto total_sleep_time		= std::chrono::milliseconds(0);
		const auto timestamp_when_stream_started = std::chrono::high_resolution_clock::now();

		cv::namedWindow(stream, cv::WindowFlags::WINDOW_GUI_NORMAL);
		cv::resizeWindow(stream, cv::Size(video_width, video_height));

		while (cap.isOpened() and recent_error_counter < 20)
		{
			cv::Mat mat;
			cap >> mat;
			if (mat.empty())
			{
				recent_error_counter ++;
				continue;
			}
			recent_error_counter = 0;

			const auto results = Darknet::predict_and_annotate(net, mat);
			cv::imshow(stream, mat);
			frame_counter ++;
			total_objects_found += results.size();

			if (frame_counter % fps_rounded == 0)
			{
				std::cout << "-> frame #" << frame_counter << "\r" << std::flush;
			}

			// sleep for a reasonable amount of time beween each frame
			const int time_to_sleep_in_milliseconds = std::clamp(milliseconds_per_frame / 2UL, 5UL, 20UL);
			total_sleep_time += std::chrono::milliseconds(time_to_sleep_in_milliseconds);
			const char c = cv::waitKey(time_to_sleep_in_milliseconds);
			if (c == 27) // ESC
			{
				std::cout << std::endl << "ESC!" << std::endl;
				break;
			}
		}

		const auto timestamp_when_stream_ended		= std::chrono::high_resolution_clock::now();
		const auto video_duration					= timestamp_when_stream_ended - timestamp_when_stream_started;
		const auto average_sleep_per_frame			= total_sleep_time / frame_counter;
		const size_t video_length_in_milliseconds	= std::chrono::duration_cast<std::chrono::milliseconds>(video_duration).count();
		const double final_fps						= 1000.0 * frame_counter / video_length_in_milliseconds;

		std::cout
			<< "-> recent error counter ..... " << recent_error_counter										<< std::endl
			<< "-> number of frames shown ... " << frame_counter											<< std::endl
			<< "-> average sleep per frame .. " << Darknet::format_duration_string(average_sleep_per_frame)	<< std::endl
			<< "-> total length of stream ... " << Darknet::format_duration_string(video_duration)			<< std::endl
			<< "-> processed frame rate ..... " << final_fps << " FPS"										<< std::endl
			<< "-> total objects founds ..... " << total_objects_found										<< std::endl
			<< "-> average objects/frame .... " << static_cast<float>(total_objects_found) / frame_counter	<< std::endl;

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
