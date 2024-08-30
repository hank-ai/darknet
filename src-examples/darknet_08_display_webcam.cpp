/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#include "darknet.hpp"

/** @file
 * This application will read from a webcam, run the video through Darknet/YOLO, and display the results.
 */


/* These define the dimensions and frame rate we'll be requesting from OpenCV once we open the webcam.  Remember that
 * each frame needs to be resized to match the network dimensions.  Don't ask for more than you need!  It takes much
 * longer to process a 1920x1080 image than it does for 640x480.
 *
 * If the webcam does not support the requested dimension or frame rate, OpenCV normally tries to find the nearest
 * values that will work.
 *
 * See the console output to compare what was requested, what OpenCV sais it found, and what is actually being returned.
 */
#if 1
const auto REQUEST_WEBCAM_WIDTH		= 1920;
const auto REQUEST_WEBCAM_HEIGHT	= 1080;
const auto REQUEST_WEBCAM_FPS		= 30.0;
#else
const auto REQUEST_WEBCAM_WIDTH		= 640;
const auto REQUEST_WEBCAM_HEIGHT	= 480;
const auto REQUEST_WEBCAM_FPS		= 30.0;
#endif

// 0==first camera, 1=second camera, etc
const auto REQUEST_WEBCAM_INDEX		= 0;

// When set to TRUE, a recording of the annotated webcam feed will be saved to output.mp4.
// When set to FALSE, the annotated webcam output is shown on screen but not saved to disk.
const auto SAVE_OUTPUT_VIDEO		= false;


cv::VideoCapture open_and_configure_camera(cv::VideoCapture & cap)
{
	std::cout << "Opening webcam..." << std::endl;

//	const cv::VideoCaptureAPIs backend = cv::VideoCaptureAPIs::CAP_ANY; // let OpenCV choose a back-end to use
	const cv::VideoCaptureAPIs backend = cv::VideoCaptureAPIs::CAP_V4L2;
//	const cv::VideoCaptureAPIs backend = cv::VideoCaptureAPIs::CAP_FFMPEG;
//	const cv::VideoCaptureAPIs backend = cv::VideoCaptureAPIs::CAP_GSTREAMER;

	cap.open(REQUEST_WEBCAM_INDEX, backend);
	if (not cap.isOpened())
	{
		throw std::runtime_error("failed to open the webcam #" + std::to_string(REQUEST_WEBCAM_INDEX));
	}

	// set the resolution and frame rate -- see the top of this file for the REQUEST_... values
	cap.set(cv::CAP_PROP_FRAME_WIDTH	, REQUEST_WEBCAM_WIDTH	);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT	, REQUEST_WEBCAM_HEIGHT	);
	cap.set(cv::CAP_PROP_FPS			, REQUEST_WEBCAM_FPS	);

	return cap;
}


double estimate_camera_fps(cv::VideoCapture & cap)
{
	std::cout << "Estimating FPS..." << std::endl;

	// the first frames can take longer to read immediately after the webcam
	// is opened and configured, so read and discard a few frames
	cv::Mat mat;
	for (int i = 0; i < 5; i ++)
	{
		cap >> mat;
	}

	// attempt to read several consecutive frames to estimate the real FPS
	size_t frame_counter = 0;
	const auto ts1 = std::chrono::high_resolution_clock::now();
	for (int i = 0; cap.isOpened() and i < 5; i ++)
	{
		cap >> mat;
		if (not mat.empty())
		{
			frame_counter ++;
		}
	}
	const auto ts2 = std::chrono::high_resolution_clock::now();

	const double actual_fps = static_cast<double>(frame_counter) / std::chrono::duration_cast<std::chrono::nanoseconds>(ts2 - ts1).count() * 1000000000.0f;

	return actual_fps;
}


int main(int argc, char * argv[])
{
	try
	{
		std::cout << "Darknet v" << DARKNET_VERSION_SHORT << std::endl;

		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		Darknet::NetworkPtr net = Darknet::load_neural_network(parms);
		int net_width = 0;
		int net_height = 0;
		int net_channels = 0;
		Darknet::network_dimensions(net, net_width, net_height, net_channels);

		cv::VideoCapture cap;
		open_and_configure_camera(cap);
		const double estimated_fps = estimate_camera_fps(cap);

		const std::string title = "Darknet/YOLO Webcam Output - " +
				Darknet::get_config_filename(net).filename().string() + " - " +
				Darknet::get_weights_filename(net).filename().string();

		cv::Mat mat;
		cap >> mat;
		cv::imshow("output", mat);
		cv::resizeWindow("output", mat.size());
		cv::setWindowTitle("output", title);
		cv::waitKey(10);

		std::string output_video_filename = "<skipped>";
		cv::VideoWriter out;
		if (SAVE_OUTPUT_VIDEO)
		{
			output_video_filename = "output.mp4";
			out.open(output_video_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), estimated_fps, mat.size());
		}

		const size_t opencv_width				= cap.get(cv::CAP_PROP_FRAME_WIDTH);
		const size_t opencv_height				= cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		const double opencv_fps					= cap.get(cv::CAP_PROP_FPS);
		const size_t actual_width				= mat.cols;
		const size_t actual_height				= mat.rows;
		const size_t show_stats_frequency		= std::round(estimated_fps * 1.5); // stats will be shown about every 1.5 seconds
		const double frame_in_nanoseconds		= 1000000000.0 / estimated_fps;
		const int frame_in_milliseconds			= std::round(frame_in_nanoseconds / 1000000.0);
		const int milliseconds_between_frames	= std::min(10, std::max(5, frame_in_milliseconds / 2));

		std::cout
			<< "-> network dimensions ....... " << net_width			<< " x " << net_height << " x " << net_channels << std::endl
			<< "-> requested dimensions ..... " << REQUEST_WEBCAM_WIDTH	<< " x " << REQUEST_WEBCAM_HEIGHT	<< std::endl
			<< "-> OpenCV reports ........... " << opencv_width			<< " x " << opencv_height			<< std::endl
			<< "-> actual dimensions ........ " << actual_width			<< " x " << actual_height			<< std::endl
			<< "-> requested frame rate ..... " << REQUEST_WEBCAM_FPS			<< " FPS"					<< std::endl
			<< "-> OpenCV reports ........... " << opencv_fps					<< " FPS"					<< std::endl
			<< "-> estimated frame rate ..... " << estimated_fps				<< " FPS"					<< std::endl
//			<< "-> each frame lasts ......... " << frame_in_nanoseconds			<< " nanoseconds"			<< std::endl
			<< "-> each frame lasts ......... " << frame_in_milliseconds		<< " milliseconds"			<< std::endl
			<< "-> sleep between frames ..... " << milliseconds_between_frames	<< " milliseconds"			<< std::endl
			<< "-> save output video ........ " << output_video_filename									<< std::endl
			<< "-> press ESC to exit"																		<< std::endl;

		size_t frame_counter		= 0;
		size_t error_counter		= 0;
		size_t recent_frame_counter	= 0;
		size_t total_objects_found	= 0;

		const auto timestamp_start = std::chrono::high_resolution_clock::now();
		auto timestamp_recent = timestamp_start;

		while (cap.isOpened() and error_counter < 10)
		{
			cap >> mat;
			if (mat.empty())
			{
				error_counter ++;
				continue;
			}
			error_counter = 0;
			frame_counter ++;

			// about once per second we'll display some statistics to the console
			if (frame_counter % show_stats_frequency == 0)
			{
				const auto now = std::chrono::high_resolution_clock::now();
				const double nanoseconds_elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - timestamp_recent).count();
				const double fps = (frame_counter - recent_frame_counter) / nanoseconds_elapsed * 1000000000.0;

				std::cout << "\r-> recent statistics ........ " << frame_counter << " frames, " << std::setprecision(1) << fps << " FPS " << std::flush;

				timestamp_recent = now;
				recent_frame_counter = frame_counter;
			}

			// apply the neural network to this frame
			const auto results = Darknet::predict_and_annotate(net, mat);
			total_objects_found += results.size();

			if (out.isOpened())
			{
				out.write(mat);
			}

			cv::imshow("output", mat);
			const auto key = cv::waitKey(milliseconds_between_frames);
			if (key == 27)
			{
				break;
			}
		}
		const auto timestamp_end = std::chrono::high_resolution_clock::now();
		std::cout << std::endl;

		const auto video_duration = timestamp_end - timestamp_start;
		const size_t video_length_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(video_duration).count();
		const double final_fps = 1000.0 * frame_counter / video_length_in_milliseconds;

		std::cout
			<< "-> recent error counter ..... " << error_counter													<< std::endl
			<< "-> total frames captured .... " << frame_counter													<< std::endl
			<< "-> total length of video .... " << video_length_in_milliseconds << " milliseconds"					<< std::endl
			<< "-> final frame rate ......... " << final_fps << " FPS"												<< std::endl
			<< "-> total objects founds ..... " << total_objects_found												<< std::endl
			<< "-> average objects/frame .... " << static_cast<float>(total_objects_found) / frame_counter			<< std::endl;

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
