/* Darknet/YOLO:  https://github.com/hank-ai/darknet
 * Copyright 2024 Stephane Charette
 */

#include "darknet.hpp"
#include "darknet_image.hpp"

#include <set>
#include <thread>

/** @file
 * This application will process one or more videos as fast as possible using multiple threads and save a new output
 * video to disk.  The results are not shown to the user.  Call it like this:
 *
 *     darknet_05_process_videos_multithreaded LegoGears DSCN1582A.MOV
 *
 * The output should be similar to this:
 *
 *     processing DSCN1582A.MOV:
 *     -> total number of CPUs ..... 16
 *     -> threads for this video ... 4
 *     -> neural network size ...... 224 x 160 x 3
 *     -> input video dimensions ... 640 x 480
 *     -> input video frame count .. 1230
 *     -> input video frame rate ... 29.970030 FPS
 *     -> input video length ....... 41041 milliseconds
 *     -> output filename .......... DSCN1582A_output.m4v
 *     -> total frames processed ... 1230
 *     -> time to process video .... 1719 milliseconds
 *     -> processed frame rate ..... 715.532286 FPS
 *     -> total objects founds ..... 6189
 *     -> average objects/frame .... 5.031707
 */


/// Everything we know about a specific video frame is stored in one of these objects.
struct Frame
{
	size_t					index;			///< zero-based frame index into the video
	cv::Mat					mat;			///< the original frame, and then the annotated frame
	Darknet::Image			img;			///< Darknet-specific image, resized original frame
	Darknet::Predictions	predictions;	///< all predictions made by Darknet/YOLO for this frame

	/// std::set uses @p operator<() to ensure the frames are ordered numerically within the set
	bool operator<(const Frame & rhs) const
	{
		return index < rhs.index;
	}
};


bool				all_threads_must_exit	= false;	///< if something goes wrong, this flag gets set to @p true
Darknet::NetworkPtr	net						= nullptr;	///< Darknet/YOLO neural network pointer
cv::Size			network_dimensions;					///< dimensions of the neural network that was loaded
std::set<Frame>		frames_waiting_for_resize;			///< once a frame is read from the video, it is stored here
std::set<Frame>		frames_waiting_for_prediction;		///< once a frame has been resized, it is stored here
std::set<Frame>		frames_waiting_for_output;			///< once a frame has been predicted, it is stored here
std::mutex			waiting_for_resize;					///< mutex to protect access to @ref frames_waiting_for_resize
std::mutex			waiting_for_prediction;				///< mutex to protect access to @ref frames_waiting_for_prediction
std::mutex			waiting_for_output;					///< mutex to protect access to @ref frames_waiting_for_output
std::chrono::high_resolution_clock::duration reader_work_duration;	///< amount of time spent reading frames
std::chrono::high_resolution_clock::duration resize_work_duration;	///< amount of time spent resizing frames
std::chrono::high_resolution_clock::duration predict_work_duration;	///< amount of time spent predicting frames
std::chrono::high_resolution_clock::duration output_work_duration;	///< amount of time spent on the output video
size_t				reader_must_pause		= 0;
size_t				resize_thread_starved	= 0;
size_t				predict_thread_starved	= 0;
size_t				output_thread_starved	= 0;


void resize_thread()
{
	try
	{
		while (all_threads_must_exit == false)
		{
			if ( frames_waiting_for_resize.empty())
			{
				resize_thread_starved ++;
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			}

			const auto timestamp_begin = std::chrono::high_resolution_clock::now();

			Frame frame;
			if (true)
			{
				std::scoped_lock lock(waiting_for_resize);
				auto iter = frames_waiting_for_resize.begin();
				frame = *iter;
				frames_waiting_for_resize.erase(iter);
			}

			cv::Mat tmp;
			if (frame.mat.size() == network_dimensions)
			{
				tmp = frame.mat.clone();
			}
			else
			{
				cv::resize(frame.mat, tmp, network_dimensions, cv::INTER_NEAREST);
			}

			if (tmp.channels() == 3)
			{
				cv::cvtColor(tmp, tmp, cv::COLOR_BGR2RGB);
			}

			frame.img = Darknet::mat_to_image(tmp);

			if (true)
			{
				std::scoped_lock lock(waiting_for_prediction);
				frames_waiting_for_prediction.insert(frame);
			}

			const auto timestamp_end = std::chrono::high_resolution_clock::now();
			resize_work_duration += timestamp_end - timestamp_begin;
		}
	}
	catch(const std::exception & e)
	{
		std::cout << "ERROR: resize thread exception: " << e.what() << std::endl;
		all_threads_must_exit = true;
	}

	return;
}


void detection_thread(size_t & total_objects_found)
{
	try
	{
		while (all_threads_must_exit == false)
		{
			if (frames_waiting_for_prediction.empty())
			{
				predict_thread_starved ++;
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			}

			// if we get here we know we have at least 1 frame where we can call predict()
			const auto timestamp_begin = std::chrono::high_resolution_clock::now();

			Frame frame;
			if (true)
			{
				std::scoped_lock lock(waiting_for_prediction);
				auto iter = frames_waiting_for_prediction.begin();
				frame = *iter;
				frames_waiting_for_prediction.erase(iter);
			}

			frame.predictions = Darknet::predict(net, frame.img, frame.mat.size());
			Darknet::annotate(net, frame.predictions, frame.mat);

			total_objects_found += frame.predictions.size();

			// in case multiple resize threads are running, we have no idea in which order the frames have been processed,
			// so pass them to another thread which will ensure the frames are re-ordered before creating the output video
			std::scoped_lock lock(waiting_for_output);
			frames_waiting_for_output.insert(frame);

			const auto timestamp_end = std::chrono::high_resolution_clock::now();
			predict_work_duration += timestamp_end - timestamp_begin;
		}
	}
	catch(const std::exception & e)
	{
		std::cout << "ERROR: detection thread exception: " << e.what() << std::endl;
		all_threads_must_exit = true;
	}

	return;
}


void output_thread(cv::VideoWriter & out)
{
	try
	{
		size_t expected_next_index = 0;

		while (all_threads_must_exit == false)
		{
			if ( frames_waiting_for_output.empty())
			{
				output_thread_starved ++;
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
				continue;
			}

			// we have at least 1 frame ready...but is it the right one?

			if (frames_waiting_for_output.begin()->index != expected_next_index)
			{
				output_thread_starved ++;
				std::this_thread::yield();
				continue;
			}

			// if we get here, then the next frame we want is at the front of the set
			const auto timestamp_begin = std::chrono::high_resolution_clock::now();

			Frame frame;
			if (true)
			{
				std::scoped_lock lock(waiting_for_output);
				auto iter = frames_waiting_for_output.begin();
				frame = *iter;
				frames_waiting_for_output.erase(iter);
			}

			out.write(frame.mat);
			expected_next_index ++;

			const auto timestamp_end = std::chrono::high_resolution_clock::now();
			output_work_duration += timestamp_end - timestamp_begin;

//			cv::imshow("output", frame.mat);
//			cv::waitKey(5);
		}
	}
	catch(const std::exception & e)
	{
		std::cout << "ERROR: output thread exception: " << e.what() << std::endl;
		all_threads_must_exit = true;
	}

	return;
}


int main(int argc, char * argv[])
{
	try
	{
		Darknet::show_version_info();

		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		net = Darknet::load_neural_network(parms);
//		Darknet::set_annotation_line_type(net, cv::LineTypes::LINE_AA);

		int network_width = 0;
		int network_height = 0;
		int network_channels = 0;
		Darknet::network_dimensions(net, network_width, network_height, network_channels);
		network_dimensions = cv::Size(network_width, network_height);

		std::vector<std::thread> threads;

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
				const size_t fps_rounded				= std::round(fps);
				const size_t nanoseconds_per_frame		= std::round(1000000000.0 / fps);
				const size_t video_length_milliseconds	= std::round(nanoseconds_per_frame / 1000000.0 * video_frames_count);
				size_t total_objects_found				= 0;
				size_t frame_counter					= 0;

				cv::VideoWriter out(output_filename, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(video_width, video_height));
				if (not out.isOpened())
				{
					std::cout << "Failed to open the output video file " << output_filename << std::endl;
					continue;
				}

				/* These are the tasks that must be performed:
				 *
				 *		1) read the frames from the video file
				 *		2) resize the frames, convert to RGB, convert to Darknet::Image format
				 *		3) call Darknet/YOLO predict() and annotate() on each frame
				 *		4) write the annotated image to the output video
				 *
				 * From a complexity point of view, each one of those tasks takes longer than the task before.  So reading is simple,
				 * resizing take a bit more time than reading, predicting takes more time than resizing, and the longest of all is
				 * writing the video back to disk because of the re-encoding.
				 *
				 * By rate limiting the first and fastest task, we can control the rest of the tasks after that.
				 */

				// start all the threads we'll need -- the "main" thread will take care of task #1 (reading)
				threads.emplace_back(resize_thread);									// task #2
				threads.emplace_back(detection_thread, std::ref(total_objects_found));	// task #3
				threads.emplace_back(output_thread, std::ref(out));						// task #4

				std::cout
					<< "-> total number of CPUs ..... " << std::thread::hardware_concurrency()			<< std::endl
					<< "-> threads for this video ... " << threads.size() + 1 /* this thread */			<< std::endl
					<< "-> neural network size ...... " << network_width << " x " << network_height << " x " << network_channels << std::endl
					<< "-> input video dimensions ... " << video_width << " x " << video_height			<< std::endl
					<< "-> input video frame count .. " << video_frames_count							<< std::endl
					<< "-> input video frame rate ... " << fps << " FPS"								<< std::endl
					<< "-> input video length ....... " << video_length_milliseconds << " milliseconds"	<< std::endl
					<< "-> output filename .......... " << output_filename								<< std::endl;

				const auto timestamp_when_video_started = std::chrono::high_resolution_clock::now();

				while (all_threads_must_exit == false)
				{
					const auto timestamp_begin = std::chrono::high_resolution_clock::now();

					Frame frame;
					frame.index = frame_counter;
					cap >> frame.mat;
					if (frame.mat.empty())
					{
						break;
					}

					// place the frame on the queue so it can be resized
					if (true)
					{
						std::scoped_lock lock(waiting_for_resize);
						frames_waiting_for_resize.insert(frame);
					}

					frame_counter ++;
					if (frame_counter % fps_rounded == 0)
					{
						const int percentage = std::round(100.0f * frame_counter / video_frames_count);
						std::cout
							<< "-> frame #" << frame_counter << "/" << video_frames_count
							<< " (" << percentage << "%)\r"
							<< std::flush;
					}

					const auto timestamp_end = std::chrono::high_resolution_clock::now();
					reader_work_duration += timestamp_end - timestamp_begin;

					while (	all_threads_must_exit == false and
							frames_waiting_for_resize		.size() +
							frames_waiting_for_prediction	.size() +
							frames_waiting_for_output		.size() > 5)
					{
						// reader thread is getting too far ahead of the other threads, we need to slow down
						reader_must_pause ++;
						std::this_thread::sleep_for(std::chrono::milliseconds(1));
					}
				}

				// even though we finished reading the frames from the input video, the other threads may not yet have finished
				while (all_threads_must_exit == false and
					frames_waiting_for_resize		.size() +
					frames_waiting_for_prediction	.size() +
					frames_waiting_for_output		.size() > 0)
				{
					std::this_thread::yield();
				}
				all_threads_must_exit = true;

				const auto timestamp_when_video_ended = std::chrono::high_resolution_clock::now();
				const auto processing_duration = timestamp_when_video_ended - timestamp_when_video_started;
				const size_t processing_time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(processing_duration).count();
				const double final_fps = 1000.0 * frame_counter / processing_time_in_milliseconds;

				std::cout
					<< "-> total frames processed ... " << frame_counter											<< std::endl
					<< "-> time to process video .... " << processing_time_in_milliseconds << " milliseconds"		<< std::endl
					<< "-> processed frame rate ..... " << final_fps << " FPS"										<< std::endl
					<< "-> total objects founds ..... " << total_objects_found										<< std::endl
					<< "-> average objects/frame .... " << static_cast<float>(total_objects_found) / frame_counter	<< std::endl
#if 0
					// timing details are commented out, they're mostly for development purpose not end user consumption
					<< "-> reader chose to pause .... " << reader_must_pause										<< std::endl
					<< "-> resize thread starved .... " << resize_thread_starved									<< std::endl
					<< "-> predict thread starved ... " << predict_thread_starved									<< std::endl
					<< "-> output thread starved .... " << output_thread_starved									<< std::endl
					<< "-> time spent reading ....... " << std::chrono::duration_cast<std::chrono::milliseconds>(reader_work_duration).count() << " milliseconds" << std::endl
					<< "-> time spent resizing ...... " << std::chrono::duration_cast<std::chrono::milliseconds>(resize_work_duration).count() << " milliseconds" << std::endl
					<< "-> time spent predicting .... " << std::chrono::duration_cast<std::chrono::milliseconds>(predict_work_duration).count() << " milliseconds" << std::endl
					<< "-> time spent output video .. " << std::chrono::duration_cast<std::chrono::milliseconds>(output_work_duration).count() << " milliseconds" << std::endl
#endif
					;

				// all threads should have exited by now
				for (auto & t : threads)
				{
					t.join();
				}
				threads.clear();
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
