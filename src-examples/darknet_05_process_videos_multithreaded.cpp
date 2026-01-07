/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2026 Stephane Charette
 */

#include "darknet.hpp"
#include "darknet_image.hpp"

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <sstream>
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
 *     -> total objects found ...... 6189
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
std::chrono::high_resolution_clock::duration wait_threads_duration;	///< amount of time spent waiting for other threads to finish running
std::chrono::high_resolution_clock::duration reader_work_duration;	///< amount of time spent reading frames
std::chrono::high_resolution_clock::duration resize_work_duration;	///< amount of time spent resizing frames
std::chrono::high_resolution_clock::duration predict_work_duration;	///< amount of time spent predicting frames
std::chrono::high_resolution_clock::duration output_work_duration;	///< amount of time spent on the output video
size_t				reader_must_pause		= 0;
size_t				resize_thread_starved	= 0;
size_t				predict_thread_starved	= 0;
size_t				output_thread_starved	= 0;
size_t				waiting_for_threads		= 0;
size_t				expected_next_index		= 0; ///< keep track of which frame has been written to disk
std::string			bench_label;
std::string			bench_suffix;
bool				bench_overlay_enabled	= false;

std::string sanitize_label(const std::string & input)
{
	std::string output;
	output.reserve(input.size());

	char last = '\0';
	for (const unsigned char ch : input)
	{
		if (std::isalnum(ch))
		{
			output.push_back(static_cast<char>(std::tolower(ch)));
			last = output.back();
		}
		else if (!output.empty() && last != '_')
		{
			output.push_back('_');
			last = '_';
		}
	}

	while (!output.empty() && output.back() == '_')
	{
		output.pop_back();
	}

	return output;
}

static void default_bench_label_and_suffix(std::string & label, std::string & suffix)
{
#ifdef DARKNET_USE_MPS
	const char *postproc_env = std::getenv("DARKNET_MPS_POSTPROC");
	const bool postproc_on = (postproc_env && postproc_env[0] != '\0' && postproc_env[0] != '0');
	const bool compare_on = (postproc_env && (postproc_env[0] == '2' || std::strncmp(postproc_env, "compare", 7) == 0));
	if (postproc_on)
	{
		label = compare_on ? "GPU - Apple MPS (postproc compare)" : "GPU - Apple MPS (postproc)";
		suffix = compare_on ? "mps_postproc_compare" : "mps_postproc";
	}
	else
	{
		label = "GPU - Apple MPS";
		suffix = "mps";
	}
#else
#ifdef DARKNET_USE_OPENBLAS
	label = "CPU - OpenBLAS";
	suffix = "openblas";
#else
	label = "CPU - no OpenBLAS";
	suffix = "cpu_only";
#endif
#endif
}


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

			frame.img = Darknet::bgr_mat_to_rgb_image(tmp);

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
			Darknet::free_image(frame.img); // release resized RGB buffer once inference is complete

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
		expected_next_index = 0;
		size_t frames_written = 0;
		const auto output_start = std::chrono::high_resolution_clock::now();

		while (all_threads_must_exit == false)
		{
			if (frames_waiting_for_output.empty())
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

			if (bench_overlay_enabled)
			{
				const auto now = std::chrono::high_resolution_clock::now();
				const std::chrono::duration<double> elapsed = now - output_start;
				const double fps = (elapsed.count() > 0.0) ? (static_cast<double>(frames_written + 1) / elapsed.count()) : 0.0;

				std::ostringstream oss;
				oss << std::fixed << std::setprecision(2) << fps;

				const cv::Point title_pos(12, 28);
				const cv::Point fps_pos(12, 54);
				const int font = cv::FONT_HERSHEY_SIMPLEX;
				const double scale = 0.6;
				const int thickness = 1;
				const cv::Scalar colour(0, 255, 0);

				cv::putText(frame.mat, bench_label, title_pos, font, scale, colour, thickness, cv::LINE_AA);
				cv::putText(frame.mat, "FPS: " + oss.str(), fps_pos, font, scale, colour, thickness, cv::LINE_AA);
			}

			out.write(frame.mat);
			expected_next_index ++;
			frames_written ++;

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
		Darknet::Parms parms = Darknet::parse_arguments(argc, argv);
		net = Darknet::load_neural_network(parms);
//		Darknet::set_annotation_line_type(net, cv::LineTypes::LINE_AA);

		int network_width = 0;
		int network_height = 0;
		int network_channels = 0;
		Darknet::network_dimensions(net, network_width, network_height, network_channels);
		network_dimensions = cv::Size(network_width, network_height);

		default_bench_label_and_suffix(bench_label, bench_suffix);
		bench_overlay_enabled = !bench_label.empty();

		if (const char * env_label = std::getenv("DARKNET_BENCH_LABEL"))
		{
			bench_label = env_label;
			bench_overlay_enabled = !bench_label.empty();
		}

		if (const char * env_suffix = std::getenv("DARKNET_BENCH_SUFFIX"))
		{
			bench_suffix = env_suffix;
		}
		else if (bench_suffix.empty() && !bench_label.empty())
		{
			bench_suffix = sanitize_label(bench_label);
		}

		std::string config_stem = "config";
		const auto config_path = Darknet::get_config_filename(parms);
		if (!config_path.empty())
		{
			config_stem = config_path.stem().string();
		}

		std::vector<std::thread> threads;

		for (const auto & parm : parms)
		{
			if (parm.type != Darknet::EParmType::kFilename)
			{
				continue;
			}

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

			std::string output_filename			= std::filesystem::path(parm.string).stem().string();
			output_filename += "_" + config_stem;
			if (!bench_suffix.empty())
			{
				output_filename += "_" + bench_suffix;
			}
			else
			{
				output_filename += "_output";
			}
			output_filename += ".m4v";
			std::filesystem::path output_dir		= std::filesystem::path(parm.string).parent_path();
			if (output_dir.empty())
			{
				output_dir = std::filesystem::current_path();
			}
			const std::string output_path			= (output_dir / output_filename).string();
			const size_t video_width				= mat.cols;
			const size_t video_height				= mat.rows;
			const size_t video_channels				= mat.channels();
			const size_t video_frames_count			= cap.get(cv::CAP_PROP_FRAME_COUNT);
			const double fps						= cap.get(cv::CAP_PROP_FPS);
			const size_t fps_rounded				= std::round(fps);
			const size_t nanoseconds_per_frame		= std::round(1000000000.0 / fps);
			const size_t video_length_milliseconds	= std::round(nanoseconds_per_frame / 1000000.0 * video_frames_count);

			size_t total_objects_found				= 0;
			size_t frame_counter					= 0;
			wait_threads_duration	= std::chrono::high_resolution_clock::duration();
			reader_work_duration	= std::chrono::high_resolution_clock::duration();
			resize_work_duration	= std::chrono::high_resolution_clock::duration();
			predict_work_duration	= std::chrono::high_resolution_clock::duration();
			output_work_duration	= std::chrono::high_resolution_clock::duration();
			reader_must_pause		= 0;
			resize_thread_starved	= 0;
			predict_thread_starved	= 0;
			output_thread_starved	= 0;
			waiting_for_threads		= 0;
			expected_next_index		= 0;

			cv::VideoWriter out(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(video_width, video_height));
			if (not out.isOpened())
			{
				std::cout << "Failed to open the output video file " << output_path << std::endl;
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
			all_threads_must_exit = false;
			threads.emplace_back(resize_thread);									// task #2
			threads.emplace_back(detection_thread, std::ref(total_objects_found));	// task #3
			threads.emplace_back(output_thread, std::ref(out));						// task #4

			std::cout
				<< "-> total number of CPUs ..... " << std::thread::hardware_concurrency()			<< std::endl
				<< "-> threads for this video ... " << threads.size() + 1 /* this thread */			<< std::endl
				<< "-> neural network size ...... " << network_width	<< " x " << network_height	<< " x " << network_channels	<< std::endl
				<< "-> input video dimensions ... " << video_width		<< " x " << video_height	<< " x " << video_channels		<< std::endl
				<< "-> input video frame count .. " << video_frames_count							<< std::endl
				<< "-> input video frame rate ... " << fps << " FPS"								<< std::endl
				<< "-> input video length ....... " << Darknet::format_duration_string(std::chrono::milliseconds(video_length_milliseconds)) << std::endl
				<< "-> output filename .......... " << output_path									<< std::endl;

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
			const auto begin_waiting = std::chrono::high_resolution_clock::now();
			while (all_threads_must_exit == false and expected_next_index < frame_counter)
			{
				waiting_for_threads ++;
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
			wait_threads_duration = std::chrono::high_resolution_clock::now() - begin_waiting;
			all_threads_must_exit = true;

			const auto timestamp_when_video_ended = std::chrono::high_resolution_clock::now();
			const auto processing_duration = timestamp_when_video_ended - timestamp_when_video_started;
			const size_t processing_time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(processing_duration).count();
			const double final_fps = 1000.0 * frame_counter / processing_time_in_milliseconds;

			std::cout
				<< "-> total frames processed ... " << frame_counter											<< std::endl
				<< "-> time to process video .... " << Darknet::format_duration_string(processing_duration)		<< std::endl
				<< "-> processed frame rate ..... " << final_fps << " FPS"										<< std::endl
				<< "-> total objects found ...... " << total_objects_found										<< std::endl
				<< "-> average objects/frame .... " << static_cast<float>(total_objects_found) / frame_counter	<< std::endl
#if 0
				// timing details are commented out, they're mostly for development purpose not end user consumption
				<< "-> reader chose to pause .... " << reader_must_pause										<< std::endl
				<< "-> resize thread starved .... " << resize_thread_starved									<< std::endl
				<< "-> predict thread starved ... " << predict_thread_starved									<< std::endl
				<< "-> output thread starved .... " << output_thread_starved									<< std::endl
				<< "-> waiting for threads ...... " << waiting_for_threads										<< std::endl
				<< "-> time spent reading ....... " << Darknet::format_duration_string(reader_work_duration)	<< std::endl
				<< "-> time spent resizing ...... " << Darknet::format_duration_string(resize_work_duration)	<< std::endl
				<< "-> time spent predicting .... " << Darknet::format_duration_string(predict_work_duration)	<< std::endl
				<< "-> time spent output video .. " << Darknet::format_duration_string(output_work_duration)	<< std::endl
				<< "-> time waiting for threads . " << Darknet::format_duration_string(wait_threads_duration)	<< std::endl
#endif
				;

			// all threads should have exited by now
			for (auto & t : threads)
			{
				t.join();
			}
			threads.clear();
		}

		Darknet::free_neural_network(net);
	}
	catch (const std::exception & e)
	{
		std::cout << "Exception: " << e.what() << std::endl;
	}

	return 0;
}
