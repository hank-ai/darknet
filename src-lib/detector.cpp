#include "darknet_internal.hpp"
#include "option_list.hpp"
#include "data.hpp"


#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	static const int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

	static std::atomic<size_t> prime_image_counter = 0;
	static std::atomic<bool> prime_loading_threads_must_exit = false;
}


void prime_training_images_cache_thread(Darknet::VStr v, const int width, const int height, const int channels) // intentionally not by reference
{
	TAT(TATPARMS);

	// this runs on a secondary thread started by prime_training_images_cache()

	while (	prime_loading_threads_must_exit		== false and
			cfg_and_state.must_immediately_exit	== false)
	{
		for (auto fn : v) // not by reference since we modify the extension to get the .txt filename
		{
			if (prime_loading_threads_must_exit or cfg_and_state.must_immediately_exit)
			{
				break;
			}

			prime_image_counter ++;
			Darknet::Image img = Darknet::load_image(fn.c_str(), width, height, channels);

			if (img.data == nullptr or img.w != width or img.h != height or img.c != channels)
			{
				Darknet::display_warning_msg("unexpected error while loading image " + fn + "\n");
			}

			Darknet::free_image(img);

			// now read the .txt file that goes with this image
			const size_t pos = fn.rfind(".");
			if (pos != std::string::npos)
			{
				fn.erase(pos);
			}
			fn += ".txt";

			std::ifstream ifs(fn);
			if (ifs.good())
			{
				std::string line;
				ifs >> line;
			}
			else
			{
				Darknet::display_warning_msg("unexpected error while reading annotations from " + fn + "\n");
			}
		}
	}

	return;
}


void prime_training_images_cache(list * image_filenames, const int width, const int height, const int channels)
{
	TAT(TATPARMS);

	if (image_filenames != nullptr and image_filenames->size > 0 and prime_image_counter == 0)
	{
		// quickly load (and immediately discard) all the images so the operating system's cache is primed

		prime_loading_threads_must_exit = false;
		const size_t n = std::max(2U, std::thread::hardware_concurrency());
		const int to_load = std::ceil(image_filenames->size / static_cast<float>(n));

		*cfg_and_state.output << "Using " << n << " threads to prime loading " << image_filenames->size << " images into cache (w=" << width << ", h=" << height << ", c=" << channels << ")." << std::endl;

		Darknet::VThreads threads;
		threads.reserve(n);

		auto * node = image_filenames->front;

		const auto t1 = std::chrono::high_resolution_clock::now();

		for (size_t i = 0; i < n; i ++)
		{
			Darknet::VStr v;
			v.reserve(to_load);
			while (v.size() < to_load and node != nullptr)
			{
				v.push_back(reinterpret_cast<char*>(image_filenames->front->val));
				node = node->next;
			}

			// start a thread to load all the images in the vector
			threads.emplace_back(prime_training_images_cache_thread, v, width, height, channels);
		}

		auto t2 = t1;
		const auto time_limit = t1 + std::chrono::seconds(30);
		while (cfg_and_state.must_immediately_exit == false)
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(333));
			const size_t recent = prime_image_counter.load();
			const int percentage = std::round(100.0f * recent / image_filenames->size);
			t2 = std::chrono::high_resolution_clock::now();

			*cfg_and_state.output << "\r-> loading image #" << recent << " (" << percentage << "%) in " << Darknet::format_duration_string(t2 - t1) << "      " << std::flush;

			if (t2 > time_limit)
			{
				*cfg_and_state.output << std::endl << "-> exit from image cache prime; time limit reached";
				break;
			}

			if (recent >= image_filenames->size * 2)
			{
				*cfg_and_state.output << std::endl << "-> exit from image cache prime; every image loaded at least twice";
				break;
			}
		}

		prime_loading_threads_must_exit = true;
		t2 = std::chrono::high_resolution_clock::now();
		for (auto & t : threads)
		{
			t.join();
		}

		const size_t nanoseconds				= std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
		const size_t load_time_for_1_image		= std::round(1.00f * nanoseconds / prime_image_counter);
		const size_t load_time_for_64_images	= std::round(64.0f * nanoseconds / prime_image_counter);

		*cfg_and_state.output
			<< std::endl
			<< "-> loaded a total of "		<< prime_image_counter << " images (" << (int)std::round(100.0f * prime_image_counter / image_filenames->size) << "%)"
			<< " in "						<< Darknet::format_duration_string(t2 - t1)														<< std::endl
			<< "-> image loading time: "	<< Darknet::format_duration_string(std::chrono::nanoseconds(load_time_for_1_image	))
			<< " (64="						<< Darknet::format_duration_string(std::chrono::nanoseconds(load_time_for_64_images	)) << ")"	<< std::endl
			<< "-> done with the "			<< threads.size() << " image cache prime threads"												<< std::endl;
	}

	return;
}


void train_detector_internal(const bool break_after_burn_in, std::string & multi_gpu_weights_fn, const char * datacfg, const char * cfgfile, const char * weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, float thresh, float iou_thresh, int show_imgs, int benchmark_layers, const char * chart_path)
{
	TAT(TATPARMS);

	#ifndef DARKNET_GPU
	*cfg_and_state.output << std::endl;
	Darknet::display_warning_msg("THIS VERSION OF DARKNET WAS NOT BUILT TO RUN ON A GPU!");
	*cfg_and_state.output															<< std::endl
		<< ""																		<< std::endl
		<< "While you can attempt to train a network using your CPU, it is not"		<< std::endl
		<< "recommended, and the output is highly unlikely to work.  Additionally,"	<< std::endl
		<< "training with a CPU versus a GPU would take an extremely long time."	<< std::endl
		<< ""																		<< std::endl
		<< "Are you certain you want to proceed with training on a CPU?"			<< std::endl
		<< ""																		<< std::endl
		<< "Type \"yes\" to begin training, or anything else to quit:  "			<< std::flush;

	std::string prompt;
	std::cin >> prompt;
	if (prompt != "yes")
	{
		darknet_fatal_error(DARKNET_LOC, "Cancel the attempt to train a neural network using a CPU.");
	}
	#endif

//	const std::filesystem::path & datacfg		= cfg_and_state.data_filename;
//	const std::filesystem::path & cfgfile		= cfg_and_state.cfg_filename;
//	const std::filesystem::path & weightfile	= cfg_and_state.weights_filename;

	list *options = read_data_cfg(datacfg);
	const char *train_images = option_find_str(options, "train", "data/train.txt");
	const char *valid_images = option_find_str(options, "valid", train_images);
	const char *backup_directory = option_find_str(options, "backup", "/backup/");

	Darknet::Network net_map;
	if (calc_map)
	{
		FILE* valid_file = fopen(valid_images, "r");
		if (!valid_file)
		{
			darknet_fatal_error(DARKNET_LOC, "There is no %s file for mAP calculation! Don't use -map flag. Or set valid=%s in %s.", valid_images, train_images, datacfg);
		}
		fclose(valid_file);

		cuda_set_device(gpus[0]);
		*cfg_and_state.output << "Prepare additional network for mAP calculation..." << std::endl;
		net_map = parse_network_cfg_custom(cfgfile, 1, 1);
		net_map.benchmark_layers = benchmark_layers;

		// free memory unnecessary arrays
		for (int k = 0; k < net_map.n - 1; ++k)
		{
			free_layer_custom(net_map.layers[k], 1);
		}
	}

	const char *base = basecfg(cfgfile);

	float avg_loss = -1.0f;
	float avg_contrastive_acc = 0.0f;

	// note we load a new network for every GPU used to train
	Darknet::Network * nets = (Darknet::Network*)xcalloc(ngpus, sizeof(Darknet::Network));
	for (int k = 0; k < ngpus; ++k)
	{
#ifdef DARKNET_GPU
		cuda_set_device(gpus[k]);
#endif
		nets[k] = parse_network_cfg(cfgfile);
		nets[k].benchmark_layers = benchmark_layers;
		if (weightfile)
		{
			load_weights(&nets[k], weightfile);
		}
		if (clear)
		{
			*nets[k].seen = 0;
			*nets[k].cur_iteration = 0;
		}

		/* 2025-04-12:  When training with multiple GPUs, this pushes the learning rate to the point where it destabilizes
		 * the network.  Loss quickly goes to NaN.  Based on what was seen at University of Florida with the "cars" dataset
		 * and the 4-GPU training rig, this line is getting commented out.
		 *
		nets[k].learning_rate *= ngpus;
		 */

		if (k == 0)
		{
			Darknet::load_names(&nets[k], option_find_str(options, "names", "unknown.names"));
		}
		else
		{
			nets[k].details->class_names = nets[0].details->class_names;
		}
	}

	Darknet::Network & net = nets[0];

	const std::string disabled = Darknet::format_in_colour("DISABLED", Darknet::EColour::kBrightRed, 0);
	if (calc_map)
	{
		net_map.details->class_names = net.details->class_names;
	}

	const int actual_batch_size = net.batch * net.subdivisions;
	if (actual_batch_size == 1)
	{
		darknet_fatal_error(DARKNET_LOC, "batch size should not be set to 1 for training");
	}
	else if (actual_batch_size < 32)
	{
		Darknet::display_warning_msg("Warning: batch=... is set quite low!  It is recommended to set batch=64.\n");
	}

	int imgs = net.batch * net.subdivisions * ngpus;

	*cfg_and_state.output
		<< "Learning Rate: "	<< net.learning_rate
		<< ", Momentum: "		<< net.momentum
		<< ", Decay: "			<< net.decay
		<< std::endl;

	data train;
	data buffer;

	Darknet::Layer l = net.layers[net.n - 1];
	for (int k = 0; k < net.n; ++k)
	{
		Darknet::Layer & lk = net.layers[k];
		if (lk.type == Darknet::ELayerType::YOLO or
			lk.type == Darknet::ELayerType::GAUSSIAN_YOLO or
			lk.type == Darknet::ELayerType::REGION)
		{
			l = lk;
			*cfg_and_state.output << "Detection layer #" << k << " is type " << static_cast<int>(l.type) << " (" << Darknet::to_string(l.type) << ")" << std::endl;
		}
	}

	int classes = l.classes;

	list *plist = get_paths(train_images);
	int train_images_num = plist->size;
	if (train_images_num == 0)
	{
		darknet_fatal_error(DARKNET_LOC, "no training images available (verify %s)", train_images);
	}
	if (train_images_num < actual_batch_size)
	{
		Darknet::display_warning_msg("Warning: there seems to be very few training images (num=" + std::to_string(train_images_num) + ", batch=" + std::to_string(actual_batch_size) + ")\n");
	}

	char **paths = (char **)list_to_array(plist);

	const int calc_map_for_each = fmax(100, train_images_num / (net.batch * net.subdivisions));  // calculate mAP for each epoch (used to be every 4 epochs)
	*cfg_and_state.output << "mAP calculations will be every " << calc_map_for_each << " iterations" << std::endl;

	// normally we save the weights every 10K, unless max batches is <= 10K in which case we save every 1K
	int how_often_we_save_weights = (net.max_batches <= 10000 ? 1000 : 10000);
	if (cfg_and_state.get("saveweights", 0) > 0)
	{
		// ...or, you can customize how often Darknet outputs the .weights file with the command-line parm "--save-weights 5000"
		how_often_we_save_weights = cfg_and_state.get_int("saveweights");
	}
	*cfg_and_state.output << "weights will be saved every " << how_often_we_save_weights << " iterations" << std::endl;

	const int init_w = net.w;
	const int init_h = net.h;
	const int init_b = net.batch;
	int iter_save		= get_current_iteration(net);
	int iter_save_last	= get_current_iteration(net);
	int iter_map		= get_current_iteration(net);
	int iter_best_map	= get_current_iteration(net);
	float mean_average_precision = -1.0f;
	float best_map = mean_average_precision;

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.c = net.c;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.flip = net.flip;
	args.jitter = l.jitter;
	args.resize = l.resize;
	args.num_boxes = l.max_boxes;
	args.truth_size = l.truth_size;
	net.num_boxes = args.num_boxes;
	net.train_images_num = train_images_num;
	args.d = &buffer;
	args.type = DETECTION_DATA; // this is the only place in the code where this type is used
	args.threads = 64;    // 16 or 64 -- see several lines below where this is set to 6 * GPUs

	args.angle = net.angle;
	args.gaussian_noise = net.gaussian_noise;
	args.blur = net.blur;
	args.mixup = net.mixup;
	args.exposure = net.exposure;
	args.saturation = net.saturation;
	args.hue = net.hue;
	args.letter_box = net.letter_box;
	args.mosaic_bound = net.mosaic_bound;
	args.contrastive = net.contrastive;
	args.contrastive_jit_flip = net.contrastive_jit_flip;
	args.contrastive_color = net.contrastive_color;
	if (dont_show and show_imgs)
	{
		show_imgs = 2;
	}
	args.show_imgs = show_imgs;
	args.threads = 6 * ngpus;   // 3 for - Amazon EC2 Tesla V100: p3.2xlarge (8 logical cores) - p3.16xlarge

	// This is where we draw the initial blank chart.  That chart is then updated by update_train_loss_chart() at every iteration.
	Darknet::initialize_new_charts(net);

	prime_training_images_cache(plist, net.w, net.h, net.c);

	if (net.contrastive and args.threads > net.batch/2)
	{
		args.threads = net.batch / 2;
	}

	if (net.track)
	{
		args.track = net.track;
		args.augment_speed = net.augment_speed;
		if (net.sequential_subdivisions)
		{
			args.threads = net.sequential_subdivisions * ngpus;
		}
		else
		{
			args.threads = net.subdivisions * ngpus;
		}
		args.mini_batch = net.batch / net.time_steps;
		*cfg_and_state.output
			<< std::endl
			<< "Tracking!"
			<< " batch=" << net.batch
			<< ", subdiv=" << net.subdivisions
			<< ", time_steps=" << net.time_steps
			<< ", mini_batch=" << args.mini_batch
			<< std::endl;
	}

	const auto first_iteration = get_current_iteration(net); // normally this is zero unless we're resuming training
	const auto start_of_training = std::chrono::high_resolution_clock::now();
	std::thread load_thread = std::thread(Darknet::run_image_loading_control_thread, args);
	int count = 0;

	// ***************************************
	// THIS is the start of the training loop!
	// ***************************************

	while (get_current_iteration(net) < net.max_batches and cfg_and_state.must_immediately_exit == false)
	{
		// we're starting a new iteration
		errno = 0;

#if 0 // useful when debugging to abort the training session
		if (get_current_iteration(net) >= 200)
		{
			break;
		}
#endif

		const std::chrono::high_resolution_clock::time_point iteration_start_time = std::chrono::high_resolution_clock::now();

		if (break_after_burn_in and get_current_iteration(net) == net.burn_in)
		{
			Darknet::display_warning_msg("\nRe-start training with multiple GPUs now that we've reached burn-in at iteration #" + std::to_string(net.burn_in) + ".\n\n");
			multi_gpu_weights_fn = backup_directory + std::string("/") + base + "_last.weights";
			save_weights(net, multi_gpu_weights_fn.c_str());
			break;
		}

		// yolov3-tiny, yolov3-tiny-3l, yolov3, and yolov4 all use "random=1"
		// yolov4-tiny and yolov4-tiny-3l both use "random=0"
		if (l.random and count++ % 10 == 0)
		{
			float rand_coef = 1.4;
			if (l.random != 1.0)
			{
				rand_coef = l.random;
			}
			float random_val = rand_scale(rand_coef);    // *x or /x
			int dim_w = roundl(random_val*init_w / net.resize_step + 1) * net.resize_step;
			int dim_h = roundl(random_val*init_h / net.resize_step + 1) * net.resize_step;
			if (random_val < 1 and (dim_w > init_w or dim_h > init_h))
			{
				dim_w = init_w, dim_h = init_h;
			}

			int max_dim_w = roundl(rand_coef*init_w / net.resize_step + 1) * net.resize_step;
			int max_dim_h = roundl(rand_coef*init_h / net.resize_step + 1) * net.resize_step;

			// at the beginning (check if enough memory) and at the end (calc rolling mean/variance)
			if (avg_loss < 0.0f or get_current_iteration(net) > net.max_batches - 100)
			{
				dim_w = max_dim_w;
				dim_h = max_dim_h;
			}

			if (dim_w < net.resize_step) dim_w = net.resize_step;
			if (dim_h < net.resize_step) dim_h = net.resize_step;
			int dim_b = (init_b * max_dim_w * max_dim_h) / (dim_w * dim_h);
			int new_dim_b = (int)(dim_b * 0.8);
			if (new_dim_b > init_b) dim_b = new_dim_b;

			args.w = dim_w;
			args.h = dim_h;

			if (net.dynamic_minibatch)
			{
				for (int k = 0; k < ngpus; ++k)
				{
					(*nets[k].seen) = init_b * net.subdivisions * get_current_iteration(net); // remove this line, when you will save to weights-file both: seen & cur_iteration
					nets[k].batch = dim_b;
					int j;
					for (j = 0; j < nets[k].n; ++j)
					{
						nets[k].layers[j].batch = dim_b;
					}
				}
				net.batch = dim_b;
				imgs = net.batch * net.subdivisions * ngpus;
				args.n = imgs;
			}

			*cfg_and_state.output
				<< "Resizing, random_coef=" << rand_coef
				<< ", batch=" << net.batch
				<< ", " << dim_w << "x" << dim_h
				<< std::endl;

			// discard what we had started loading, and re-start loading
			load_thread.join();
			train = buffer;
			Darknet::free_data(train);
			load_thread = std::thread(Darknet::run_image_loading_control_thread, args);

			for (int k = 0; k < ngpus; ++k)
			{
				resize_network(nets + k, dim_w, dim_h);
			}
			net = nets[0];
		} // random=1

		load_thread.join();
		train = buffer;

		if (net.track)
		{
			net.sequential_subdivisions = get_current_seq_subdivisions(net);
			args.threads = net.sequential_subdivisions * ngpus;
			*cfg_and_state.output
				<< "sequential_subdivisions=" << net.sequential_subdivisions
				<< ", sequence=" << get_sequence_value(net)
				<< std::endl;
		}

		load_thread = std::thread(Darknet::run_image_loading_control_thread, args);

		const auto train_start_time = std::chrono::high_resolution_clock::now();
		float loss = 0.0f;
#ifdef DARKNET_GPU
		if (ngpus == 1)
		{
			int wait_key = (dont_show) ? 0 : 1;
			loss = train_network_waitkey(net, train, wait_key);
		}
		else
		{
			loss = train_networks(nets, ngpus, train, 4);
		}
#else
		loss = train_network(net, train); // CPU-only
#endif

		if (avg_loss < 0.0f or avg_loss != avg_loss)
		{
			avg_loss = loss;    // if(-inf or nan)
		}
		avg_loss = avg_loss * 0.9f + loss * 0.1f;

		const auto iteration_end_time	= std::chrono::high_resolution_clock::now();
		const auto iteration_duration	= iteration_end_time - iteration_start_time;
		const auto train_end_time		= iteration_end_time;
		const auto train_duration		= train_end_time - train_start_time;
		const float elapsed_seconds		= std::chrono::duration_cast<std::chrono::seconds>(iteration_end_time - start_of_training).count();
		const int iteration				= get_current_iteration(net);
		const float current_iter		= iteration;
		const float iters_per_second	= (current_iter - first_iteration) / elapsed_seconds;
		const float seconds_remaining	= (net.max_batches - current_iter) / iters_per_second;
		const auto time_remaining		= Darknet::format_duration_string(std::chrono::seconds(static_cast<long>(seconds_remaining)), 1, Darknet::EFormatDuration::kTrim);
		const auto time_to_load_images	= std::chrono::nanoseconds(train.nanoseconds_to_load);

		if (time_to_load_images >= train_duration and avg_loss > 0.0f)
		{
			static size_t warning_counter = 0;
			warning_counter ++;
			static auto previous_performance_warning = iteration_end_time - std::chrono::seconds(60);
			if (iteration_end_time > previous_performance_warning + std::chrono::seconds(5))
			{
				previous_performance_warning = iteration_end_time;
				Darknet::display_warning_msg("Performance bottleneck (#" + std::to_string(warning_counter) + "): loading " + std::to_string(args.n) + " images took longer than it takes to train.  Slow CPU or hard drive?  Loading images from a network share?\n");
			}
		}

		// updating the console titlebar requires some ANSI/VT100 escape codes, so only do this if colour is also enabled
		if (cfg_and_state.colour_is_enabled)
		{
			if (std::isfinite(mean_average_precision) and mean_average_precision > 0.0f)
			{
				*cfg_and_state.output
					<< "\033]2;"
					<< iteration << "/" << net.max_batches
					<< ": loss=" << std::setprecision(1) << loss
					<< " map=" << std::setprecision(2) << mean_average_precision
					<< " best=" << std::setprecision(2) << best_map
					<< " time=" << time_remaining
					<< "\007";
			}
			else
			{
				*cfg_and_state.output
					<< "\033]2;"
					<< iteration << "/" << net.max_batches
					<< ": loss=" << std::setprecision(1) << loss
					<< " time=" << time_remaining
					<< "\007";
			}
		}

		if (cfg_and_state.is_verbose	and
			net.cudnn_half				and
			iteration < net.burn_in * 3)
		{
			*cfg_and_state.output << "Tensor cores are disabled until iteration #" << (3 * net.burn_in) << "." << std::endl;
		}

		const int next_map_calc = fmax(net.burn_in, iter_map + calc_map_for_each);

		// 5989: loss=0.444, avg loss=0.329, rate=0.000026, 64.424 milliseconds, 383296 images, time remaining=7 seconds
		*cfg_and_state.output
			<< Darknet::in_colour(Darknet::EColour::kBrightWhite, iteration)
			<< ": loss=" << Darknet::format_loss(loss)
			<< ", avg loss=" << Darknet::format_loss(avg_loss)
			<< ", last=" << Darknet::format_map_accuracy(mean_average_precision)
			<< ", best=" << Darknet::format_map_accuracy(best_map)
			<< ", mAP=" << (calc_map ? std::to_string(next_map_calc) : disabled)
			<< ", rate=" << std::setprecision(8) << get_current_rate(net) << std::setprecision(2)
			<< ", load " << args.n << "=" << Darknet::format_duration_string(time_to_load_images, 1)
			<< ", train=" << Darknet::format_duration_string(train_duration, 1)
//			<< ", iter=" << Darknet::format_duration_string(iteration_duration, 1)
			<< ", " << iteration * imgs << " images"
			<< ", time remaining=" << time_remaining
			<< std::endl;

		// This is where we decide if we have to do the mAP% calculations.
		if (calc_map and (iteration >= next_map_calc or iteration == net.max_batches))
		{
			if (l.random)
			{
				*cfg_and_state.output << "Resizing to initial size: " << init_w << " x " << init_h << std::endl;
				args.w = init_w;
				args.h = init_h;
				if (net.dynamic_minibatch)
				{
					for (int k = 0; k < ngpus; ++k)
					{
						nets[k].batch = init_b;
						for (int j = 0; j < nets[k].n; ++j)
						{
							nets[k].layers[j].batch = init_b;
						}
					}
					net.batch = init_b;
					imgs = init_b * net.subdivisions * ngpus;
					args.n = imgs;
					*cfg_and_state.output << init_w << " x " << init_h << " (batch=" << init_b << ")" << std::endl;
				}

				// discard the next set of images we began loading and re-start the loading process
				load_thread.join();
				Darknet::free_data(train);
				train = buffer;
				load_thread = std::thread(Darknet::run_image_loading_control_thread, args);

				for (int k = 0; k < ngpus; ++k)
				{
					resize_network(nets + k, init_w, init_h);
				}
				net = nets[0];
			}

			/// @todo copy the weights...?
			copy_weights_net(net, &net_map);

			iter_map = iteration;
			mean_average_precision = validate_detector_map(datacfg, cfgfile, weightfile, thresh, iou_thresh, 0, net.letter_box, &net_map);
			if (mean_average_precision >= best_map)
			{
				iter_best_map = iteration;
				best_map = mean_average_precision;
				*cfg_and_state.output << "New best mAP, saving weights!" << std::endl;
				char buff[256];
				sprintf(buff, "%s/%s_best.weights", backup_directory, base);
				save_weights(net, buff);
			}

			Darknet::update_accuracy_in_new_charts(-1, mean_average_precision);
			// done doing mAP% calculation
		}

		if (net.contrastive)
		{
			float cur_con_acc = -1;
			for (int k = 0; k < net.n; ++k)
			{
				if (net.layers[k].type == Darknet::ELayerType::CONTRASTIVE)
				{
					cur_con_acc = *net.layers[k].loss;
				}
			}
			if (cur_con_acc >= 0) avg_contrastive_acc = avg_contrastive_acc*0.99 + cur_con_acc * 0.01;
			*cfg_and_state.output << "average contrastive acc=" << avg_contrastive_acc << std::endl;
		}

		// this is where we draw the chart while training
		Darknet::update_loss_in_new_charts(iteration, avg_loss, time_remaining, dont_show);

		if (iteration >= iter_save + how_often_we_save_weights or (iteration % how_often_we_save_weights) == 0)
		{
			iter_save = iteration;
#ifdef DARKNET_GPU
			if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, iteration);
			save_weights(net, buff);
		}

		if (iteration >= (iter_save_last + 100) or (iteration % 100 == 0 and iteration > 1))
		{
			iter_save_last = iteration;
#ifdef DARKNET_GPU
			if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_last.weights", backup_directory, base);
			save_weights(net, buff);

			if (net.ema_alpha and is_ema_initialized(net))
			{
				sprintf(buff, "%s/%s_ema.weights", backup_directory, base);
				save_weights_upto(net, buff, net.n, 1);
				*cfg_and_state.output << "EMA weights are saved to " << buff << std::endl;
			}
		}
		Darknet::free_data(train);

	} // end of training loop

	if (break_after_burn_in == false)
	{
#ifdef DARKNET_GPU
		if (ngpus != 1)
		{
			sync_nets(nets, ngpus, 0);
		}
#endif
		char buff[256];
		sprintf(buff, "%s/%s_final.weights", backup_directory, base);
		save_weights(net, buff);

		if (mean_average_precision > 0.0f or best_map > 0.0f)
		{
			*cfg_and_state.output
				<< std::endl
				<< "Last accuracy mAP@" << std::setprecision(2) << iou_thresh
				<< "="			<< Darknet::format_map_accuracy(mean_average_precision)
				<< ", best="	<< Darknet::format_map_accuracy(best_map)
				<< " at iteration #" << iter_best_map << "."
				<< std::endl;
		}

		*cfg_and_state.output															<< std::endl
			<< Darknet::in_colour(Darknet::EColour::kBrightWhite)
			<< "Training iteration has reached max batch limit of "
			<< Darknet::in_colour(Darknet::EColour::kBrightGreen, net.max_batches)
			<< Darknet::in_colour(Darknet::EColour::kBrightWhite)
			<< ".  If you want"															<< std::endl
			<< "to restart training with these weights, either increase the limit, or"	<< std::endl
			<< "use the \""
			<< Darknet::in_colour(Darknet::EColour::kYellow, "-clear")
			<< Darknet::in_colour(Darknet::EColour::kBrightWhite)
			<< "\" flag to reset the training images counter to zero."					<< std::endl
			<< Darknet::in_colour(Darknet::EColour::kNormal)							<< std::endl;

		cv::destroyAllWindows();
	}

	// free memory
	load_thread.join();
	Darknet::free_data(buffer);

	Darknet::stop_image_loading_threads();

	free((void*)base);
	free(paths);
	free_list_contents(plist);
	free_list(plist);

	free_list_contents_kvp(options);
	free_list(options);

	for (int k = 0; k < ngpus; ++k)
	{
		free_network(nets[k]);
	}
	free(nets);

	if (calc_map)
	{
		net_map.n = 0;
		free_network(net_map);
	}

	return;
}


void train_detector(const char * datacfg, const char * cfgfile, const char * weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, float thresh, float iou_thresh, int show_imgs, int benchmark_layers, const char * chart_path)
{
	TAT(TATPARMS);

	const char * weights_fn_ptr = nullptr;
	std::string weights_fn_str;
	if (weightfile != nullptr)
	{
		weights_fn_str = Darknet::trim(weightfile);
		weights_fn_ptr = weights_fn_str.c_str();
	}

	// if we have multiple GPUs, then we may need to run on a single GPU for the burn-in period before we enable the rest of the GPUs
	if (ngpus > 1 and weights_fn_str.empty())
	{
		Darknet::display_warning_msg("\nTraining GPUs modified from " + std::to_string(ngpus) + " down to 1 until burn-in.\n\n");

		train_detector_internal(true, weights_fn_str, datacfg, cfgfile, weights_fn_ptr, gpus, 1, clear, dont_show, calc_map, thresh, iou_thresh, show_imgs, benchmark_layers, chart_path);
		weights_fn_ptr = weights_fn_str.c_str();
		clear = 0;
	}

	train_detector_internal(false, weights_fn_str, datacfg, cfgfile, weights_fn_ptr, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh, show_imgs, benchmark_layers, chart_path);

	return;
}


static void print_cocos(FILE *fp, char *image_path, Darknet::Detection * dets, int num_boxes, int classes, int w, int h)
{
	TAT(TATPARMS);

	int i, j;
	const char *p = basecfg(image_path);
	int image_id = atoi(p);
	for (i = 0; i < num_boxes; ++i)
	{
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		float bx = xmin;
		float by = ymin;
		float bw = xmax - xmin;
		float bh = ymax - ymin;

		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j] > 0)
			{
				char buff[1024];
				sprintf(buff, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
				fprintf(fp, "%s", buff);
			}
		}
	}
}

void print_detector_detections(FILE **fps, const char *id, Darknet::Detection * dets, int total, int classes, int w, int h)
{
	TAT(TATPARMS);

	int i, j;
	for (i = 0; i < total; ++i) {
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

		if (xmin < 1) xmin = 1;
		if (ymin < 1) ymin = 1;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j])
			{
				fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j], xmin, ymin, xmax, ymax);
			}
		}
	}
}

void print_imagenet_detections(FILE *fp, int id, Darknet::Detection * dets, int total, int classes, int w, int h)
{
	TAT(TATPARMS);

	int i, j;
	for (i = 0; i < total; ++i)
	{
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j)
		{
			int myclass = j;
			if (dets[i].prob[myclass] > 0)
			{
				fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[myclass], xmin, ymin, xmax, ymax);
			}
		}
	}
}

static void print_kitti_detections(FILE **fps, const char *id, Darknet::Detection * dets, int total, int classes, int w, int h, const char *outfile, const char *prefix)
{
	TAT(TATPARMS);

	const char *kitti_ids[] = { "car", "pedestrian", "cyclist" };
	FILE *fpd = 0;
	char buffd[1024];
	snprintf(buffd, 1024, "%s/%s/data/%s.txt", prefix, outfile, id);

	fpd = fopen(buffd, "w");
	int i, j;
	for (i = 0; i < total; ++i)
	{
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j])
			{
				fprintf(fpd, "%s -1 -1 -10 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 %f\n", kitti_ids[j], xmin, ymin, xmax, ymax, dets[i].prob[j]);
			}
		}
	}
	fclose(fpd);
}

static void eliminate_bdd(char *buf, const char *a)
{
	TAT(TATPARMS);

	int n = 0;
	int i, k;
	for (i = 0; buf[i] != '\0'; i++)
	{
		if (buf[i] == a[n])
		{
			k = i;
			while (buf[i] == a[n])
			{
				if (a[++n] == '\0')
				{
					for (; buf[k + n] != '\0'; k++)
					{
						buf[k] = buf[k + n];
					}
					buf[k] = '\0';
					break;
				}
				i++;
			}
			n = 0; i--;
		}
	}
}

static void get_bdd_image_id(char *filename)
{
	TAT(TATPARMS);

	char *p = strrchr(filename, '/');
	eliminate_bdd(p, ".jpg");
	eliminate_bdd(p, "/");
	strcpy(filename, p);
}

static void print_bdd_detections(FILE *fp, char *image_path, Darknet::Detection * dets, int num_boxes, int classes, int w, int h)
{
	TAT(TATPARMS);

	const char *bdd_ids[] = { "bike" , "bus" , "car" , "motor" ,"person", "rider", "traffic light", "traffic sign", "train", "truck" };
	get_bdd_image_id(image_path);
	int i, j;

	for (i = 0; i < num_boxes; ++i)
	{
		float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
		float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
		float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
		float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

		if (xmin < 0) xmin = 0;
		if (ymin < 0) ymin = 0;
		if (xmax > w) xmax = w;
		if (ymax > h) ymax = h;

		float bx1 = xmin;
		float by1 = ymin;
		float bx2 = xmax;
		float by2 = ymax;

		for (j = 0; j < classes; ++j)
		{
			if (dets[i].prob[j])
			{
				fprintf(fp, "\t{\n\t\t\"name\":\"%s\",\n\t\t\"category\":\"%s\",\n\t\t\"bbox\":[%f, %f, %f, %f],\n\t\t\"score\":%f\n\t},\n", image_path, bdd_ids[j], bx1, by1, bx2, by2, dets[i].prob[j]);
			}
		}
	}
}

void validate_detector(char *datacfg, char *cfgfile, char *weightfile, const char *outfile)
{
	TAT(TATPARMS);

	int j;
	list *options = read_data_cfg(datacfg);
	const char *valid_images = option_find_str(options, "valid", nullptr);
	const char *prefix = option_find_str(options, "results", "results");
	const char *mapf = option_find_str(options, "map", 0);
	int *map = 0;
	if (mapf) map = read_map(mapf);

	Darknet::Network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
	if (weightfile)
	{
		load_weights(&net, weightfile);
	}
	//set_batch_network(&net, 1);
	fuse_conv_batchnorm(net);
	calculate_binary_weights(&net);
	*cfg_and_state.output
		<< "Learning Rate: "	<< net.learning_rate
		<< ", Momentum: "		<< net.momentum
		<< ", Decay: "			<< net.decay
		<< std::endl;

	Darknet::load_names(&net, option_find_str(options, "names", "unknown.names"));

	list *plist = get_paths(valid_images);
	char **paths = (char **)list_to_array(plist);

	Darknet::Layer l = net.layers[net.n - 1];
	for (int k = 0; k < net.n; ++k)
	{
		Darknet::Layer & lk = net.layers[k];
		if (lk.type == Darknet::ELayerType::YOLO or
			lk.type == Darknet::ELayerType::GAUSSIAN_YOLO or
			lk.type == Darknet::ELayerType::REGION)
		{
			l = lk;
			*cfg_and_state.output << "Detection layer #" << k << " is type " << static_cast<int>(l.type) << " (" << Darknet::to_string(l.type) << ")" << std::endl;
		}
	}
	int classes = l.classes;

	char buff[1024];
	const char *type = option_find_str(options, "eval", "voc");
	FILE *fp = 0;
	FILE **fps = 0;
	int coco = 0;
	int imagenet = 0;
	int bdd = 0;
	int kitti = 0;

	if (0 == strcmp(type, "coco"))
	{
		if (!outfile) outfile = "coco_results";
		snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
		fp = fopen(buff, "w");
		fprintf(fp, "[\n");
		coco = 1;
	}
	else if (0 == strcmp(type, "bdd"))
	{
		if (!outfile) outfile = "bdd_results";
		snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
		fp = fopen(buff, "w");
		fprintf(fp, "[\n");
		bdd = 1;
	}
	else if (0 == strcmp(type, "kitti"))
	{
		char buff2[1024];
		if (!outfile) outfile = "kitti_results";
		*cfg_and_state.output << outfile << std::endl;
		snprintf(buff, 1024, "%s/%s", prefix, outfile);
		/* int mkd = */ make_directory(buff, 0777);
		snprintf(buff2, 1024, "%s/%s/data", prefix, outfile);
		/*int mkd2 = */ make_directory(buff2, 0777);
		kitti = 1;
	}
	else if (0 == strcmp(type, "imagenet"))
	{
		if (!outfile) outfile = "imagenet-detection";
		snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
		fp = fopen(buff, "w");
		imagenet = 1;
		classes = 200;
	}
	else
	{
		if (!outfile) outfile = "comp4_det_test_";
		fps = (FILE**) xcalloc(classes, sizeof(FILE *));
		for (j = 0; j < classes; ++j)
		{
			snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, net.details->class_names[j].c_str());
			fps[j] = fopen(buff, "w");
		}
	}

	int m = plist->size;
	int i = 0;
	int t;

	float thresh = .001;
	float nms = .6;

	int nthreads = 4;
	if (m < 4) nthreads = m;
	Darknet::Image* val = (Darknet::Image*)xcalloc(nthreads, sizeof(Darknet::Image));
	Darknet::Image* val_resized = (Darknet::Image*)xcalloc(nthreads, sizeof(Darknet::Image));
	Darknet::Image* buf = (Darknet::Image*)xcalloc(nthreads, sizeof(Darknet::Image));
	Darknet::Image* buf_resized = (Darknet::Image*)xcalloc(nthreads, sizeof(Darknet::Image));

	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;
	args.c = net.c;
	args.type = IMAGE_DATA;
	const int letter_box = net.letter_box;
	if (letter_box) args.type = LETTERBOX_DATA;

	Darknet::VThreads thr;
	thr.reserve(nthreads);
	for (t = 0; t < nthreads; ++t)
	{
		args.path = paths[i + t];
		args.im = &buf[t];
		args.resized = &buf_resized[t];
		thr.emplace_back(Darknet::load_single_image_data, args);
		cfg_and_state.set_thread_name(thr.back(), "validate loading thread #" + std::to_string(t));
	}
	time_t start = time(0);
	for (i = nthreads; i < m + nthreads; i += nthreads)
	{
		*cfg_and_state.output << i << std::endl;
		for (t = 0; t < nthreads and i + t - nthreads < m; ++t)
		{
			thr[t].join();
			cfg_and_state.del_thread_name(thr[t]);
			val[t] = buf[t];
			val_resized[t] = buf_resized[t];
		}
		for (t = 0; t < nthreads and i + t < m; ++t)
		{
			args.path = paths[i + t];
			args.im = &buf[t];
			args.resized = &buf_resized[t];
			thr[t] = std::thread(Darknet::load_single_image_data, args);
			cfg_and_state.set_thread_name(thr.back(), "validate loading thread #" + std::to_string(t));
		}
		for (t = 0; t < nthreads and i + t - nthreads < m; ++t)
		{
			char *path = paths[i + t - nthreads];
			const char *id = basecfg(path);
			float *X = val_resized[t].data;
			network_predict(net, X);
			int w = val[t].w;
			int h = val[t].h;
			int nboxes = 0;
			Darknet::Detection * dets = get_network_boxes(&net, w, h, thresh, .5, map, 0, &nboxes, letter_box);
			if (nms)
			{
				if (l.nms_kind == DEFAULT_NMS)
				{
					do_nms_sort(dets, nboxes, l.classes, nms);
				}
				else
				{
					diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
				}
			}

			if (coco)
			{
				print_cocos(fp, path, dets, nboxes, classes, w, h);
			}
			else if (imagenet)
			{
				print_imagenet_detections(fp, i + t - nthreads + 1, dets, nboxes, classes, w, h);
			}
			else if (bdd)
			{
				print_bdd_detections(fp, path, dets, nboxes, classes, w, h);
			}
			else if (kitti)
			{
				print_kitti_detections(fps, id, dets, nboxes, classes, w, h, outfile, prefix);
			}
			else
			{
				print_detector_detections(fps, id, dets, nboxes, classes, w, h);
			}

			free_detections(dets, nboxes);
			free((void*)id);
			Darknet::free_image(val[t]);
			Darknet::free_image(val_resized[t]);
		}
	}
	if (fps)
	{
		for (j = 0; j < classes; ++j)
		{
			fclose(fps[j]);
		}
		free(fps);
	}

	if (coco)
	{
#ifdef WIN32
		fseek(fp, -3, SEEK_CUR);
#else
		fseek(fp, -2, SEEK_CUR);
#endif
		fprintf(fp, "\n]\n");
	}

	if (bdd)
	{
#ifdef WIN32
		fseek(fp, -3, SEEK_CUR);
#else
		fseek(fp, -2, SEEK_CUR);
#endif
		fprintf(fp, "\n]\n");
	}

	if (fp) fclose(fp);

	if (val) free(val);
	if (val_resized) free(val_resized);
	if (buf) free(buf);
	if (buf_resized) free(buf_resized);

	*cfg_and_state.output << "Total detection time: " << (time(0) - start) << " seconds" << std::endl;
}

void validate_detector_recall(char *datacfg, char *cfgfile, char *weightfile)
{
	TAT(TATPARMS);

	Darknet::Network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
	if (weightfile)
	{
		load_weights(&net, weightfile);
	}
	//set_batch_network(&net, 1);
	fuse_conv_batchnorm(net);

	//list *plist = get_paths("data/coco_val_5k.list");
	list *options = read_data_cfg(datacfg);
	const char *valid_images = option_find_str(options, "valid", "data/train.txt");
	list *plist = get_paths(valid_images);
	char **paths = (char **)list_to_array(plist);

	//layer l = net.layers[net.n - 1];

	int j, k;

	int m = plist->size;
	int i = 0;

	float thresh = .001;
	float iou_thresh = .5;
	float nms = .4;

	int total = 0;
	int correct = 0;
	int proposals = 0;
	float avg_iou = 0;

	for (i = 0; i < m; ++i)
	{
		char *path = paths[i];
		Darknet::Image orig = Darknet::load_image(path, 0, 0, net.c);
		Darknet::Image sized = Darknet::resize_image(orig, net.w, net.h);
		const char *id = basecfg(path);
		network_predict(net, sized.data);
		int nboxes = 0;
		int letterbox = 0;
		Darknet::Detection * dets = get_network_boxes(&net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes, letterbox);
		if (nms) do_nms_obj(dets, nboxes, 1, nms);

		char labelpath[4096];
		replace_image_to_label(path, labelpath);

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		for (k = 0; k < nboxes; ++k) {
			if (dets[k].objectness > thresh) {
				++proposals;
			}
		}
		for (j = 0; j < num_labels; ++j)
		{
			++total;
			Darknet::Box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
			float best_iou = 0;
			for (k = 0; k < nboxes; ++k)
			{
				float iou = box_iou(dets[k].bbox, t);
				if (dets[k].objectness > thresh and iou > best_iou)
				{
					best_iou = iou;
				}
			}
			avg_iou += best_iou;
			if (best_iou > iou_thresh)
			{
				++correct;
			}
		}

		*cfg_and_state.output
			<< i << " " << correct << " " << total
			<< "\tRPs/Img: "	<< (float)proposals / (i + 1.0f)
			<< "\tIOU: "		<< 100.0f * avg_iou / total << "%"
			<< "\tRecall: "		<< 100.0f * correct / total << "%"
			<< std::endl;

		free(truth);
		free((void*)id);
		Darknet::free_image(orig);
		Darknet::free_image(sized);
	}
}


typedef struct {
	float w, h;
} anchors_t;

int anchors_comparator(const void *pa, const void *pb)
{
	TAT(TATPARMS);

	anchors_t a = *(const anchors_t *)pa;
	anchors_t b = *(const anchors_t *)pb;
	float diff = b.w * b.h - a.w * a.h;
	if (diff < 0)
	{
		return 1;
	}
	else if (diff > 0)
	{
		return -1;
	}
	return 0;
}

int anchors_data_comparator(const float **pa, const float **pb)
{
	TAT(TATPARMS);

	float *a = (float *)*pa;
	float *b = (float *)*pb;
	float diff = b[0] * b[1] - a[0] * a[1];
	if (diff < 0)
	{
		return 1;
	}
	else if (diff > 0)
	{
		return -1;
	}
	return 0;
}


void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
	TAT(TATPARMS);

	/// @todo shouldn't we check the .cfg file instead, and get the anchors, width, and height from there instead of requiring them as parms?

	*cfg_and_state.output
		<< "Recalculating anchors"
		<< ", num_of_clusters="	<< num_of_clusters
		<< ", width="			<< width
		<< ", height="			<< height
		<< std::endl;

	if (width < 32 or height < 32 or num_of_clusters <= 0)
	{
		*cfg_and_state.output << std::endl << "Usage example: darknet detector calc_anchors animals.data -num_of_clusters 6 -width 320 -height 256" << std::endl;
		darknet_fatal_error(DARKNET_LOC, "missing or invalid parameter required to recalculate YOLO anchors");
	}

	if ((width % 32) or (height % 32))
	{
		darknet_fatal_error(DARKNET_LOC, "cannot recalculate anchors due to invalid network dimensions (must be divisible by 32)");
	}

	//float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
	float* rel_width_height_array = (float*)xcalloc(1000, sizeof(float));

	list *options = read_data_cfg(datacfg);
	const char *train_images = option_find_str(options, "train", "data/train.list");
	list *plist = get_paths(train_images);
	int number_of_images = plist->size;
	char **paths = (char **)list_to_array(plist);

	int classes = option_find_int(options, "classes", 1);
	int* counter_per_class = (int*)xcalloc(classes, sizeof(int));

	int number_of_boxes = 0;
	*cfg_and_state.output << "read labels from " << number_of_images << " images" << std::endl;

	int i, j;
	for (i = 0; i < number_of_images; ++i)
	{
		char *path = paths[i];
		char labelpath[4096];
		replace_image_to_label(path, labelpath);

		int num_labels = 0;
		box_label *truth = read_boxes(labelpath, &num_labels);
		char *buff = (char*)xcalloc(6144, sizeof(char));
		for (j = 0; j < num_labels; ++j)
		{
			if (truth[j].x > 1 or truth[j].x <= 0 or truth[j].y > 1 or truth[j].y <= 0 or
				truth[j].w > 1 or truth[j].w <= 0 or truth[j].h > 1 or truth[j].h <= 0)
			{
				darknet_fatal_error(DARKNET_LOC, "invalid annotation coordinates or size (x=%f, y=%f, w=%f, h=%f) for class #%d in %s line #%d",
						truth[j].x, truth[j].y, truth[j].w, truth[j].h, truth[j].id, labelpath, j+1);
			}

			if (truth[j].id >= classes)
			{
				classes = truth[j].id + 1;
				counter_per_class = (int*)xrealloc(counter_per_class, classes * sizeof(int));
			}
			counter_per_class[truth[j].id]++;

			number_of_boxes++;
			rel_width_height_array = (float*)xrealloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));

			rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
			rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
			*cfg_and_state.output << "\rloaded image: " << i + 1 << " box: " << number_of_boxes << std::flush;
		}
		free(buff);
		free(truth);
	}

	*cfg_and_state.output
		<< std::endl
		<< "All loaded." << std::endl
		<< "Calculating k-means++ ..." << std::endl;

	matrix boxes_data;
	model anchors_data;
	boxes_data = make_matrix(number_of_boxes, 2);

	for (i = 0; i < number_of_boxes; ++i)
	{
		boxes_data.vals[i][0] = rel_width_height_array[i * 2];
		boxes_data.vals[i][1] = rel_width_height_array[i * 2 + 1];
	}

	// Is used: distance(box, centroid) = 1 - IoU(box, centroid)

	// K-means
	anchors_data = do_kmeans(boxes_data, num_of_clusters);

	/// @todo replace qsort() lowest priority
	qsort((void*)anchors_data.centers.vals, num_of_clusters, 2 * sizeof(float), (__compar_fn_t)anchors_data_comparator);

	float avg_iou = 0;
	for (i = 0; i < number_of_boxes; ++i)
	{
		float box_w = rel_width_height_array[i * 2]; //points->data.fl[i * 2];
		float box_h = rel_width_height_array[i * 2 + 1]; //points->data.fl[i * 2 + 1];
														//int cluster_idx = labels->data.i[i];
		int cluster_idx = 0;
		float min_dist = FLT_MAX;
		float best_iou = 0;
		for (j = 0; j < num_of_clusters; ++j)
		{
			float anchor_w = anchors_data.centers.vals[j][0];   // centers->data.fl[j * 2];
			float anchor_h = anchors_data.centers.vals[j][1];   // centers->data.fl[j * 2 + 1];
			float min_w = (box_w < anchor_w) ? box_w : anchor_w;
			float min_h = (box_h < anchor_h) ? box_h : anchor_h;
			float box_intersect = min_w*min_h;
			float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
			float iou = box_intersect / box_union;
			float distance = 1 - iou;
			if (distance < min_dist)
			{
				min_dist = distance;
				cluster_idx = j;
				best_iou = iou;
			}
		}

		float anchor_w = anchors_data.centers.vals[cluster_idx][0]; //centers->data.fl[cluster_idx * 2];
		float anchor_h = anchors_data.centers.vals[cluster_idx][1]; //centers->data.fl[cluster_idx * 2 + 1];
		if (best_iou > 1.0f or best_iou < 0.0f)
		{
			darknet_fatal_error(DARKNET_LOC, "wrong label: i=%d, box_w=%f, box_h=%f, anchor_w=%f, anchor_h=%f, iou=%f", i, box_w, box_h, anchor_w, anchor_h, best_iou);
		}

		avg_iou += best_iou;
	}

	char buff[1024];
	FILE* fwc = fopen("counters_per_class.txt", "wb");
	if (fwc)
	{
		sprintf(buff, "counters_per_class=");
		*cfg_and_state.output << buff;
		fwrite(buff, sizeof(char), strlen(buff), fwc);
		for (i = 0; i < classes; ++i)
		{
			sprintf(buff, "%d", counter_per_class[i]);
			*cfg_and_state.output << buff;
			fwrite(buff, sizeof(char), strlen(buff), fwc);
			if (i < classes - 1)
			{
				fwrite(", ", sizeof(char), 2, fwc);
				*cfg_and_state.output << ", ";
			}
		}
		*cfg_and_state.output << std::endl;
		fclose(fwc);
	}
	else
	{
		darknet_fatal_error(DARKNET_LOC, "Error: failed to open file counters_per_class.txt");
	}

	avg_iou = 100.0f * avg_iou / number_of_boxes;
	*cfg_and_state.output << "avg IoU=" << avg_iou << "%" << std::endl;

	FILE* fw = fopen("anchors.txt", "wb");
	if (fw)
	{
		*cfg_and_state.output
			<< "Saving anchors to the file: anchors.txt" << std::endl
			<< "anchors=";

		for (i = 0; i < num_of_clusters; ++i)
		{
			float anchor_w = anchors_data.centers.vals[i][0]; //centers->data.fl[i * 2];
			float anchor_h = anchors_data.centers.vals[i][1]; //centers->data.fl[i * 2 + 1];
			if (width > 32)
			{
				sprintf(buff, "%d, %d", (int)anchor_w, (int)anchor_h);
			}
			else
			{
				sprintf(buff, "%2.4f,%2.4f", anchor_w, anchor_h);
			}

			*cfg_and_state.output << buff;

			fwrite(buff, sizeof(char), strlen(buff), fw);
			if (i + 1 < num_of_clusters)
			{
				fwrite(", ", sizeof(char), 2, fw);
				*cfg_and_state.output << ", ";
			}
		}
		*cfg_and_state.output << std::endl;
		fclose(fw);
	}
	else
	{
		darknet_fatal_error(DARKNET_LOC, "Error: failed to open anchors.txt");
	}

	if (show)
	{
		show_anchors(number_of_boxes, num_of_clusters, rel_width_height_array, anchors_data, width, height);
	}
	free(rel_width_height_array);
	free(counter_per_class);
}


void test_detector(const char *datacfg, const char *cfgfile, const char *weightfile, const char *filename, float thresh,
	float hier_thresh, int dont_show, int ext_output, int save_labels, const char *outfile, int letter_box, int benchmark_layers)
{
	TAT(TATPARMS);

	list *options = read_data_cfg(datacfg);

	Darknet::Network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
	if (weightfile)
	{
		load_weights(&net, weightfile);
	}
	if (net.letter_box)
	{
		letter_box = 1;
	}
	net.benchmark_layers = benchmark_layers;
	fuse_conv_batchnorm(net);
	calculate_binary_weights(&net);

	Darknet::load_names(&net, option_find_str(options, "names", "unknown.names"));

	char buff[256];
	char *input = buff;
	char *json_buf = NULL;
	int json_image_id = 0;
	FILE* json_file = NULL;
	if (outfile)
	{
		json_file = fopen(outfile, "wb");
		if (!json_file)
		{
			file_error(outfile, DARKNET_LOC);
		}
		const char *tmp = "[\n";
		fwrite(tmp, sizeof(char), strlen(tmp), json_file);
	}

	float nms = 0.45f;    // 0.4F
	while (1)
	{
		if (filename)
		{
			strncpy(input, filename, 256);
			if (strlen(input) > 0)
			{
				if (input[strlen(input) - 1] == 0x0d)
				{
					input[strlen(input) - 1] = 0;
				}
			}
		}
		else
		{
			printf("Enter Image Path: ");
			fflush(stdout);
			input = fgets(input, 256, stdin);
			if (!input)
			{
				break;
			}
			strtok(input, "\n");
		}
		//image im;
		//image sized = load_image_resize(input, net.w, net.h, net.c, &im);
		Darknet::Image im = Darknet::load_image(input, 0, 0, net.c);
		Darknet::Image sized;
		if (letter_box)
		{
			sized = Darknet::letterbox_image(im, net.w, net.h);
		}
		else
		{
			sized = Darknet::resize_image(im, net.w, net.h);
		}

		Darknet::Layer l = net.layers[net.n - 1];
		for (int k = 0; k < net.n; ++k)
		{
			Darknet::Layer & lk = net.layers[k];
			if (lk.type == Darknet::ELayerType::YOLO or
				lk.type == Darknet::ELayerType::GAUSSIAN_YOLO or
				lk.type == Darknet::ELayerType::REGION)
			{
				l = lk;
				*cfg_and_state.output << "Detection layer #" << k << " is type " << static_cast<int>(l.type) << " (" << Darknet::to_string(l.type) << ")" << std::endl;
			}
		}

		float *X = sized.data;

		network_predict(net, X);

		int nboxes = 0;
		Darknet::Detection * dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
		if (nms)
		{
			if (l.nms_kind == DEFAULT_NMS)
			{
				do_nms_sort(dets, nboxes, l.classes, nms);
			}
			else
			{
				diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
			}
		}

		// Load the image explicitly asking for 3 color channels
		Darknet::Image im_color = Darknet::load_image(input, 0, 0, 3);

		Darknet::draw_detections_v3(im_color, dets, nboxes, thresh, net.details->class_names, l.classes, ext_output);
		Darknet::save_image(im_color, "predictions");
		if (!dont_show)
		{
			Darknet::show_image(im_color, "predictions");
		}

		if (json_file)
		{
			if (json_buf)
			{
				const char *tmp = ", \n";
				fwrite(tmp, sizeof(char), strlen(tmp), json_file);
			}
			++json_image_id;
			json_buf = Darknet::detection_to_json(dets, nboxes, l.classes, net.details->class_names, json_image_id, input);

			fwrite(json_buf, sizeof(char), strlen(json_buf), json_file);
			free(json_buf);
		}

		// pseudo labeling concept - fast.ai
		if (save_labels)
		{
			char labelpath[4096];
			replace_image_to_label(input, labelpath);

			FILE* fw = fopen(labelpath, "wb");
			for (int i = 0; i < nboxes; ++i)
			{
				char tmp[1024];
				int class_id = -1;
				float prob = 0;
				for (int j = 0; j < l.classes; ++j)
				{
					if (dets[i].prob[j] > thresh and dets[i].prob[j] > prob)
					{
						prob = dets[i].prob[j];
						class_id = j;
					}
				}
				if (class_id >= 0)
				{
					sprintf(tmp, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
					fwrite(tmp, sizeof(char), strlen(tmp), fw);
				}
			}
			fclose(fw);
		}

		free_detections(dets, nboxes);
		Darknet::free_image(im_color);
		Darknet::free_image(im);
		Darknet::free_image(sized);

		if (!dont_show)
		{
			cv::waitKey(0);
			cv::destroyAllWindows();
		}

		if (filename) break;
	}

	if (json_file)
	{
		const char *tmp = "\n]";
		fwrite(tmp, sizeof(char), strlen(tmp), json_file);
		fclose(json_file);
	}

	// free memory
	free_list_contents_kvp(options);
	free_list(options);

	free_network(net);
}


void run_detector(int argc, char **argv)
{
	TAT(TATPARMS);

	int benchmark = find_arg(argc, argv, "-benchmark");
	int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
	int letter_box = find_arg(argc, argv, "-letter_box");
	int map_points = find_int_arg(argc, argv, "-points", 0);
	int show_imgs = find_arg(argc, argv, "-show_imgs");
//	int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
//	int avgframes = find_int_arg(argc, argv, "-avgframes", 3);
//	int dontdraw_bbox = find_arg(argc, argv, "-dontdraw_bbox");
//	int json_port = find_int_arg(argc, argv, "-json_port", -1);
//	char *http_post_host = find_char_arg(argc, argv, "-http_post_host", 0);
//	int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
//	char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
	const char *outfile = find_char_arg(argc, argv, "-out", 0);
//	char *prefix = find_char_arg(argc, argv, "-prefix", 0);
	float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
	float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
	float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
//	int cam_index = find_int_arg(argc, argv, "-c", 0);
//	int frame_skip = find_int_arg(argc, argv, "-s", 0);
	// extended output in test mode (output of rect bound coords)
	// and for recall mode (extended output table-like format with results for best_class fit)
	int ext_output = find_arg(argc, argv, "-ext_output");
	int save_labels = find_arg(argc, argv, "-save_labels");
	const char* chart_path = find_char_arg(argc, argv, "-chart", 0);
	const char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
	int *gpus = 0;
	int gpu = 0;
	int ngpus = 0;
	if (gpu_list)
	{
		*cfg_and_state.output << gpu_list << std::endl;
		int len = (int)strlen(gpu_list);
		ngpus = 1;
		int i;
		for (i = 0; i < len; ++i)
		{
			if (gpu_list[i] == ',') ++ngpus;
		}
		gpus = (int*)xcalloc(ngpus, sizeof(int));
		for (i = 0; i < ngpus; ++i)
		{
			gpus[i] = atoi(gpu_list);
			gpu_list = strchr(gpu_list, ',') + 1;
		}
	}
	else
	{
		gpu = cfg_and_state.gpu_index;
		gpus = &gpu;
		ngpus = 1;
	}

	int clear		= cfg_and_state.is_set("clear"	)	? 1 : 0;
	int calc_map	= cfg_and_state.is_set("map"	)	? 1 : 0;
	int dont_show	= cfg_and_state.is_shown			? 0 : 1;
	if (benchmark) dont_show = 1;

	/// @todo get rid of the old C-style filename access and use std::filesystem::path within the functions so we're not passing these around as char* parms
	char * datacfg	= nullptr;
	char * cfg		= nullptr;
	char * weights	= nullptr;
	char * input_fn	= nullptr; // if we're passing in an image for example

	std::string fn1 = cfg_and_state.data_filename		.string();
	std::string fn2 = cfg_and_state.cfg_filename		.string();
	std::string fn3 = cfg_and_state.weights_filename	.string();
	std::string fn4 = (cfg_and_state.filenames.size() > 3 ? cfg_and_state.filenames[3] : "");
	if (fn4.empty() and not cfg_and_state.additional_arguments.empty())
	{
		fn4 = cfg_and_state.additional_arguments[0];
	}

	if (not fn1.empty())	{ datacfg	= const_cast<char*>(fn1.c_str()); }
	if (not fn2.empty())	{ cfg		= const_cast<char*>(fn2.c_str()); }
	if (not fn3.empty())	{ weights	= const_cast<char*>(fn3.c_str()); }
	if (not fn4.empty())	{ input_fn	= const_cast<char*>(fn4.c_str()); }

	if		(cfg_and_state.function == "test"		) { test_detector(datacfg, cfg, weights, input_fn, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers); }
	else if (cfg_and_state.function == "train"		) { train_detector(datacfg, cfg, weights, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh, show_imgs, benchmark_layers, chart_path); }
	else if (cfg_and_state.function == "valid"		) { validate_detector(datacfg, cfg, weights, outfile); }
	else if (cfg_and_state.function == "recall"		) { validate_detector_recall(datacfg, cfg, weights); }
	else if (cfg_and_state.function == "map"		) { validate_detector_map(datacfg, cfg, weights, thresh, iou_thresh, map_points, letter_box, NULL); }
	else if (cfg_and_state.function == "calcanchors")
	{
		const int show				= cfg_and_state.is_set	("show"			) ? 1 : 0;
		const int width				= cfg_and_state.get_int	("width"		);
		const int height			= cfg_and_state.get_int	("height"		);
		const int num_of_clusters	= cfg_and_state.get_int	("numofclusters");

		calc_anchors(datacfg, num_of_clusters, width, height, show);
	}
	else
	{
		darknet_fatal_error(DARKNET_LOC, "invalid Darknet command: %s %s", cfg_and_state.command.c_str(), cfg_and_state.function.c_str());
	}

	/// @todo why only do this if ngpus > 1?  Is this a bug?  Should it be zero?
	if (gpus and gpu_list and ngpus > 1)
	{
		free(gpus);
	}
}
