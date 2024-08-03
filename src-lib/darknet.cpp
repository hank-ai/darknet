#include "darknet_internal.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	// remember that OpenCV colours are BGR, not RGB
	static const auto		white									= cv::Scalar(255, 255, 255);
	static const auto		black									= cv::Scalar(0, 0, 0);

	static float			detection_threshold						= 0.25f;
	static float			non_maximal_suppression_threshold		= 0.45;
	static bool				fix_out_of_bound_normalized_coordinates	= true;
	static cv::LineTypes	cv_font_line_type						= cv::LineTypes::LINE_4;
	static cv::HersheyFonts	cv_font_face							= cv::HersheyFonts::FONT_HERSHEY_PLAIN;
	static int				cv_font_thickness						= 1;
	static double			cv_font_scale							= 1.0;
	static cv::Scalar		colour_bb_lines							= white;
	static cv::Scalar		colour_label_text						= black;
	static bool				annotate_draw_rounded_bb_toggle			= false;
	static float			annotate_draw_rounded_bb_roundness		= 0.5f;
	static bool				annotate_draw_bb						= true;
	static bool				annotate_draw_label						= true;

	// shamlessly stolen from DarkHelp
	static inline void fix_out_of_bound_normalized_rect(float & cx, float & cy, float & w, float & h)
	{
		// coordinates are all normalized!

		if (cx - w / 2.0f < 0.0f or	// too far left
			cx + w / 2.0f > 1.0f)	// too far right
		{
			// calculate a new X and width to use for this prediction
			const float new_x1 = std::max(0.0f, cx - w / 2.0f);
			const float new_x2 = std::min(1.0f, cx + w / 2.0f);
			const float new_w = new_x2 - new_x1;
			const float new_x = (new_x1 + new_x2) / 2.0f;
			cx = new_x;
			w = new_w;
		}

		if (cy - h / 2.0f < 0.0f or	// too far above
			cy + h / 2.0f > 1.0f)	// too far below
		{
			// calculate a new Y and height to use for this prediction
			const float new_y1 = std::max(0.0f, cy - h / 2.0f);
			const float new_y2 = std::min(1.0f, cy + h / 2.0f);
			const float new_h = new_y2 - new_y1;
			const float new_y = (new_y1 + new_y2) / 2.0f;
			cy = new_y;
			h = new_h;
		}

		return;
	}

	void draw_rounded_rectangle(cv::Mat & mat, const cv::Rect & r, const float roundness)
	{
		/* This is what decides how "round" the bounding box needs to be.  The divider
		 * decides the length of the line segments and the radius of each rounded corner.
		 *
		 * 0.0	-> 12
		 * 0.25	-> 9.5
		 * 0.5	-> 7
		 * 0.75	-> 4.5
		 * 1.0	-> 2
		 *
		 * so:  y = mx + b which gives us:  y = -10x + 12
		 */
		const float divider = std::clamp(-10.0f * roundness + 12.0f, 2.0f, 12.0f);

		const cv::Point tl(r.tl());
		const cv::Point br(r.br());
		const cv::Point tr(br.x, tl.y);
		const cv::Point bl(tl.x, br.y);

		// the radius of each corner
		const int hoffset = std::round((tr.x - tl.x) / divider);
		const int voffset = std::round((bl.y - tl.y) / divider);

		if (hoffset * 2 >= r.width and
			voffset * 2 >= r.height)
		{
			// corners are so big that we're actually drawing a circle (or an ellipse if the bb is not square)
			cv::ellipse(mat, cv::Point(r.x + r.width / 2, r.y + r.height / 2), r.size() / 2, 0.0, 0.0, 360.0, colour_bb_lines, 1, cv_font_line_type);
		}
		else
		{
			// draw horizontal and vertical segments
			cv::line(mat, cv::Point(tl.x + hoffset, tl.y), cv::Point(tr.x - hoffset, tr.y), colour_bb_lines, 1, cv_font_line_type);
			cv::line(mat, cv::Point(tr.x, tr.y + voffset), cv::Point(br.x, br.y - voffset), colour_bb_lines, 1, cv_font_line_type);
			cv::line(mat, cv::Point(br.x - hoffset, br.y), cv::Point(bl.x + hoffset, bl.y), colour_bb_lines, 1, cv_font_line_type);
			cv::line(mat, cv::Point(bl.x, bl.y - voffset), cv::Point(tl.x, tl.y + voffset), colour_bb_lines, 1, cv_font_line_type);

			cv::ellipse(mat, tl + cv::Point(+hoffset, +voffset), cv::Size(hoffset, voffset), 0.0, 180.0	, 270.0	, colour_bb_lines, 1, cv_font_line_type);
			cv::ellipse(mat, tr + cv::Point(-hoffset, +voffset), cv::Size(hoffset, voffset), 0.0, 270.0	, 360.0	, colour_bb_lines, 1, cv_font_line_type);
			cv::ellipse(mat, br + cv::Point(-hoffset, -voffset), cv::Size(hoffset, voffset), 0.0, 0.0	, 90.0	, colour_bb_lines, 1, cv_font_line_type);
			cv::ellipse(mat, bl + cv::Point(+hoffset, -voffset), cv::Size(hoffset, voffset), 0.0, 90.0	, 180.0	, colour_bb_lines, 1, cv_font_line_type);
		}

		return;
	}
}


extern "C"
{
	void darknet_set_verbose(const bool flag)
	{
		TAT(TATPARMS);
		Darknet::set_verbose(flag);
		return;
	}


	void darknet_set_trace(const bool flag)
	{
		TAT(TATPARMS);
		Darknet::set_trace(flag);
		return;
	}


	void darknet_set_gpu_index(int idx)
	{
		TAT(TATPARMS);
		Darknet::set_gpu_index(idx);
		return;
	}


	void darknet_set_detection_threshold(float threshold)
	{
		TAT(TATPARMS);
		Darknet::set_detection_threshold(threshold);
		return;
	}


	void darknet_set_non_maximal_suppression_threshold(float threshold)
	{
		TAT(TATPARMS);
		Darknet::set_non_maximal_suppression_threshold(threshold);
		return;
	}

	void darknet_fix_out_of_bound_values(const bool toggle)
	{
		TAT(TATPARMS);
		Darknet::fix_out_of_bound_values(toggle);
		return;
	}

	void darknet_network_dimensions(NetworkPtr ptr, int * w, int * h, int * c)
	{
		TAT(TATPARMS);

		int width = 0;
		int height = 0;
		int channels = 0;
		Darknet::network_dimensions(ptr, width, height, channels);

		if (w) *w = width;
		if (h) *h = height;
		if (c) *c = channels;

		return;
	}

	NetworkPtr darknet_load_neural_network(const char * const cfg_filename, const char * const names_filename, const char * const weights_filename)
	{
		TAT(TATPARMS);

		return Darknet::load_neural_network(std::filesystem::path(cfg_filename), std::filesystem::path(names_filename), std::filesystem::path(weights_filename));
	}
}


Darknet::Parms Darknet::parse_arguments(int argc, char * argv[])
{
	TAT(TATPARMS);

	// on purpose skip argv[0] which is the application name
	const VStr v(&argv[1], &argv[argc]);

	auto parms = parse_arguments(v);

	// fix up the indexes, since we started with argv[1] and not argv[0]
	for (auto & parm : parms)
	{
		parm.idx ++;
	}

	return parms;
}


Darknet::Parms Darknet::parse_arguments(const Darknet::VStr & v)
{
	TAT(TATPARMS);

	Darknet::Parms parms;
	parms.reserve(v.size());

	for (int idx = 0; idx < v.size(); idx ++)
	{
		Darknet::Parm parm;
		parm.idx		= idx;
		parm.type		= EParmType::kOther;
		parm.original	= v[idx];
		parm.string		= v[idx];

		parms.push_back(parm);
	}

	// 1st step:  see if we can identify the 3 files we need to load the network
	for (auto & parm : parms)
	{
		std::filesystem::path path(parm.original);
		if (not std::filesystem::exists(path))
		{
			continue;
		}

		// if we get here then the parm is some sort of valid path

		path = std::filesystem::canonical(path);
		parm.string = path.string();

		if (std::filesystem::is_directory(path))
		{
			// nothing we can do at this point with a directory,
			// so we identify it as such and move to the next parm
			parm.type = EParmType::kDirectory;
			continue;
		}

		// if we get here then we know we have some sort of file

		parm.type = EParmType::kFilename;
		const auto extension = path.extension();
		if (extension == ".cfg")
		{
			if (cfg_and_state.is_verbose)
			{
				std::cout << "Found configuration: " << Darknet::in_colour(Darknet::EColour::kBrightWhite, path.string()) << std::endl;
			}
			parm.type = EParmType::kCfgFilename;
		}
		else if (extension == ".names")
		{
			if (cfg_and_state.is_verbose)
			{
				std::cout << "Found names file:    " << Darknet::in_colour(Darknet::EColour::kBrightWhite, path.string()) << std::endl;
			}
			parm.type = EParmType::kNamesFilename;
		}
		else if (extension == ".weights")
		{
			if (cfg_and_state.is_verbose)
			{
				std::cout << "Found weights file:  " << Darknet::in_colour(Darknet::EColour::kBrightWhite, path.string()) << std::endl;
			}
			parm.type = EParmType::kWeightsFilename;
		}
	}

	// 2nd step:  if we have the .cfg then see if we can guess what the .names and .weights file might be called
	int cfg_idx		= -1;
	int names_idx	= -1;
	int weights_idx	= -1;

	// find the *first* parameter of each type
	for (int idx = 0; idx < parms.size(); idx ++)
	{
		auto & parm = parms[idx];

		if (parm.type == EParmType::kCfgFilename		and cfg_idx		== -1) cfg_idx		= idx;
		if (parm.type == EParmType::kNamesFilename		and names_idx	== -1) names_idx	= idx;
		if (parm.type == EParmType::kWeightsFilename	and weights_idx	== -1) weights_idx	= idx;
	}

	if (cfg_idx >= 0)
	{
		std::filesystem::path path = parms[cfg_idx].string;
		if (names_idx == -1)
		{
			path.replace_extension(".names");
			if (std::filesystem::exists(path))
			{
				std::cout << "Guessed names:       " << Darknet::in_colour(Darknet::EColour::kBrightGreen, path.string()) << std::endl;
				Parm parm = parms[cfg_idx];
				parm.type = EParmType::kNamesFilename;
				parm.string = path.string();
				parms.push_back(parm);
				names_idx = parms.size() - 1;
			}
		}

		if (weights_idx == -1)
		{
			path.replace_extension(".weights");
			if (std::filesystem::exists(path))
			{
				std::cout << "Guessed weights:     " << Darknet::in_colour(Darknet::EColour::kBrightGreen, path.string()) << std::endl;
				Parm parm = parms[cfg_idx];
				parm.type = EParmType::kWeightsFilename;
				parm.string = path.string();
				parms.push_back(parm);
				weights_idx = parms.size() - 1;
			}
			else
			{
				std::string tmp = path.string();
				auto pos = tmp.rfind(".weights");
				tmp.erase(pos);
				tmp += "_best.weights";
				if (std::filesystem::exists(tmp))
				{
					std::cout << "Guessed weights:     " << Darknet::in_colour(Darknet::EColour::kBrightGreen, tmp) << std::endl;
					Parm parm = parms[cfg_idx];
					parm.type = EParmType::kWeightsFilename;
					parm.string = tmp;
					parms.push_back(parm);
					weights_idx = parms.size() - 1;
				}
			}
		}
	}

	// 3rd step:  if we have the .cfg, and we're missing the .weights, but we have other possible filenames to use...
	if (cfg_idx > -1 and weights_idx == -1)
	{
		// the weights file might have an unusual extension?  look for a file > 10 MiB in size and peek at the header

		// 4-byte values * 3 = 12 bytes total
		const uint32_t expected_header[] = {DARKNET_WEIGHTS_VERSION_MAJOR, DARKNET_WEIGHTS_VERSION_MINOR, DARKNET_WEIGHTS_VERSION_PATCH};

		for (int idx = 0; idx < parms.size(); idx ++)
		{
			Parm & parm = parms[idx];
			if (parm.type != EParmType::kFilename)
			{
				continue;
			}

			// if we get here, we have an unknown filename type which might be weights

			const auto filename = parm.string;

			if (std::filesystem::file_size(filename) > 10 * 1024 * 1024) // at least 10 MiB in size
			{
				// read the first 12 bytes and see if it matches what we think it should be for a .weights file

				/// @todo confirm that this works just as well on ARM

				size_t header_bytes_matched = 0;
				std::ifstream ifs(filename, std::ifstream::binary | std::ifstream::in);
				for (size_t idx = 0; ifs.good() and idx < 3; idx ++)
				{
					uint32_t tmp = 0;
					ifs.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));

					if (tmp == expected_header[idx])
					{
						header_bytes_matched ++;
					}
				}

				if (header_bytes_matched == 3)
				{
					std::cout << "Found these weights: " << Darknet::in_colour(Darknet::EColour::kYellow, filename) << std::endl;
					parm.type = EParmType::kWeightsFilename;
					weights_idx = idx;
					break;
				}
			}
		}
	}

	// 4th step:  see if we were given the *stem* of the filenames we need
	// in which case we need to glob files and match a certain pattern
	if (cfg_idx		== -1 and
		names_idx	== -1 and
		weights_idx	== -1)
	{
		// one at a time, try each argument to see if we can find a file that starts with the same text
		for (int idx = 0; idx < parms.size(); idx ++)
		{
			auto & parm = parms[idx];

			if (parm.type != EParmType::kOther)
			{
				continue;
			}

			std::filesystem::path tmp = parm.string;
			std::filesystem::path parent = tmp.parent_path();
			if (parent.empty())
			{
				parent = std::filesystem::current_path();
			}
			std::string stem = tmp.filename().string();

			VStr matching_files;
			for (auto iter : std::filesystem::directory_iterator(parent))
			{
				const std::filesystem::path path = iter.path();
				if (path.filename().string().find(stem) == 0)
				{
					matching_files.push_back(path.string());
				}
			}

			std::sort(matching_files.begin(), matching_files.end());

			std::string backup_weights;

			for (const auto & filename : matching_files)
			{
				const auto extension = std::filesystem::path(filename).extension().string();
				if (extension == ".cfg" and cfg_idx == -1)
				{
					std::cout << tmp << " matches this config file:  " << Darknet::in_colour(Darknet::EColour::kBrightCyan, filename) << std::endl;
					parm.type = EParmType::kCfgFilename;
					parm.string = filename;
					cfg_idx = idx;
				}
				else if (extension == ".names" and names_idx == -1)
				{
					std::cout << tmp << " matches this names file:   " << Darknet::in_colour(Darknet::EColour::kBrightCyan, filename) << std::endl;
					Parm parm = parms[idx];
					parm.type = EParmType::kNamesFilename;
					parm.string = filename;
					parms.push_back(parm);
					names_idx = parms.size() - 1;
				}
				else if (extension == ".weights")
				{
					if (weights_idx == -1 and filename.find("_best.weights") != std::string::npos)
					{
						std::cout << tmp << " matches this weights file: " << Darknet::in_colour(Darknet::EColour::kBrightCyan, filename) << std::endl;
						Parm parm = parms[idx];
						parm.type = EParmType::kWeightsFilename;
						parm.string = filename;
						parms.push_back(parm);
						weights_idx = parms.size() - 1;
					}
					else
					{
						backup_weights = filename;
					}
				}
			}

			if (weights_idx == -1 and backup_weights.empty() == false)
			{
				// in case we don't find "best" weights, we'll end up here and use whatever weights we found
				std::cout << tmp << " matches this weights file: " << Darknet::in_colour(Darknet::EColour::kBrightWhite, backup_weights) << std::endl;
				Parm parm = parms[idx];
				parm.type = EParmType::kWeightsFilename;
				parm.string = backup_weights;
				parms.push_back(parm);
				weights_idx = parms.size() - 1;
			}
		}
	}

	return parms;
}


void Darknet::set_verbose(const bool flag)
{
	TAT(TATPARMS);

	cfg_and_state.is_verbose = flag;

	// when verbose is disabled, then disable trace as well
	if (not flag)
	{
		set_trace(flag);
	}

	return;
}


void Darknet::set_trace(const bool flag)
{
	TAT(TATPARMS);

	cfg_and_state.is_trace = flag;

	// when trace is enabled, then enable verbose as well
	if (flag)
	{
		set_verbose(flag);
	}

	return;
}


void Darknet::set_gpu_index(int idx)
{
	TAT(TATPARMS);

	#ifdef GPU
	cfg_and_state.gpu_index = idx;
	#else
	// don't allow the GPU index to be set when Darknet was not compiled with CUDA support
	cfg_and_state.gpu_index = -1;
	#endif

	return;
}


void Darknet::set_detection_threshold(float threshold)
{
	TAT(TATPARMS);

	if (threshold > 1.0f and threshold < 100.0f)
	{
		// user must be using percentages instead
		threshold /= 100.0f;
	}

	if (threshold >= 0.0f and threshold <= 1.0f)
	{
		detection_threshold = threshold;
		return;
	}

	throw std::invalid_argument("detection threshold must be between 0.0 and 1.0");
}


void Darknet::set_non_maximal_suppression_threshold(float threshold)
{
	TAT(TATPARMS);

	if (threshold > 1.0f and threshold < 100.0f)
	{
		// user must be using percentages instead
		threshold /= 100.0f;
	}
	if (threshold >= 0.0f and threshold <= 1.0f)
	{
		non_maximal_suppression_threshold = threshold;
		return;
	}
	throw std::invalid_argument("nms threshold must be between 0.0 and 1.0");
}


void Darknet::fix_out_of_bound_values(const bool toggle)
{
	TAT(TATPARMS);

	fix_out_of_bound_normalized_coordinates = toggle;

	return;
}


void Darknet::set_annotation_font(const cv::LineTypes line_type, const cv::HersheyFonts font_face, const int font_thickness, const double font_scale)
{
	TAT(TATPARMS);

	cv_font_line_type	= line_type;
	cv_font_face		= font_face;
	cv_font_thickness	= font_thickness;
	cv_font_scale		= font_scale;

	return;
}


void Darknet::set_annotation_bb_line_colour(const cv::Scalar colour)
{
	TAT(TATPARMS);

	colour_bb_lines = colour;

	return;
}


void Darknet::set_annotation_label_text_colour(const cv::Scalar colour)
{
	TAT(TATPARMS);

	colour_label_text = colour;

	return;
}


void Darknet::set_annotation_draw_rounded_bb(const bool rounded, const float roundness)
{
	TAT(TATPARMS);

	annotate_draw_rounded_bb_toggle		= rounded;
	annotate_draw_rounded_bb_roundness	= roundness;

	return;
}


void Darknet::set_annotation_draw_bb(const bool draw)
{
	TAT(TATPARMS);

	annotate_draw_bb = draw;

	return;
}


void Darknet::set_annotation_draw_label(const bool draw)
{
	TAT(TATPARMS);

	annotate_draw_label = draw;

	return;
}


Darknet::NetworkPtr Darknet::load_neural_network(const std::filesystem::path & cfg_filename, const std::filesystem::path & names_filename, const std::filesystem::path & weights_filename)
{
	TAT(TATPARMS);

	if (cfg_filename.empty())
	{
		throw std::invalid_argument("cannot load a neural network without a configuration file (filename is blank)");
	}

	if (weights_filename.empty())
	{
		throw std::invalid_argument("cannot load a neural network without a weights file (filename is blank)");
	}

	if (not std::filesystem::exists(cfg_filename))
	{
		throw std::invalid_argument("configuration filename is invalid: \"" + cfg_filename.string() + "\"");
	}

	if (not std::filesystem::exists(weights_filename))
	{
		throw std::invalid_argument("weights filename is invalid: \"" + weights_filename.string() + "\"");
	}

	// the .names file is optional and shouldn't stop us from loading the neural network
	if (names_filename.empty() == false and std::filesystem::exists(names_filename) == false)
	{
		throw std::invalid_argument("names filename is invalid: \"" + names_filename.string() + "\"");
	}

	static bool initialized = false;
	#ifdef GPU
	if (cfg_and_state.gpu_index < 0)
	{
		// no idea what GPU to use, so attempt to use the first one
		cfg_and_state.gpu_index = 0;
	}
	cudaError_t status = cudaSetDevice(cfg_and_state.gpu_index);
	if (status == cudaSuccess)
	{
		initialized = true;
	}
	else
	{
		display_warning_msg("failed to set the GPU device to #" + std::to_string(cfg_and_state.gpu_index) + "\n");
		cfg_and_state.gpu_index = -1;
	}
	#endif

	if (not initialized)
	{
		cfg_and_state.gpu_index = -1;
		init_cpu();
		initialized = true;
	}

	network * net = load_network_custom(cfg_filename.string().c_str(), weights_filename.string().c_str(), 0, 1);

	/// @todo need to load and store the names somewhere

	return reinterpret_cast<NetworkPtr>(net);
}


Darknet::NetworkPtr Darknet::load_neural_network(const Darknet::Parms & parms)
{
	TAT(TATPARMS);

	std::filesystem::path cfg;
	std::filesystem::path names;
	std::filesystem::path weights;

	for (const auto & parm : parms)
	{
		if (parm.type == EParmType::kCfgFilename		and cfg		.empty())	cfg		= parm.string;
		if (parm.type == EParmType::kNamesFilename		and names	.empty())	names	= parm.string;
		if (parm.type == EParmType::kWeightsFilename	and weights	.empty())	weights	= parm.string;
	}

	return load_neural_network(cfg, names, weights);
}


void Darknet::free_neural_network(Darknet::NetworkPtr & ptr)
{
	TAT(TATPARMS);

	if (ptr)
	{
		network * net = reinterpret_cast<network *>(ptr);
		free_network_ptr(net);
		ptr = nullptr;
	}

	return;
}


void Darknet::network_dimensions(Darknet::NetworkPtr & ptr, int & w, int & h, int & c)
{
	TAT(TATPARMS);

	w = -1;
	h = -1;
	c = -1;

	network * net = reinterpret_cast<network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot determine dimensions without a network pointer");
	}

	w = net->w;
	h = net->h;
	c = net->c;

	return;
}


Darknet::Predictions Darknet::predict(const Darknet::NetworkPtr ptr, cv::Mat mat)
{
	TAT(TATPARMS);

	network * net = reinterpret_cast<network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot predict without a network pointer");
	}
	if (mat.empty())
	{
		throw std::invalid_argument("cannot predict without a valid image");
	}

	const cv::Size network_dimensions(net->w, net->h);
	const cv::Size original_image_size = mat.size();

	if (mat.size() != network_dimensions)
	{
		// Note that INTER_NEAREST gives us *speed*, not image quality.
		//
		// If quality matters, you'll want to resize the image yourself
		// using INTER_AREA, INTER_CUBIC or INTER_LINEAR prior to calling
		// predict().  See DarkHelp or OpenCV documentation for details.

		cv::resize(mat, mat, network_dimensions, cv::INTER_NEAREST);
	}

	// OpenCV uses BGR, but Darknet expects RGB
	if (mat.channels() == 3)
	{
		cv::cvtColor(mat, mat, cv::COLOR_BGR2RGB);
	}

	image img = mat_to_image(mat);

	return predict(ptr, img, original_image_size);
}


Darknet::Predictions Darknet::predict(const Darknet::NetworkPtr ptr, image & img, cv::Size original_image_size)
{
	TAT(TATPARMS);

	network * net = reinterpret_cast<network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot predict without a network pointer");
	}

	// If we don't know the original image size, then use the current image size.
	// Note the bounding box results will be wrong if the image has been resized!
	if (original_image_size.width	< 1) original_image_size.width	= img.w;
	if (original_image_size.height	< 1) original_image_size.height	= img.h;

	network_predict(*net, img.data); /// todo pass net by ref or pointer, not copy constructor!
	free_image(img);

	int nboxes = 0;
	const float hierarchy_threshold = 0.5f;
	auto darknet_results = get_network_boxes(net, img.w, img.h, detection_threshold, hierarchy_threshold, 0, 1, &nboxes, 0);

	if (non_maximal_suppression_threshold)
	{
		auto & layer = net->layers[net->n - 1];
		do_nms_sort(darknet_results, nboxes, layer.classes, non_maximal_suppression_threshold);
	}

	Predictions predictions;
	predictions.reserve(nboxes); // this is likely too many (depends on the detection threshold) but gets us in the ballpark

	for (int detection_idx = 0; detection_idx < nboxes; detection_idx ++)
	{
		auto & det = darknet_results[detection_idx];

		/* The "det" object has an array called det.prob[].  That array is large enough for 1 entry per class in the network.
		 * Each entry will be set to 0.0f, except for the ones that correspond to the class that was detected.  Note that it
		 * is possible that multiple entries are non-zero!  We need to look at every entry and remember which ones are set.
		 */

		Prediction pred;
		pred.best_class = -1;

		for (int class_idx = 0; class_idx < det.classes; class_idx ++)
		{
			const auto probability = det.prob[class_idx];
			if (probability >= detection_threshold)
			{
				// remember this probability since it is higher than the user-specified threshold
				pred.prob[class_idx] = probability;
				if (pred.best_class == -1 or probability > det.prob[pred.best_class])
				{
					pred.best_class = class_idx;
				}
			}
		}

		// most of the output from Darknet/YOLO will have a confidence of 0.0f which we need to completely ignore
		if (pred.best_class == -1)
		{
			continue;
		}

		if (fix_out_of_bound_normalized_coordinates)
		{
			fix_out_of_bound_normalized_rect(det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h);
		}

		const int w = std::round(det.bbox.w * original_image_size.width				);
		const int h = std::round(det.bbox.h * original_image_size.height			);
		const int x = std::round(det.bbox.x * original_image_size.width	- w / 2.0f	);
		const int y = std::round(det.bbox.y * original_image_size.height- h / 2.0f	);

		pred.rect				= cv::Rect(cv::Point(x, y), cv::Size(w, h));
		pred.normalized_point	= cv::Point2f(det.bbox.x, det.bbox.y);
		pred.normalized_size	= cv::Size2f(det.bbox.w, det.bbox.h);

		predictions.push_back(pred);
	}

	free_detections(darknet_results, nboxes);

	return predictions;
}


Darknet::Predictions Darknet::predict(const Darknet::NetworkPtr ptr, const std::filesystem::path & image_filename)
{
	TAT(TATPARMS);

	if (not std::filesystem::exists(image_filename))
	{
		throw std::invalid_argument("cannot predict due to invalid image filename: \"" + image_filename.string() + "\"");
	}

	cv::Mat mat = cv::imread(image_filename);

	return predict(ptr, mat);
}


cv::Mat Darknet::annotate(const Darknet::Predictions & predictions, cv::Mat mat)
{
	TAT(TATPARMS);

	if (mat.empty())
	{
		throw std::invalid_argument("cannot annotate empty image");
	}

	for (const auto & pred : predictions)
	{
		if (annotate_draw_bb)
		{
			// draw the bounding box around the entire object
			if (annotate_draw_rounded_bb_toggle == false)
			{
				cv::rectangle(mat, pred.rect, colour_bb_lines, 1, cv_font_line_type);
			}
			else
			{
				draw_rounded_rectangle(mat, pred.rect, annotate_draw_rounded_bb_roundness);
			}
		}

		if (annotate_draw_label)
		{
			std::string text = "TEST #" + std::to_string(pred.best_class) + " ";
			text += std::to_string(static_cast<int>(std::round(100.0f * pred.prob.at(pred.best_class)))) + "%";

			int				font_baseline	= 0;
			const cv::Size	size			= cv::getTextSize(text, cv_font_face, cv_font_scale, cv_font_thickness, &font_baseline);
			cv::Rect		label			= pred.rect;
			label.y							= label.y - size.height - font_baseline;
			label.height					= size.height + font_baseline;
			label.width						= size.width + 2;

			// draw a rectangle above that to use as a label
			cv::rectangle(mat, label, colour_bb_lines, cv::FILLED, cv_font_line_type);

			// and finally we draw the text on top of the label background
			cv::putText(mat, text, cv::Point(label.x + 1, label.y + label.height - font_baseline / 2), cv_font_face, cv_font_scale, colour_label_text, cv_font_thickness, cv_font_line_type);
		}
	}

	return mat;
}


Darknet::Predictions Darknet::predict_and_annotate(const Darknet::NetworkPtr ptr, cv::Mat mat)
{
	TAT(TATPARMS);

	const auto predictions = predict(ptr, mat);

	annotate(predictions, mat);

	return predictions;
}


std::ostream & Darknet::operator<<(std::ostream & os, const Darknet::Prediction & pred)
{
	TAT(TATPARMS);

	os	<< "#" << pred.best_class
		<< " prob=" << pred.prob.at(pred.best_class)
		<< " x=" << pred.rect.x
		<< " y=" << pred.rect.y
		<< " w=" << pred.rect.width
		<< " h=" << pred.rect.height
		<< " entries=" << pred.prob.size();

	if (pred.prob.size() > 1)
	{
		os << " [";
		for (auto & [key, val] : pred.prob)
		{
			os << " " << key << "=" << val;
		}
		os << " ]";
	}

	return os;
}


std::ostream & Darknet::operator<<(std::ostream & os, const Darknet::Predictions & preds)
{
	TAT(TATPARMS);

	os << "prediction results: " << preds.size();

	for (size_t idx = 0; idx < preds.size(); idx ++)
	{
		os << std::endl << "-> " << (idx + 1) << "/" << preds.size() << ": ";
		operator<<(os, preds.at(idx));
	}

	return os;
}
