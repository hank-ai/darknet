#include "darknet_internal.hpp"

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

std::string get_windows_version()
{
	TAT(TATPARMS);

	// Oh, this is ugly...why is it so complicated to get an accurate version number or product name from Windows?

	std::string txt = "Windows";

	HMODULE module = LoadLibrary("winbrand.dll");
	if (module)
	{
		wchar_t * (WINAPI * BrandingFormatString)(const wchar_t *);
		(FARPROC&)BrandingFormatString = GetProcAddress(module, "BrandingFormatString");

		if (BrandingFormatString)
		{
			// this should return a wide string like "Windows 11 Home"
			wchar_t * name = BrandingFormatString(L"%WINDOWS_LONG%");

			char buffer[200] = "";
			std::wcstombs(buffer, name, sizeof(buffer));
			txt = buffer;

			GlobalFree((HGLOBAL)name);
		}

		FreeLibrary(module);
	}

	return txt;
}
#endif


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();

	// remember that OpenCV colours are BGR, not RGB
	static const auto white = cv::Scalar(255, 255, 255);
	static const auto black	= cv::Scalar(0, 0, 0);

	// shamlessly stolen from DarkHelp
	static inline void fix_out_of_bound_normalized_rect(float & cx, float & cy, float & w, float & h)
	{
		TAT(TATPARMS);

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

	static inline void draw_rounded_rectangle(cv::Mat & mat, const cv::Rect & r, const float roundness, const cv::Scalar & colour, const cv::LineTypes line_type)
	{
		TAT(TATPARMS);

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
			cv::ellipse(mat, cv::Point(r.x + r.width / 2, r.y + r.height / 2), r.size() / 2, 0.0, 0.0, 360.0, colour, 1, line_type);
		}
		else
		{
			// draw horizontal and vertical segments
			cv::line(mat, cv::Point(tl.x + hoffset, tl.y), cv::Point(tr.x - hoffset, tr.y), colour, 1, line_type);
			cv::line(mat, cv::Point(tr.x, tr.y + voffset), cv::Point(br.x, br.y - voffset), colour, 1, line_type);
			cv::line(mat, cv::Point(br.x - hoffset, br.y), cv::Point(bl.x + hoffset, bl.y), colour, 1, line_type);
			cv::line(mat, cv::Point(bl.x, bl.y - voffset), cv::Point(tl.x, tl.y + voffset), colour, 1, line_type);

			cv::ellipse(mat, tl + cv::Point(+hoffset, +voffset), cv::Size(hoffset, voffset), 0.0, 180.0	, 270.0	, colour, 1, line_type);
			cv::ellipse(mat, tr + cv::Point(-hoffset, +voffset), cv::Size(hoffset, voffset), 0.0, 270.0	, 360.0	, colour, 1, line_type);
			cv::ellipse(mat, br + cv::Point(-hoffset, -voffset), cv::Size(hoffset, voffset), 0.0, 0.0	, 90.0	, colour, 1, line_type);
			cv::ellipse(mat, bl + cv::Point(+hoffset, -voffset), cv::Size(hoffset, voffset), 0.0, 90.0	, 180.0	, colour, 1, line_type);
		}

		return;
	}
}


extern "C"
{
	void darknet_show_version_info()
	{
		TAT(TATPARMS);
		Darknet::show_version_info();
		return;
	}


	const char * darknet_version_string()
	{
		TAT(TATPARMS);
		return DARKNET_VERSION_STRING;
	}


	const char * darknet_version_short()
	{
		TAT(TATPARMS);
		return DARKNET_VERSION_SHORT;
	}


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


	void darknet_set_detection_threshold(DarknetNetworkPtr ptr, float threshold)
	{
		TAT(TATPARMS);
		Darknet::set_detection_threshold(ptr, threshold);
		return;
	}


	void darknet_set_non_maximal_suppression_threshold(DarknetNetworkPtr ptr, float threshold)
	{
		TAT(TATPARMS);
		Darknet::set_non_maximal_suppression_threshold(ptr, threshold);
		return;
	}

	void darknet_fix_out_of_bound_values(DarknetNetworkPtr ptr, const bool toggle)
	{
		TAT(TATPARMS);
		Darknet::fix_out_of_bound_values(ptr, toggle);
		return;
	}

	void darknet_network_dimensions(DarknetNetworkPtr ptr, int * w, int * h, int * c)
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

	DarknetNetworkPtr darknet_load_neural_network(const char * const cfg_filename, const char * const names_filename, const char * const weights_filename)
	{
		TAT(TATPARMS);

		std::filesystem::path cfg;
		std::filesystem::path names;
		std::filesystem::path weights;

		if (cfg_filename)		cfg		= cfg_filename;
		if (names_filename)		names	= names_filename;
		if (weights_filename)	weights	= weights_filename;

		return Darknet::load_neural_network(cfg, names, weights);
	}

	void darknet_free_neural_network(DarknetNetworkPtr * ptr)
	{
		TAT(TATPARMS);

		if (ptr)
		{
			Darknet::free_neural_network(*ptr);
			ptr = nullptr;
		}

		return;
	}

	void darknet_clear_skipped_classes(DarknetNetworkPtr ptr)
	{
		TAT(TATPARMS);

		Darknet::clear_skipped_classes(ptr);

		return;
	}

	void darknet_add_skipped_class(DarknetNetworkPtr ptr, const int class_to_skip)
	{
		TAT(TATPARMS);

		Darknet::add_skipped_class(ptr, class_to_skip);

		return;
	}

	void darknet_del_skipped_class(DarknetNetworkPtr ptr, const int class_to_include)
	{
		TAT(TATPARMS);

		Darknet::del_skipped_class(ptr, class_to_include);

		return;
	}

	void darknet_set_output_stream(const char * const filename)
	{
		TAT(TATPARMS);

		std::filesystem::path path;
		if (filename != nullptr and filename[0] != '\0')
		{
			path = filename;
		}
		Darknet::set_output_stream(path);

		return;
	}
}


void Darknet::show_version_info()
{
	TAT(TATPARMS);

	*cfg_and_state.output << "Darknet V6 \"" << DARKNET_VERSION_KEYWORD << "\" " << Darknet::in_colour(Darknet::EColour::kBrightWhite, DARKNET_VERSION_STRING);
	if (DARKNET_BRANCH_NAME != std::string("master"))
	{
		*cfg_and_state.output << " [" << Darknet::in_colour(Darknet::EColour::kBrightWhite, DARKNET_BRANCH_NAME) << "]";
	}
	#ifndef NDEBUG
	*cfg_and_state.output << " [" << Darknet::in_colour(Darknet::EColour::kBrightRed, "DEBUG BUILD!") << "]";
	#endif
	#ifdef DARKNET_TIMING_AND_TRACKING_ENABLED
	*cfg_and_state.output << " [" << Darknet::in_colour(Darknet::EColour::kBrightBlue, "TIMING BUILD!") << "]";
	#endif

	#ifdef DARKNET_PROFILE_GEN
	*cfg_and_state.output << " [" << Darknet::in_colour(Darknet::EColour::kBrightCyan, "PGO GENERATE!") << "]";
	#endif

	#ifdef DARKNET_PROFILE_USE
	*cfg_and_state.output << " [" << Darknet::in_colour(Darknet::EColour::kBrightCyan, "PGO ENABLED!") << "]";
	#endif

	*cfg_and_state.output << " " << Darknet::in_colour(Darknet::EColour::kDarkGrey, "[" __DATE__ "]") << std::endl;

	#if DARKNET_GPU_ROCM
		Darknet::show_rocm_info();
	#elif defined(DARKNET_GPU_CUDA)
		show_cuda_cudnn_info();
	#elif defined(DARKNET_USE_MPS)
		Darknet::show_mps_info();
	#else
		Darknet::display_warning_msg("Darknet is compiled to use the CPU.");
		*cfg_and_state.output << "  GPU is " << Darknet::in_colour(Darknet::EColour::kBrightRed, "disabled") << "." << std::endl;

		#ifdef DARKNET_USE_OPENBLAS
			*cfg_and_state.output << Darknet::trim(OPENBLAS_VERSION) << ", ";
		#else
			*cfg_and_state.output << "OpenBLAS is " << Darknet::in_colour(Darknet::EColour::kBrightRed, "disabled") << ", ";
		#endif
	#endif

	#if DARKNET_HAS_PROTOBUF
		// 3021012 = "3.21.12"
		const int pb_major = (GOOGLE_PROTOBUF_VERSION / 1000000	) % 1000;	// 3
		const int pb_minor = (GOOGLE_PROTOBUF_VERSION / 1000	) % 1000;	// 21
		const int pb_patch = (GOOGLE_PROTOBUF_VERSION / 1		) % 1000;	// 12
		const std::string pb_ver =
			std::to_string(pb_major) + "." +
			std::to_string(pb_minor) + "." +
			std::to_string(pb_patch);
		*cfg_and_state.output << "Protobuf " << Darknet::in_colour(Darknet::EColour::kBrightWhite, pb_ver) << ", ";
	#endif

	*cfg_and_state.output << "OpenCV " << Darknet::in_colour(Darknet::EColour::kBrightWhite, CV_VERSION);


	#ifdef WIN32
	*cfg_and_state.output << ", " << get_windows_version();
	#else
	const std::string lsb_release = "/etc/lsb-release";
	if (std::filesystem::exists(lsb_release))
	{
		std::ifstream ifs(lsb_release);
		if (ifs.good())
		{
			std::string id;
			std::string release;

			std::string line;
			while (std::getline(ifs, line))
			{
				// for example, the line could be "DISTRIB_ID=Ubuntu"

				const size_t pos = line.find("=");
				if (pos == std::string::npos)
				{
					continue;
				}
				const std::string key = line.substr(0, pos);
				const std::string val = line.substr(pos + 1);

				if (key == "DISTRIB_ID")		id = val;
				if (key == "DISTRIB_RELEASE")	release = val;
			}

			if (not id.empty())
			{
				*cfg_and_state.output << ", " << id;

				if (not release.empty())
				{
					*cfg_and_state.output << " " << Darknet::in_colour(Darknet::EColour::kBrightWhite, release);
				}
			}
		}
	}

	// attempt to log if we're running in WSL or some other unusual environment
	const std::string detect_virt = "/usr/bin/systemd-detect-virt";
	if (std::filesystem::exists(detect_virt))
	{
		const std::string output = Darknet::trim(Darknet::get_command_output(detect_virt));
		if (not output.empty() and output != "none")
		{
			*cfg_and_state.output << ", " << output;
		}
	}

	#endif

	*cfg_and_state.output << std::endl;
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
		if (parm.idx >= 0)
		{
			parm.idx ++;
		}
	}

	return parms;
}


Darknet::Parms Darknet::parse_arguments(const Darknet::VStr & v)
{
	TAT(TATPARMS);

	if (not v.empty())
	{
		// attempt to parse all the common parms -- note we don't yet have a network, so we need to pass in nullptr
		cfg_and_state.process_arguments(v, nullptr);
	}

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

	// before we start looking at the parms, we need to expand wildcards (globbing)
	// (mostly for Windows, on Linux this is normally handled by the shell)
	for (int idx = 0; idx < parms.size(); idx ++)
	{
		auto parm = parms[idx]; // not by reference since we may resize the vector below

		if (parm.string.find_first_of("?*") == std::string::npos)
		{
			continue;
		}

		// if we get here, then we have either a "*" or "?"
		if (cfg_and_state.is_trace)
		{
			*cfg_and_state.output << "-> performing file globbing with parameter \"" << parm.original << "\"" << std::endl;
		}
		std::filesystem::path path(parm.original);
		std::filesystem::path parent = path.parent_path();
		if (parent.empty())
		{
			parent = std::filesystem::current_path();
		}

		if (not std::filesystem::exists(parent))
		{
			// not much we can do with this parm, we have no idea what it is
			if (cfg_and_state.is_trace)
			{
				*cfg_and_state.output << "-> failed to find parent directory " << parent << std::endl;
			}
			continue;
		}

		// convert the filename to a regular expression
		std::string txt;
		for (const char & c : path.filename().string())
		{
			if (c == '.')
			{
				txt += "\\.";
			}
			else if (c == '?')
			{
				txt += ".";
			}
			else if (c == '*')
			{
				txt += ".*";
			}
			else
			{
				// file globbing is primarily for Windows, which is case insensitive, so go ahead
				// and convert the name to lowercase and we'll do a case-insensitive compare below
				//
				// yes, this will be strange behaviour on Linux, but we shouldn't even be doing file
				// globbing on Linux since the shell normally takes care of all this
				txt += std::tolower(c);
			}
		}

		const std::regex rx(txt);
		VStr filenames_which_matched;
		if (cfg_and_state.is_trace)
		{
			*cfg_and_state.output << "looking at files in \"" << parent.string() << "\"" << std::endl;
		}
		for (const auto & entry : std::filesystem::directory_iterator(parent))
		{
			std::string fn = Darknet::lowercase(entry.path().filename().string());

			if (std::regex_match(fn, rx))
			{
				filenames_which_matched.push_back(entry.path().string());
			}
		}

		if (not filenames_which_matched.empty())
		{
			// great, we found some matching filenames!
			// so remove this parameter, and insert the filenames instead
			std::sort(/** @todo try this again in 2026? std::execution::par_unseq,*/ filenames_which_matched.begin(), filenames_which_matched.end());

			for (const auto & fn : filenames_which_matched)
			{
				Darknet::Parm p = parm;
				p.string = fn;
				parms.push_back(p);
			}

			parms.erase(parms.begin() + idx);
			idx --;
		}
	}

	find_neural_network_files(parms);

	// Anything which was recognized and consumed needs to be marked as a parameter.
	// For example, when processing "-log output.log" and "-thresh 0.4" we don't want those 2nd parms to be processed as input files.
	for (const auto & [key, val] : cfg_and_state.args)
	{
		if (val.expect_parm)
		{
			parms[val.arg_index + 1].type = EParmType::kOther;
		}
	}

	if (cfg_and_state.is_trace)
	{
		for (size_t idx = 0; idx < parms.size(); idx ++)
		{
			*cfg_and_state.output << "Parameter parsing: #" << idx << " [type " << static_cast<int>(parms[idx].type) << ", " << parms[idx].type << "] -> " << parms[idx].string;
			if (parms[idx].original != parms[idx].string)
			{
				*cfg_and_state.output << " (" << parms[idx].original << ")";
			}
			*cfg_and_state.output << std::endl;
		}
	}

	return parms;
}


bool Darknet::find_neural_network_files(Darknet::Parms & parms)
{
	TAT(TATPARMS);

	// 1st step:  see if we can identify the 3 files we need to load the network
	for (auto & parm : parms)
	{
		std::filesystem::path path(parm.string);
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
				*cfg_and_state.output << "Found configuration: " << Darknet::in_colour(Darknet::EColour::kBrightWhite, path.string()) << std::endl;
			}
			parm.type = EParmType::kCfgFilename;
		}
		else if (extension == ".names")
		{
			if (cfg_and_state.is_verbose)
			{
				*cfg_and_state.output << "Found names file:    " << Darknet::in_colour(Darknet::EColour::kBrightWhite, path.string()) << std::endl;
			}
			parm.type = EParmType::kNamesFilename;
		}
		else if (extension == ".weights")
		{
			if (cfg_and_state.is_verbose)
			{
				*cfg_and_state.output << "Found weights file:  " << Darknet::in_colour(Darknet::EColour::kBrightWhite, path.string()) << std::endl;
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
				*cfg_and_state.output << "Guessed names:       " << Darknet::in_colour(Darknet::EColour::kBrightGreen, path.string()) << std::endl;
				Parm parm = parms[cfg_idx];
				parm.type = EParmType::kNamesFilename;
				parm.string = path.string();
				parms.push_back(parm);
				names_idx = parms.size() - 1;
			}
		}

		if (weights_idx == -1)
		{
			std::string tmp = path.string();
			auto pos = tmp.rfind(".");
			tmp.erase(pos);
			tmp += "_best.weights";
			if (std::filesystem::exists(tmp))
			{
				*cfg_and_state.output << "Guessed weights:     " << Darknet::in_colour(Darknet::EColour::kBrightGreen, tmp) << std::endl;
				Parm parm = parms[cfg_idx];
				parm.type = EParmType::kWeightsFilename;
				parm.string = tmp;
				parms.push_back(parm);
				weights_idx = parms.size() - 1;
			}
			else
			{
				path.replace_extension(".weights");
				if (std::filesystem::exists(path))
				{
					*cfg_and_state.output << "Guessed weights:     " << Darknet::in_colour(Darknet::EColour::kBrightGreen, path.string()) << std::endl;
					Parm parm = parms[cfg_idx];
					parm.type = EParmType::kWeightsFilename;
					parm.string = path.string();
					parms.push_back(parm);
					weights_idx = parms.size() - 1;
				}
			}
		}
	}

	// 3rd step:  if we have the .cfg, and we're missing the .weights, but we have other possible filenames to use...
	if (cfg_idx >= 0 and weights_idx == -1)
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
				for (size_t i = 0; ifs.good() and i < 3; i ++)
				{
					uint32_t tmp = 0;
					ifs.read(reinterpret_cast<char*>(&tmp), sizeof(tmp));

					if (tmp == expected_header[i])
					{
						header_bytes_matched ++;
					}
				}

				if (header_bytes_matched == 3)
				{
					*cfg_and_state.output << "Found these weights: " << Darknet::in_colour(Darknet::EColour::kYellow, filename) << std::endl;
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
			auto parm = parms[idx]; // not by reference!  we'll be modifying the vector

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
			if (cfg_and_state.is_trace)
			{
				*cfg_and_state.output << "looking for neural network in \"" << parent.string() << "\"" << std::endl;
			}
			try
			{
				for (auto iter : std::filesystem::directory_iterator(parent))
				{
					const std::filesystem::path path = iter.path();
					if (path.filename().string().find(stem) == 0)
					{
						matching_files.push_back(path.string());
					}
				}
			}
			catch (...)
			{
				// probably not a directory, so we need to ignore this parameter
			}

			std::sort(/** @todo try this again in 2026? std::execution::par_unseq,*/ matching_files.begin(), matching_files.end());

			std::string backup_weights;

			for (const auto & filename : matching_files)
			{
				const auto extension = std::filesystem::path(filename).extension().string();
				if (extension == ".cfg" and cfg_idx == -1)
				{
					*cfg_and_state.output << tmp << " matches this config file:  " << Darknet::in_colour(Darknet::EColour::kBrightCyan, filename) << std::endl;
					parm.type = EParmType::kCfgFilename;
					parm.string = filename;
					parms[idx] = parm;
					cfg_idx = idx;
				}
				else if (extension == ".names" and names_idx == -1)
				{
					*cfg_and_state.output << tmp << " matches this names file:   " << Darknet::in_colour(Darknet::EColour::kBrightCyan, filename) << std::endl;
					Parm names = parms[idx];
					names.type = EParmType::kNamesFilename;
					names.string = filename;
					parms.push_back(names);
					names_idx = parms.size() - 1;
				}
				else if (extension == ".weights")
				{
					if (weights_idx == -1 and filename.find("_best.weights") != std::string::npos)
					{
						*cfg_and_state.output << tmp << " matches this weights file: " << Darknet::in_colour(Darknet::EColour::kBrightCyan, filename) << std::endl;
						Parm weights = parms[idx];
						weights.type = EParmType::kWeightsFilename;
						weights.string = filename;
						parms.push_back(weights);
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
				*cfg_and_state.output << tmp << " matches this weights file: " << Darknet::in_colour(Darknet::EColour::kBrightCyan, backup_weights) << std::endl;
				Parm weights = parms[idx];
				weights.type = EParmType::kWeightsFilename;
				weights.string = backup_weights;
				parms.push_back(weights);
				weights_idx = parms.size() - 1;
			}
		}
	}

	return (cfg_idx		!= -1 and
			names_idx	!= -1 and
			weights_idx	!= -1);
}


Darknet::Parms & Darknet::set_default_neural_network(Darknet::Parms & parms, const std::string & hint1, const std::string & hint2, const std::string & hint3)
{
	TAT(TATPARMS);

	if (get_config_filename(parms).empty())
	{
		Parms hints;

		for (const std::string & hint : {hint1, hint2, hint3})
		{
			if (not hint.empty())
			{
				Parm p;
				p.idx		= -1;
				p.type		= EParmType::kOther;
				p.original	= hint;
				p.string	= hint;
				hints.push_back(p);
			}
		}

		find_neural_network_files(hints);
		parms.insert(parms.end(), hints.begin(), hints.end());
	}

	return parms;
}


std::filesystem::path Darknet::get_config_filename(const Darknet::Parms & parms)
{
	TAT(TATPARMS);

	std::filesystem::path path;

	for (const auto & parm : parms)
	{
		if (parm.type == EParmType::kCfgFilename)
		{
			path = parm.string;
			break;
		}
	}

	return path;
}


std::filesystem::path Darknet::get_names_filename(const Darknet::Parms & parms)
{
	TAT(TATPARMS);

	std::filesystem::path path;

	for (const auto & parm : parms)
	{
		if (parm.type == EParmType::kNamesFilename)
		{
			path = parm.string;
			break;
		}
	}

	return path;
}


std::filesystem::path Darknet::get_weights_filename(const Darknet::Parms & parms)
{
	TAT(TATPARMS);

	std::filesystem::path path;

	for (const auto & parm : parms)
	{
		if (parm.type == EParmType::kWeightsFilename)
		{
			path = parm.string;
			break;
		}
	}

	return path;
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

	#ifdef DARKNET_GPU
	cfg_and_state.gpu_index = idx;
	#else
	// don't allow the GPU index to be set when Darknet was not compiled with CUDA support
	cfg_and_state.gpu_index = -1;
	#endif

	return;
}


void Darknet::set_detection_threshold(Darknet::NetworkPtr ptr, float threshold)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "-> attempting to set detection threshold to " << threshold << std::endl;
	}

	Darknet::Network * net = reinterpret_cast<Darknet::Network*>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("pointer to neural network cannot be NULL");
	}

	if (threshold > 1.0f and threshold < 100.0f)
	{
		// user must be using percentages instead
		threshold /= 100.0f;
	}

	if (threshold >= 0.0f and threshold <= 1.0f)
	{
		net->details->detection_threshold = threshold;
		return;
	}

	throw std::invalid_argument("detection threshold must be between 0.0 and 1.0");
}


void Darknet::set_non_maximal_suppression_threshold(Darknet::NetworkPtr ptr, float threshold)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "-> attempting to set NMS threshold to " << threshold << std::endl;
	}

	Darknet::Network * net = reinterpret_cast<Darknet::Network*>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("pointer to neural network cannot be NULL");
	}

	if (threshold > 1.0f and threshold < 100.0f)
	{
		// user must be using percentages instead
		threshold /= 100.0f;
	}
	if (threshold >= 0.0f and threshold <= 1.0f)
	{
		net->details->non_maximal_suppression_threshold = threshold;
		return;
	}
	throw std::invalid_argument("nms threshold must be between 0.0 and 1.0");
}


void Darknet::fix_out_of_bound_values(Darknet::NetworkPtr ptr, const bool toggle)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network*>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("pointer to neural network cannot be NULL");
	}

	net->details->fix_out_of_bound_normalized_coordinates = toggle;

	return;
}


void Darknet::set_annotation_font(Darknet::NetworkPtr ptr, const cv::LineTypes line_type, const cv::HersheyFonts font_face, const int font_thickness, const double font_scale)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network*>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("pointer to neural network cannot be NULL");
	}

	net->details->cv_line_type		= line_type;
	net->details->cv_font_face		= font_face;
	net->details->cv_font_thickness	= font_thickness;
	net->details->cv_font_scale		= font_scale;

	return;
}


void Darknet::set_annotation_line_type(Darknet::NetworkPtr ptr, const cv::LineTypes line_type)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network*>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("pointer to neural network cannot be NULL");
	}

	net->details->cv_line_type = line_type;

	return;
}


void Darknet::set_rounded_corner_bounding_boxes(Darknet::NetworkPtr ptr, const bool toggle, const float roundness)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network*>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("pointer to neural network cannot be NULL");
	}

	net->details->bounding_boxes_with_rounded_corners = toggle;
	net->details->bounding_boxes_corner_roundness = roundness;

	return;
}


void Darknet::set_annotation_draw_bb(Darknet::NetworkPtr ptr, const bool toggle)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network*>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("pointer to neural network cannot be NULL");
	}

	net->details->annotate_draw_bb = toggle;

	return;
}


void Darknet::set_annotation_draw_label(Darknet::NetworkPtr ptr, const bool toggle)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network*>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("pointer to neural network cannot be NULL");
	}

	net->details->annotate_draw_label = toggle;

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
	#ifdef DARKNET_GPU
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

	NetworkPtr ptr = load_network_custom(cfg_filename.string().c_str(), weights_filename.string().c_str(), 0, 1);

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "-> network loaded: cfg=" << cfg_filename.string() << " ptr=" << (void*)ptr << std::endl;
	}

	if (not names_filename.empty())
	{
		Darknet::load_names(ptr, names_filename);
	}

	// detection threshold is often modified by the user on the CLI, so apply that change now that the network has been loaded
	//
	// (probably lots of other flags we need to apply, but start with the threshold)
	if (cfg_and_state.is_set("threshold"))
	{
		set_detection_threshold(ptr, cfg_and_state.get_float("threshold"));
	}
	if (cfg_and_state.is_set("nms"))
	{
		set_non_maximal_suppression_threshold(ptr, cfg_and_state.get_float("nms"));
	}

	return ptr;
}


Darknet::NetworkPtr Darknet::load_neural_network(Darknet::Parms & parms)
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

	auto ptr = load_neural_network(cfg, names, weights);

	VStr v;
	for (const auto & parm : parms)
	{
		v.push_back(parm.string);
	}
	if (not v.empty())
	{
		cfg_and_state.process_arguments(v, ptr);
	}

	v.clear();
	for (const auto & parm : parms)
	{
		if (parm.type == EParmType::kDirectory)
		{
			if (cfg_and_state.is_trace)
			{
				*cfg_and_state.output << "looking for images in \"" << parm.string << "\"" << std::endl;
			}
			for (const auto & entry : std::filesystem::directory_iterator(parm.string))
			{
				const auto ext = Darknet::lowercase(entry.path().extension().string());
				if (ext == ".jpg" or ext == ".png")
				{
					v.push_back(entry.path().string());
				}
			}
		}
	}
	if (not v.empty())
	{
		if (cfg_and_state.is_set("random"))
		{
			std::mt19937 rng(std::random_device{}());

			std::shuffle(/** @todo try this again in 2026? std::execution::par_unseq,*/ v.begin(), v.end(), rng);
		}
		else
		{
			std::sort(/** @todo try this again in 2026? std::execution::par_unseq,*/ v.begin(), v.end());
		}

		// insert all the image filenames into "parms"
		for (const auto & fn : v)
		{
			Parm parm;
			parm.idx = -1;
			parm.original = fn;
			parm.string = fn;
			parm.type = EParmType::kFilename;
			parms.push_back(parm);
		}
	}

	return ptr;
}


void Darknet::free_neural_network(Darknet::NetworkPtr & ptr)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "-> freeing network ptr=" << reinterpret_cast<void*>(ptr) << std::endl;
	}

	if (ptr)
	{
		Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
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

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot determine dimensions without a network pointer");
	}

	w = net->w;
	h = net->h;
	c = net->c;

	return;
}


Darknet::Predictions Darknet::predict(const Darknet::NetworkPtr ptr, const cv::Mat & mat)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
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

	cv::Mat bgr;
	if (mat.size() != network_dimensions)
	{
		// Note that INTER_NEAREST gives us *speed*, not image quality.
		//
		// If quality matters, you'll want to resize the image yourself
		// using INTER_AREA, INTER_CUBIC or INTER_LINEAR prior to calling
		// predict().  See DarkHelp or OpenCV documentation for details.

		cv::resize(mat, bgr, network_dimensions, cv::INTER_NEAREST);
	}
	else
	{
		bgr = mat;
	}

	// OpenCV uses BGR, but Darknet requires RGB
	Darknet::Image img;
	if (bgr.channels() == 4)
	{
		cv::Mat rgb;
		cv::cvtColor(bgr, rgb, cv::COLOR_BGRA2RGB);
		img = rgb_mat_to_rgb_image(rgb);
	}
	else
	{
		// anything else we currently assume is 3-channel BGR
		img = bgr_mat_to_rgb_image(bgr);
	}

	return predict(ptr, img, original_image_size);
}


Darknet::Predictions Darknet::predict(Darknet::NetworkPtr ptr, Darknet::Image & img, cv::Size original_image_size)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot predict without a network pointer");
	}

	// If we don't know the original image size, then use the current image size.
	// Note the bounding box results will be wrong if the image has been resized!
	if (original_image_size.width	< 1) original_image_size.width	= img.w;
	if (original_image_size.height	< 1) original_image_size.height	= img.h;

	network_predict(*net, img.data); /// @todo pass net by ref or pointer, not copy constructor!
	Darknet::free_image(img);

	int nboxes = 0;
	const float hierarchy_threshold = 0.5f;
	auto darknet_results = get_network_boxes(net, img.w, img.h, net->details->detection_threshold, hierarchy_threshold, 0, 1, &nboxes, 0);

	if (net->details->non_maximal_suppression_threshold)
	{
		auto & layer = net->layers[net->n - 1];
		do_nms_sort(darknet_results, nboxes, layer.classes, net->details->non_maximal_suppression_threshold);
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
			if (probability >= net->details->detection_threshold)
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

		// optional:  sometimes there are classes we want to completely ignore
		if (net->details->classes_to_ignore.count(pred.best_class))
		{
			continue;
		}

		if (net->details->fix_out_of_bound_normalized_coordinates)
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

	cv::Mat mat = cv::imread(image_filename.string());

	return predict(ptr, mat);
}


cv::Mat Darknet::annotate(const Darknet::NetworkPtr ptr, const Darknet::Predictions & predictions, cv::Mat mat)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot annotate without a network pointer");
	}

	if (mat.empty())
	{
		throw std::invalid_argument("cannot annotate empty image");
	}

	for (const auto & pred : predictions)
	{
		if (net->details->annotate_draw_bb)
		{
			// draw the bounding box around the entire object

			if (not net->details->bounding_boxes_with_rounded_corners)
			{
				cv::rectangle(mat, pred.rect, net->details->class_colours.at(pred.best_class), 1, net->details->cv_line_type);
			}
			else
			{
				draw_rounded_rectangle(mat, pred.rect, net->details->bounding_boxes_corner_roundness, net->details->class_colours.at(pred.best_class), net->details->cv_line_type);
			}
		}

		if (net->details->annotate_draw_label)
		{
			std::string text = net->details->class_names.at(pred.best_class) + " ";
			text += std::to_string(static_cast<int>(std::round(100.0f * pred.prob.at(pred.best_class)))) + "%";

			int				font_baseline	= 0;
			const cv::Size	size			= cv::getTextSize(text, net->details->cv_font_face, net->details->cv_font_scale, net->details->cv_font_thickness, &font_baseline);
			cv::Rect		label			= pred.rect;
			label.y							= label.y - size.height - font_baseline;
			label.height					= size.height + font_baseline;
			label.width						= size.width + 2;

			// draw a rectangle above that to use as a label
			cv::rectangle(mat, label, net->details->class_colours.at(pred.best_class), cv::FILLED, net->details->cv_line_type);

			cv::mean(net->details->class_colours.at(pred.best_class));

			// and finally we draw the text on top of the label background
			cv::putText(mat, text, cv::Point(label.x + 1, label.y + label.height - font_baseline / 2), net->details->cv_font_face, net->details->cv_font_scale, net->details->text_colours.at(pred.best_class), net->details->cv_font_thickness, net->details->cv_line_type);
		}
	}

	return mat;
}


Darknet::Predictions Darknet::predict_and_annotate(const Darknet::NetworkPtr ptr, cv::Mat mat)
{
	TAT(TATPARMS);

	const auto predictions = predict(ptr, mat);

	annotate(ptr, predictions, mat);

	return predictions;
}


const Darknet::VStr & Darknet::get_class_names(const Darknet::NetworkPtr ptr)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot get the class names without a network pointer");
	}

	if (net->details == nullptr)
	{
		throw std::invalid_argument("the network pointer was not initialized correctly (null details pointer!?)");
	}

	return net->details->class_names;
}


const Darknet::VScalars & Darknet::get_class_colours(const Darknet::NetworkPtr ptr)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot get the class colours without a network pointer");
	}

	if (net->details == nullptr)
	{
		throw std::invalid_argument("the network pointer was not initialized correctly (null details pointer!?)");
	}

	return net->details->class_colours;
}


const Darknet::VScalars & Darknet::set_class_colours(Darknet::NetworkPtr ptr, const Darknet::VScalars & user_colours)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot set the class colours without a network pointer");
	}

	if (net->details == nullptr)
	{
		throw std::invalid_argument("the network pointer was not initialized correctly (null details pointer!?)");
	}

	// Only copy over the necessary colours.  If the user provided us with too many colours, then we'll only read the first
	// few.  Meanwhile, if they gave us too-few then we'll take what we can and continue to use the default colours for the
	// remainder of the clases.

	auto & class_colours = net->details->class_colours;
	for (size_t idx = 0; idx < std::min(user_colours.size(), class_colours.size()); idx ++)
	{
		class_colours[idx] = user_colours[idx];
	}

	return net->details->class_colours;
}


std::filesystem::path Darknet::get_config_filename(const Darknet::NetworkPtr ptr)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot get the configuration filename without a network pointer");
	}

	return net->details->cfg_path;
}


std::filesystem::path Darknet::get_names_filename(const Darknet::NetworkPtr ptr)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot get the names filename without a network pointer");
	}

	return net->details->names_path;
}


std::filesystem::path Darknet::get_weights_filename(const Darknet::NetworkPtr ptr)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot get the weights filename without a network pointer");
	}

	return net->details->weights_path;
}


cv::Mat Darknet::resize_keeping_aspect_ratio(cv::Mat & mat, cv::Size desired_size, const cv::InterpolationFlags method)
{
	TAT(TATPARMS);

	const float width		= mat.cols;
	const float height		= mat.rows;
	const float horizontal	= width		/ desired_size.width;
	const float vertical	= height	/ desired_size.height;
	const float factor		= std::max(horizontal, vertical);

	const cv::Size new_size(std::round(width / factor), std::round(height / factor));

	cv::resize(mat, mat, new_size, method);

	return mat;
}


Darknet::SInt Darknet::skipped_classes(const Darknet::NetworkPtr ptr)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot get the skipped classes without a network pointer");
	}

	return net->details->classes_to_ignore;
}


Darknet::SInt Darknet::skipped_classes(Darknet::NetworkPtr ptr, const Darknet::SInt & classes_to_skip)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot set the skipped classes without a network pointer");
	}

	net->details->classes_to_ignore.clear();
	for (const int & idx : classes_to_skip)
	{
		if (idx >= 0 and idx < net->details->class_names.size())
		{
			net->details->classes_to_ignore.insert(idx);
		}
	}

	return net->details->classes_to_ignore;
}


Darknet::SInt Darknet::clear_skipped_classes(Darknet::NetworkPtr ptr)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot clear the skipped classes without a network pointer");
	}

	net->details->classes_to_ignore.clear();

	return net->details->classes_to_ignore;
}


Darknet::SInt Darknet::add_skipped_class(Darknet::NetworkPtr ptr, const int class_to_skip)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot add to the skipped classes without a network pointer");
	}

	if (class_to_skip >= 0 and class_to_skip < net->details->class_names.size())
	{
		net->details->classes_to_ignore.insert(class_to_skip);
	}

	return net->details->classes_to_ignore;
}


Darknet::SInt Darknet::del_skipped_class(Darknet::NetworkPtr ptr, const int class_to_include)
{
	TAT(TATPARMS);

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		throw std::invalid_argument("cannot remove from the skipped classes without a network pointer");
	}

	net->details->classes_to_ignore.erase(class_to_include);

	return net->details->classes_to_ignore;
}


std::ostream & Darknet::operator<<(std::ostream & os, const Darknet::EParmType & type)
{
	TAT(TATPARMS);

	switch (type)
	{
		case EParmType::kUnknown:			return os << "unknown";
		case EParmType::kCfgFilename:		return os << "config";
		case EParmType::kNamesFilename:		return os << "names";
		case EParmType::kWeightsFilename:	return os << "weights";
		case EParmType::kDirectory:			return os << "directory";
		case EParmType::kFilename:			return os << "filename";
		case EParmType::kOther:				return os << "other";
	}

	return os << "?";
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


std::string Darknet::format_duration_string(std::chrono::high_resolution_clock::duration duration, const int decimals, const uint32_t flags)
{
	TAT(TATPARMS);

	const float nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();

	// it is difficult to compare things when we have nanoseconds, microseconds, and milliseconds,
	// so if we have enough decimals to work with, prefer using milliseconds when possible

	std::stringstream ss;
	ss << std::fixed << std::setprecision(decimals) << std::setfill('0');

	if (duration <= std::chrono::high_resolution_clock::duration::zero())
	{
		ss << "unknown";
	}
	else if (decimals < 3 and duration <= std::chrono::microseconds(1))
	{
		/// @todo V5: why did we introduce this special case?
		ss << nanoseconds << " nanoseconds";
	}
	else if (decimals < 3 and duration <= std::chrono::microseconds(100))
	{
		/// @todo V5: why did we introduce this special case?
		const float microseconds = nanoseconds / 1000.0f;
		ss << microseconds << " microseconds";
	}
	else if (duration == std::chrono::milliseconds(1))
	{
		// this is ***EXACTLY*** 1 millisecond, so use singular not plural
		ss << 1.0f << " millisecond"; // singular, not plural
	}
	else if (duration < std::chrono::seconds(1))
	{
		const float milliseconds = nanoseconds / 1000000.0f;
		ss << milliseconds << " milliseconds";
	}
	else if (duration >= std::chrono::seconds(1) and duration < (std::chrono::seconds(1) + std::chrono::microseconds(500)))
	{
		// this is ***EXACTLY*** 1 second, or it would get rounded to "1 seconds", so use the singular "second"
		ss << 1.0f << " second"; // singular, not plural
	}
	else if (duration <= std::chrono::minutes(2))
	{
		const float seconds = nanoseconds / 1000000000.0f;
		ss << seconds << " seconds";
	}
	else if (duration <= std::chrono::hours(2))
	{
		const float minutes = nanoseconds / (60.0f * 1000000000.0f);
		ss << minutes << " minutes";
	}
	else if (duration <= std::chrono::hours(48))
	{
		const float hours = nanoseconds / (60.0f * 60.0f * 1000000000.0f);
		ss << hours << " hours";
	}
	else if (duration <= std::chrono::hours(24 * 7))
	{
		const float days = nanoseconds / (24.0f * 60.0f * 60.0f * 1000000000.0f);
		ss << days << " days";
	}
	else
	{
		const float weeks = nanoseconds / (7.0f * 24.0f * 60.0f * 60.0f * 1000000000.0f);
		ss << weeks << " weeks";
	}

	std::string str = ss.str();

	if (flags & EFormatDuration::kPad)
	{
		// insert some blank spaces at the front of the string to pad the number so things line up in columns
		auto pos = str.find('.');
		if (pos < 3)
		{
			str = std::string(3 - pos, ' ') + str;
		}
	}

	if (flags & EFormatDuration::kTrim)
	{
		// trim the trailing ".000" so numbers like "2.000 hours" show up as "2 hours"

		const std::string needle = "." + std::string(decimals, '0') + " ";
		auto pos = str.find(needle);
		if (pos != std::string::npos)
		{
			str.erase(pos, needle.size() - 1);
		}
	}

	return str;
}


std::ostream * Darknet::set_output_stream(const std::filesystem::path & filename)
{
	TAT(TATPARMS);

	if (cfg_and_state.output != &std::cout)
	{
		delete cfg_and_state.output;
		cfg_and_state.output = nullptr;
	}

	if (not filename.empty())
	{
		cfg_and_state.output = new std::ofstream(filename.string(), std::ofstream::out | std::ofstream::app);

		if (not cfg_and_state.output->good())
		{
			std::cout << "Failed to open output log file " << filename << std::endl; // must be std::cout since we failed to setup a log stream
			delete cfg_and_state.output;
			cfg_and_state.output = nullptr;
		}
		else
		{
			cfg_and_state.colour_is_enabled = false;

#if 0
			// Koshee has asked that we don't print the version and timestamp in the log file.
			const std::time_t current_time = std::time(nullptr);
			std::tm * tm = std::localtime(&current_time);
			char timestamp[100];
			std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S %z", tm);

			*cfg_and_state.output																		<< std::endl
				<< "========================="															<< std::endl
				<< timestamp																			<< std::endl
				<< "Darknet/YOLO " << DARKNET_VERSION_STRING << " output logging set to " << filename	<< std::endl
				<< "========================="															<< std::endl
				<< ""																					<< std::endl;
#else
			*cfg_and_state.output << std::endl;
#endif
		}
	}

	// if something has gone wrong, or no filename was given, then default to std::cout so the output pointer can always be de-referenced
	if (cfg_and_state.output == nullptr)
	{
		cfg_and_state.output = &std::cout;
	}

	// prefer using 500 over 5e+02 when outputting floats
	*cfg_and_state.output << std::fixed; // if this is changed, see CfgAndState::reset()

	return cfg_and_state.output;
}
