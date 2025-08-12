#include "option_list.hpp"
#include "darknet_internal.hpp"

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();


	/// @returns the total number of bytes read
	static inline size_t xfread(void * dst, const size_t size, const size_t count, std::FILE * fp, const std::string & description = "")
	{
		TAT(TATPARMS);

		if (dst == nullptr)
		{
			darknet_fatal_error(DARKNET_LOC, "attempting to load %lu %s, but destination pointer is NULL", count, description.c_str());
		}

		const auto items_read = std::fread(dst, size, count, fp);
		if (items_read != count)
		{
			Darknet::display_warning_msg(
				"The .weights file does not match the .cfg file (not enough fields to read in the weights).\n"
				"Normally this means the .weights file was corrupted, or you've mixed up which .cfg file goes with which .weights file.\n"
				"Another common problem is if you edit your .names file or .cfg file and you forget to re-train your network.\n");

			darknet_fatal_error(DARKNET_LOC, "expected to read %lu fields, but only read %lu", count, items_read);
		}

		if (cfg_and_state.is_trace)
		{
			*cfg_and_state.output << "-> read " << count << " x " << (size * 8) << "-bit values (" << size_to_IEC_string(size * count) << ")" << (description.empty() ? "" : " as " + description) << std::endl;
		}

		return size * count;
	}
}


Darknet::Network parse_network_cfg(const char * filename)
{
	TAT(TATPARMS);

	return parse_network_cfg_custom(filename, 0, 0);
}


Darknet::Network parse_network_cfg_custom(const char * filename, int batch, int time_steps)
{
	TAT(TATPARMS);

	if (filename == nullptr or filename[0] == '\0')
	{
		darknet_fatal_error(DARKNET_LOC, "expected a .cfg filename but got a NULL filename instead");
	}

	// V3 JAZZ:  we now use the new CfgFile class to load configuration

	Darknet::CfgFile cfg_file(filename);
	cfg_file.create_network(batch, time_steps);

	if (cfg_and_state.is_trace)
	{
		Darknet::dump(cfg_file);
	}

	return cfg_file.net;
}


void save_convolutional_weights_binary(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_convolutional_layer(l);
	}
#endif
	int size = (l.c/l.groups)*l.size*l.size;
	binarize_weights(l.weights, l.n, size, l.binary_weights);
	int i, j, k;
	fwrite(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance, sizeof(float), l.n, fp);
	}
	for (i = 0; i < l.n; ++i)
	{
		float mean = l.binary_weights[i*size];
		if (mean < 0)
		{
			mean = -mean;
		}
		fwrite(&mean, sizeof(float), 1, fp);
		for (j = 0; j < size/8; ++j)
		{
			int index = i*size + j*8;
			unsigned char c = 0;
			for (k = 0; k < 8; ++k)
			{
				if (j*8 + k >= size)
				{
					break;
				}
				if (l.binary_weights[index + k] > 0)
				{
					c = (c | 1<<k);
				}
			}
			fwrite(&c, sizeof(char), 1, fp);
		}
	}
}

void save_shortcut_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_shortcut_layer(l);
		*cfg_and_state.output << std::endl << "pull_shortcut_layer" << std::endl;
	}
#endif
	for (int i = 0; i < l.nweights; ++i)
	{
		*cfg_and_state.output << " " << l.weights[i] << ", ";
	}
	*cfg_and_state.output << "l.nweights=" << l.nweights << std::endl << std::endl;

	int num = l.nweights;
	fwrite(l.weights, sizeof(float), num, fp);
}

void save_convolutional_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	if (l.binary)
	{
		//save_convolutional_weights_binary(l, fp);
		//return;
	}
#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_convolutional_layer(l);
	}
#endif
	int num = l.nweights;
	fwrite(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance, sizeof(float), l.n, fp);
	}
	fwrite(l.weights, sizeof(float), num, fp);
	//if (l.adam){
	//    fwrite(l.m, sizeof(float), num, fp);
	//    fwrite(l.v, sizeof(float), num, fp);
	//}
}

void save_convolutional_weights_ema(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	if (l.binary)
	{
		//save_convolutional_weights_binary(l, fp);
		//return;
	}
#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_convolutional_layer(l);
	}
#endif
	int num = l.nweights;
	fwrite(l.biases_ema, sizeof(float), l.n, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales_ema, sizeof(float), l.n, fp);
		fwrite(l.rolling_mean, sizeof(float), l.n, fp);
		fwrite(l.rolling_variance, sizeof(float), l.n, fp);
	}
	fwrite(l.weights_ema, sizeof(float), num, fp);
	//if (l.adam){
	//    fwrite(l.m, sizeof(float), num, fp);
	//    fwrite(l.v, sizeof(float), num, fp);
	//}
}

void save_batchnorm_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_batchnorm_layer(l);
	}
#endif
	fwrite(l.biases, sizeof(float), l.c, fp);
	fwrite(l.scales, sizeof(float), l.c, fp);
	fwrite(l.rolling_mean, sizeof(float), l.c, fp);
	fwrite(l.rolling_variance, sizeof(float), l.c, fp);
}

void save_connected_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_connected_layer(l);
	}
#endif
	fwrite(l.biases, sizeof(float), l.outputs, fp);
	fwrite(l.weights, sizeof(float), l.outputs*l.inputs, fp);
	if (l.batch_normalize)
	{
		fwrite(l.scales, sizeof(float), l.outputs, fp);
		fwrite(l.rolling_mean, sizeof(float), l.outputs, fp);
		fwrite(l.rolling_variance, sizeof(float), l.outputs, fp);
	}
}

void save_weights_upto(const Darknet::Network & net, const char *filename, int cutoff, int save_ema)
{
	TAT(TATPARMS);

#ifdef DARKNET_GPU
	if (net.gpu_index >= 0)
	{
		cuda_set_device(net.gpu_index);
	}
#endif

	*cfg_and_state.output << "Saving weights to " << Darknet::in_colour(Darknet::EColour::kBrightMagenta, filename) << std::endl;

	FILE *fp = fopen(filename, "wb");
	if (not fp)
	{
		file_error(filename, DARKNET_LOC);
	}

	const int major = DARKNET_WEIGHTS_VERSION_MAJOR;
	const int minor = DARKNET_WEIGHTS_VERSION_MINOR;
	const int revision = DARKNET_WEIGHTS_VERSION_PATCH;

	fwrite(&major, sizeof(int), 1, fp);
	fwrite(&minor, sizeof(int), 1, fp);
	fwrite(&revision, sizeof(int), 1, fp);
	(*net.seen) = get_current_iteration(net) * net.batch * net.subdivisions; // remove this line, when you will save to weights-file both: seen & cur_iteration
	fwrite(net.seen, sizeof(uint64_t), 1, fp);

	for (int i = 0; i < net.n && i < cutoff; ++i)
	{
		Darknet::Layer & l = net.layers[i];
		if (l.type == Darknet::ELayerType::CONVOLUTIONAL && l.share_layer == NULL)
		{
			if (save_ema)
			{
				save_convolutional_weights_ema(l, fp);
			}
			else
			{
				save_convolutional_weights(l, fp);
			}
		}
		if (l.type == Darknet::ELayerType::SHORTCUT && l.nweights > 0)
		{
			save_shortcut_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::CONNECTED)
		{
			save_connected_weights(l, fp);
		}
		if (l.type == Darknet::ELayerType::RNN)
		{
			save_connected_weights(*(l.input_layer), fp);
			save_connected_weights(*(l.self_layer), fp);
			save_connected_weights(*(l.output_layer), fp);
		}
		if (l.type == Darknet::ELayerType::LSTM)
		{
			save_connected_weights(*(l.wf), fp);
			save_connected_weights(*(l.wi), fp);
			save_connected_weights(*(l.wg), fp);
			save_connected_weights(*(l.wo), fp);
			save_connected_weights(*(l.uf), fp);
			save_connected_weights(*(l.ui), fp);
			save_connected_weights(*(l.ug), fp);
			save_connected_weights(*(l.uo), fp);
		}
		if (l.type == Darknet::ELayerType::CRNN)
		{
			save_convolutional_weights(*(l.input_layer), fp);
			save_convolutional_weights(*(l.self_layer), fp);
			save_convolutional_weights(*(l.output_layer), fp);
		}
	}
	fclose(fp);
}


void save_weights(const Darknet::Network & net, const char *filename)
{
	TAT(TATPARMS);

	save_weights_upto(net, filename, net.n, 0);
}


void transpose_matrix(float *a, int rows, int cols)
{
	TAT(TATPARMS);

	float* transpose = (float*)xcalloc(rows * cols, sizeof(float));

	for (int x = 0; x < rows; ++x)
	{
		for (int y = 0; y < cols; ++y)
		{
			transpose[y*rows + x] = a[x*cols + y];
		}
	}
	memcpy(a, transpose, rows*cols*sizeof(float));
	free(transpose);
}


size_t load_connected_weights(Darknet::Layer & l, FILE *fp, int transpose)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;

	bytes_read += xfread(l.biases	, sizeof(float), l.outputs				, fp, "biases"	);
	bytes_read += xfread(l.weights	, sizeof(float), l.outputs * l.inputs	, fp, "weights"	);
	if (transpose)
	{
		transpose_matrix(l.weights, l.inputs, l.outputs);
	}
	if (l.batch_normalize && (not l.dontloadscales))
	{
		bytes_read += xfread(l.scales			, sizeof(float), l.outputs, fp, "scales"			);
		bytes_read += xfread(l.rolling_mean		, sizeof(float), l.outputs, fp, "rolling mean"		);
		bytes_read += xfread(l.rolling_variance	, sizeof(float), l.outputs, fp, "rolling variance"	);
	}
#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_connected_layer(l);
	}
#endif

	return bytes_read;
}


size_t load_convolutional_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;
	const int num = l.nweights;

	bytes_read += xfread(l.biases, sizeof(float), l.n, fp, "biases");
	if (l.batch_normalize && (not l.dontloadscales))
	{
		bytes_read += xfread(l.scales			, sizeof(float), l.n, fp, "scales"			);
		bytes_read += xfread(l.rolling_mean		, sizeof(float), l.n, fp, "rolling mean"	);
		bytes_read += xfread(l.rolling_variance	, sizeof(float), l.n, fp, "rolling variance");
	}
	bytes_read += xfread(l.weights, sizeof(float), num, fp, "weights");

	if (l.flipped)
	{
		transpose_matrix(l.weights, (l.c / l.groups) * l.size * l.size, l.n);
	}
	//if (l.binary) binarize_weights(l.weights, l.n, (l.c/l.groups)*l.size*l.size, l.weights);
#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_convolutional_layer(l);
	}
#endif

	return bytes_read;
}


size_t load_shortcut_weights(Darknet::Layer & l, FILE *fp)
{
	TAT(TATPARMS);

	size_t bytes_read = 0;
	int num = l.nweights;

	bytes_read += xfread(l.weights, sizeof(float), num, fp, "weights");

#ifdef DARKNET_GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_shortcut_layer(l);
	}
#endif

	return bytes_read;
}


void load_weights_upto(Darknet::Network * net, const char * filename, int cutoff)
{
	TAT(TATPARMS);

	if (net			== nullptr or
		filename	== nullptr or
		filename[0]	== '\0')
	{
		// nothing we can do
		Darknet::display_warning_msg("Cannot load weights due to NULL configuration or weights filename.\n");
		return;
	}

	if (net->details == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "network structure was not created correctly (details pointer is null!?)");
	}
	net->details->weights_path = filename;

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loading weights from \"" << filename << "\""
			<< " (" << size_to_IEC_string(std::filesystem::file_size(filename)) << ")"
			<< std::endl;
	}

#ifdef DARKNET_GPU
	if (net->gpu_index >= 0)
	{
		cuda_set_device(net->gpu_index);
	}
#endif

	FILE *fp = fopen(filename, "rb");
	if (not fp)
	{
		file_error(filename, DARKNET_LOC);
	}

	int major;
	int minor;
	int revision;
	xfread(&major	, sizeof(int), 1, fp, "major version number");
	xfread(&minor	, sizeof(int), 1, fp, "minor version number");
	xfread(&revision, sizeof(int), 1, fp, "patch version number");

	if ((major * 10 + minor) >= 2)
	{
		uint64_t iseen = 0;
		xfread(&iseen, sizeof(uint64_t), 1, fp, "images seen during training");
		*net->seen = iseen;
	}
	else
	{
		uint32_t iseen = 0;
		xfread(&iseen, sizeof(uint32_t), 1, fp, "images seen during training");
		*net->seen = iseen;
	}

	*net->cur_iteration = get_current_batch(*net);
	int transpose = (major > 1000) || (minor > 1000);

	size_t layers_with_weights = 0;
	size_t total_bytes_read = 0;

	for (int i = 0; i < net->n && i < cutoff; ++i)
	{
		Darknet::Layer & l = net->layers[i];
		if (l.dontload)
		{
			if (cfg_and_state.is_trace)
			{
				*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): dontload is set" << std::endl;
			}
			continue;
		}

		// also see Darknet::ONNXExport::populate_graph_initializers()
		switch(l.type)
		{
			case Darknet::ELayerType::CONVOLUTIONAL:
			{
				size_t bytes_read = 0;
				if (l.share_layer == NULL)
				{
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading convolutional weights" << std::endl;
					}
					layers_with_weights ++;
					bytes_read += load_convolutional_weights(l, fp);
				}
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::SHORTCUT:
			{
				size_t bytes_read = 0;
				if (l.nweights > 0)
				{
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading shortcut weights" << std::endl;
					}
					layers_with_weights ++;
					bytes_read += load_shortcut_weights(l, fp);
				}
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::CONNECTED:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading connected weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_connected_weights(l, fp, transpose);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::CRNN:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading convolutional weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_convolutional_weights(*(l.input_layer)	, fp);
				bytes_read += load_convolutional_weights(*(l.self_layer)	, fp);
				bytes_read += load_convolutional_weights(*(l.output_layer)	, fp);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::RNN:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading connected weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_connected_weights(*(l.input_layer)	, fp, transpose);
				bytes_read += load_connected_weights(*(l.self_layer)	, fp, transpose);
				bytes_read += load_connected_weights(*(l.output_layer)	, fp, transpose);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			case Darknet::ELayerType::LSTM:
			{
				size_t bytes_read = 0;
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): loading connected weights" << std::endl;
				}
				layers_with_weights ++;
				bytes_read += load_connected_weights(*(l.wf), fp, transpose);
				bytes_read += load_connected_weights(*(l.wi), fp, transpose);
				bytes_read += load_connected_weights(*(l.wg), fp, transpose);
				bytes_read += load_connected_weights(*(l.wo), fp, transpose);
				bytes_read += load_connected_weights(*(l.uf), fp, transpose);
				bytes_read += load_connected_weights(*(l.ui), fp, transpose);
				bytes_read += load_connected_weights(*(l.ug), fp, transpose);
				bytes_read += load_connected_weights(*(l.uo), fp, transpose);
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "-> layer #" << i << " (" << Darknet::to_string(l.type) << "): loaded " << size_to_IEC_string(bytes_read) << std::endl;
				}
				total_bytes_read += bytes_read;
				break;
			}
			default:
			{
				// this layer does not have weights to load
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << i << " (" << Darknet::to_string(l.type) << "): no weights to load" << std::endl;
				}
				continue;
			}
		}

		if (feof(fp))
		{
			Darknet::display_warning_msg("premature end-of-file reached while loading weights " + std::string(filename) + "\n");
			break;
		}
	}

	// if everything has gone well, there will be zero bytes left to read at this point
	const auto position = ftell(fp);
	const auto filesize = std::filesystem::file_size(filename);
	if (position != filesize and cutoff >= net->n)
	{
		Darknet::display_warning_msg(
			"The .weights file does not match the .cfg file (weights file is larger than expected as described in the configuration).\n"
			"Normally this means the .weights file was corrupted, or you've mixed up which .cfg file goes with which .weights file.\n"
			"Another common problem is if you edit your .names file or .cfg file and you forget to re-train your network.\n");

		darknet_fatal_error(DARKNET_LOC, "failure detected while reading weights (fn=%s, layers=%d, pos=%lu, filesize=%lu)", filename, net->n, position, filesize);
	}

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loaded " << size_to_IEC_string(total_bytes_read) << " in weights for " << layers_with_weights << " of " << net->n << " layers from " << filename << std::endl;
	}

	fclose(fp);

	return;
}


void load_weights(Darknet::Network * net, const char * filename)
{
	TAT(TATPARMS);

	load_weights_upto(net, filename, net->n);
}


// load network & force - set batch size
DarknetNetworkPtr load_network_custom(const char * cfg, const char * weights, int clear, int batch)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loading configuration from \"" << cfg << "\"" << std::endl;
	}

	Darknet::Network * net = (Darknet::Network*)xcalloc(1, sizeof(Darknet::Network));
	*net = parse_network_cfg_custom(cfg, batch, 1);
	load_weights(net, weights);
	fuse_conv_batchnorm(*net);

	/** @todo V3 Some code seems to also call this next function, and some not.  This was not originally called here, but
	 * I copied it from several other code locations.  Need to invetigate whether or not it should be here.  2024-08-03
	 */
	calculate_binary_weights(net);

	if (clear)
	{
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}

	return net;
}


// load network & get batch size from cfg-file
DarknetNetworkPtr load_network(const char * cfg, const char * weights, int clear)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loading configuration from \"" << cfg << "\"" << std::endl;
	}

	Darknet::Network* net = (Darknet::Network*)xcalloc(1, sizeof(Darknet::Network));
	*net = parse_network_cfg(cfg);
	load_weights(net, weights);

	/// @todo V3 why do we not call fuse_conv_batchnorm() here?

	if (clear)
	{
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}

	return net;
}


void Darknet::load_names(Darknet::NetworkPtr ptr, const std::filesystem::path & filename)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "Loading names from \"" << filename.string() << "\"" << std::endl;
	}

	if (not std::filesystem::exists(filename))
	{
		darknet_fatal_error(DARKNET_LOC, "expected a .names file but got a bad filename instead: \"%s\"", filename.string().c_str());
	}

	Darknet::Network * net = reinterpret_cast<Darknet::Network *>(ptr);
	if (net == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "cannot set .names to \"%s\" when network pointer is null", filename.string().c_str());
	}

	if (net->details == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "network structure was not created correctly (details pointer is null!?)");
	}

	net->details->names_path = filename;
	net->details->class_names.clear();

	std::string line;
	std::ifstream ifs(filename);
	while (std::getline(ifs, line))
	{
		// strip whitespace at the end of line, which should help us ignore \n and \r\n problems between Windows and Linux
		Darknet::trim(line);

		if (line.empty())
		{
			Darknet::display_error_msg("The .names file appears to contain a blank line.\n");
		}

		net->details->class_names.push_back(line);
	}

	if (net->layers[net->n - 1].classes != net->details->class_names.size())
	{
		darknet_fatal_error(DARKNET_LOC, "mismatch between number of classes in %s and the number of lines in %s", net->details->cfg_path.string().c_str(), net->details->names_path.string().c_str());
	}

	return;
}


void Darknet::assign_default_class_colours(Darknet::Network * net)
{
	TAT(TATPARMS);

	if (net == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "cannot assign class colours when the network pointer is null");
	}

	if (net->details == nullptr)
	{
		darknet_fatal_error(DARKNET_LOC, "network structure was not created correctly (details pointer is null!?)");
	}

	if (net->n < 1)
	{
		darknet_fatal_error(DARKNET_LOC, "no network layers exist (was the network loaded?)");
	}

	const auto number_of_classes = net->layers[net->n - 1].classes;

	if (net->details->class_names.empty())
	{
		// we may not have the network names available, so create fake labels we can use
		net->details->class_names.reserve(number_of_classes);
		for (int i = 0; i < number_of_classes; i++)
		{
			net->details->class_names.push_back("#" + std::to_string(i));
		}
	}

	if (number_of_classes != net->details->class_names.size())
	{
		darknet_fatal_error(DARKNET_LOC, "last layer indicates %d classes, but %ld classes exist", number_of_classes, net->details->class_names.size());
	}

	// assign a colour to each class

	net->details->text_colours	.clear();
	net->details->class_colours	.clear();
	net->details->text_colours	.reserve(number_of_classes);
	net->details->class_colours	.reserve(number_of_classes);

	for (int i = 0; i < number_of_classes; i++)
	{
		const int offset = i * 123457 % number_of_classes;
		const int r = std::min(255.0f, std::round(256.0f * Darknet::get_color(2, offset, number_of_classes)));
		const int g = std::min(255.0f, std::round(256.0f * Darknet::get_color(1, offset, number_of_classes)));
		const int b = std::min(255.0f, std::round(256.0f * Darknet::get_color(0, offset, number_of_classes)));

		const cv::Scalar background = CV_RGB(r, g, b);
		const cv::Scalar foreground = CV_RGB(0, 0, 0);

		net->details->class_colours.push_back(background);
		net->details->text_colours.push_back(foreground);

		if (cfg_and_state.is_verbose)
		{
			*cfg_and_state.output << "-> class #" << i << " will use colour 0x"
				<< std::setw(2) << std::setfill('0') << std::hex << r
				<< std::setw(2) << std::setfill('0') << std::hex << g
				<< std::setw(2) << std::setfill('0') << std::hex << b
				<< std::setw(1) << std::setfill(' ') << std::dec << std::endl;
		}
	}

	return;
}
