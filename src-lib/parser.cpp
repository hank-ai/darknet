#include "option_list.hpp"
#include "darknet_internal.hpp"
#include "dump.hpp"

namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}

network parse_network_cfg(const char * filename)
{
	TAT(TATPARMS);

	return parse_network_cfg_custom(filename, 0, 0);
}

network parse_network_cfg_custom(const char * filename, int batch, int time_steps)
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


void save_convolutional_weights_binary(layer l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef GPU
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

void save_shortcut_weights(layer l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_shortcut_layer(l);
		printf("\n pull_shortcut_layer \n");
	}
#endif
	int i;
	//if (l.weight_updates) for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weight_updates[i]);
	//printf(" l.nweights = %d - update \n", l.nweights);
	for (i = 0; i < l.nweights; ++i)
	{
		printf(" %f, ", l.weights[i]);
	}
	printf(" l.nweights = %d \n\n", l.nweights);

	int num = l.nweights;
	fwrite(l.weights, sizeof(float), num, fp);
}

void save_implicit_weights(layer l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		pull_implicit_layer(l);
		//printf("\n pull_implicit_layer \n");
	}
#endif
	//int i;
	//if (l.weight_updates) for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weight_updates[i]);
	//printf(" l.nweights = %d - update \n", l.nweights);
	//for (i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
	//printf(" l.nweights = %d \n\n", l.nweights);

	int num = l.nweights;
	fwrite(l.weights, sizeof(float), num, fp);
}

void save_convolutional_weights(layer l, FILE *fp)
{
	TAT(TATPARMS);

	if (l.binary)
	{
		//save_convolutional_weights_binary(l, fp);
		//return;
	}
#ifdef GPU
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

void save_convolutional_weights_ema(layer l, FILE *fp)
{
	TAT(TATPARMS);

	if (l.binary)
	{
		//save_convolutional_weights_binary(l, fp);
		//return;
	}
#ifdef GPU
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

void save_batchnorm_weights(layer l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef GPU
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

void save_connected_weights(layer l, FILE *fp)
{
	TAT(TATPARMS);

#ifdef GPU
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

void save_weights_upto(network net, char *filename, int cutoff, int save_ema)
{
	TAT(TATPARMS);

#ifdef GPU
	if (net.gpu_index >= 0)
	{
		cuda_set_device(net.gpu_index);
	}
#endif

	std::cout << "Saving weights to " << Darknet::in_colour(Darknet::EColour::kBrightMagenta, filename) << std::endl;

	FILE *fp = fopen(filename, "wb");
	if (!fp)
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

	int i;
	for (i = 0; i < net.n && i < cutoff; ++i)
	{
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL && l.share_layer == NULL)
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
		if (l.type == SHORTCUT && l.nweights > 0)
		{
			save_shortcut_weights(l, fp);
		}
		if (l.type == IMPLICIT)
		{
			save_implicit_weights(l, fp);
		}
		if (l.type == CONNECTED)
		{
			save_connected_weights(l, fp);
		}
		if (l.type == BATCHNORM)
		{
			save_batchnorm_weights(l, fp);
		}
		if (l.type == RNN)
		{
			save_connected_weights(*(l.input_layer), fp);
			save_connected_weights(*(l.self_layer), fp);
			save_connected_weights(*(l.output_layer), fp);
		}
		if (l.type == GRU)
		{
			save_connected_weights(*(l.input_z_layer), fp);
			save_connected_weights(*(l.input_r_layer), fp);
			save_connected_weights(*(l.input_h_layer), fp);
			save_connected_weights(*(l.state_z_layer), fp);
			save_connected_weights(*(l.state_r_layer), fp);
			save_connected_weights(*(l.state_h_layer), fp);
		}
		if (l.type == LSTM)
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
		if (l.type == CONV_LSTM)
		{
			if (l.peephole)
			{
				save_convolutional_weights(*(l.vf), fp);
				save_convolutional_weights(*(l.vi), fp);
				save_convolutional_weights(*(l.vo), fp);
			}
			save_convolutional_weights(*(l.wf), fp);
			if (!l.bottleneck)
			{
				save_convolutional_weights(*(l.wi), fp);
				save_convolutional_weights(*(l.wg), fp);
				save_convolutional_weights(*(l.wo), fp);
			}
			save_convolutional_weights(*(l.uf), fp);
			save_convolutional_weights(*(l.ui), fp);
			save_convolutional_weights(*(l.ug), fp);
			save_convolutional_weights(*(l.uo), fp);
		}
		if (l.type == CRNN)
		{
			save_convolutional_weights(*(l.input_layer), fp);
			save_convolutional_weights(*(l.self_layer), fp);
			save_convolutional_weights(*(l.output_layer), fp);
		}
		if (l.type == LOCAL)
		{
#ifdef GPU
			if (cfg_and_state.gpu_index >= 0)
			{
				pull_local_layer(l);
			}
#endif
			int locations = l.out_w*l.out_h;
			int size = l.size*l.size*l.c*l.n*locations;
			fwrite(l.biases, sizeof(float), l.outputs, fp);
			fwrite(l.weights, sizeof(float), size, fp);
		}
	}
	fclose(fp);
}

void save_weights(network net, char *filename)
{
	TAT(TATPARMS);

	save_weights_upto(net, filename, net.n, 0);
}

void transpose_matrix(float *a, int rows, int cols)
{
	TAT(TATPARMS);

	float* transpose = (float*)xcalloc(rows * cols, sizeof(float));
	int x, y;
	for (x = 0; x < rows; ++x)
	{
		for (y = 0; y < cols; ++y)
		{
			transpose[y*rows + x] = a[x*cols + y];
		}
	}
	memcpy(a, transpose, rows*cols*sizeof(float));
	free(transpose);
}

void load_connected_weights(layer l, FILE *fp, int transpose)
{
	TAT(TATPARMS);

	fread(l.biases, sizeof(float), l.outputs, fp);
	fread(l.weights, sizeof(float), l.outputs*l.inputs, fp);
	if (transpose)
	{
		transpose_matrix(l.weights, l.inputs, l.outputs);
	}
	//printf("Biases: %f mean %f variance\n", mean_array(l.biases, l.outputs), variance_array(l.biases, l.outputs));
	//printf("Weights: %f mean %f variance\n", mean_array(l.weights, l.outputs*l.inputs), variance_array(l.weights, l.outputs*l.inputs));
	if (l.batch_normalize && (!l.dontloadscales))
	{
		fread(l.scales, sizeof(float), l.outputs, fp);
		fread(l.rolling_mean, sizeof(float), l.outputs, fp);
		fread(l.rolling_variance, sizeof(float), l.outputs, fp);
		//printf("Scales: %f mean %f variance\n", mean_array(l.scales, l.outputs), variance_array(l.scales, l.outputs));
		//printf("rolling_mean: %f mean %f variance\n", mean_array(l.rolling_mean, l.outputs), variance_array(l.rolling_mean, l.outputs));
		//printf("rolling_variance: %f mean %f variance\n", mean_array(l.rolling_variance, l.outputs), variance_array(l.rolling_variance, l.outputs));
	}
#ifdef GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_connected_layer(l);
	}
#endif
}

void load_batchnorm_weights(layer l, FILE *fp)
{
	TAT(TATPARMS);

	fread(l.biases, sizeof(float), l.c, fp);
	fread(l.scales, sizeof(float), l.c, fp);
	fread(l.rolling_mean, sizeof(float), l.c, fp);
	fread(l.rolling_variance, sizeof(float), l.c, fp);
#ifdef GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_batchnorm_layer(l);
	}
#endif
}

void load_convolutional_weights_binary(layer l, FILE *fp)
{
	TAT(TATPARMS);

	fread(l.biases, sizeof(float), l.n, fp);
	if (l.batch_normalize && (!l.dontloadscales))
	{
		fread(l.scales, sizeof(float), l.n, fp);
		fread(l.rolling_mean, sizeof(float), l.n, fp);
		fread(l.rolling_variance, sizeof(float), l.n, fp);
	}
	int size = (l.c / l.groups)*l.size*l.size;
	int i, j, k;
	for (i = 0; i < l.n; ++i)
	{
		float mean = 0;
		fread(&mean, sizeof(float), 1, fp);
		for (j = 0; j < size/8; ++j)
		{
			int index = i*size + j*8;
			unsigned char c = 0;
			fread(&c, sizeof(char), 1, fp);
			for (k = 0; k < 8; ++k)
			{
				if (j*8 + k >= size) break;
				l.weights[index + k] = (c & 1<<k) ? mean : -mean;
			}
		}
	}
#ifdef GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_convolutional_layer(l);
	}
#endif
}

void load_convolutional_weights(layer l, FILE *fp)
{
	TAT(TATPARMS);

	int num = l.nweights;
	int read_bytes;
	read_bytes = fread(l.biases, sizeof(float), l.n, fp);
	if (read_bytes > 0 && read_bytes < l.n)
	{
		printf("\n Warning: Unexpected end of weights-file! l.biases - l.index = %d \n", l.index);
	}
	//fread(l.weights, sizeof(float), num, fp); // as in connected layer
	if (l.batch_normalize && (!l.dontloadscales))
	{
		read_bytes = fread(l.scales, sizeof(float), l.n, fp);
		if (read_bytes > 0 && read_bytes < l.n)
		{
			printf("\n Warning: Unexpected end of weights-file! l.scales - l.index = %d \n", l.index);
		}
		read_bytes = fread(l.rolling_mean, sizeof(float), l.n, fp);
		if (read_bytes > 0 && read_bytes < l.n)
		{
			printf("\n Warning: Unexpected end of weights-file! l.rolling_mean - l.index = %d \n", l.index);
		}
		read_bytes = fread(l.rolling_variance, sizeof(float), l.n, fp);
		if (read_bytes > 0 && read_bytes < l.n)
		{
			printf("\n Warning: Unexpected end of weights-file! l.rolling_variance - l.index = %d \n", l.index);
		}
	}
	read_bytes = fread(l.weights, sizeof(float), num, fp);
	if (read_bytes > 0 && read_bytes < l.n)
	{
		printf("\n Warning: Unexpected end of weights-file! l.weights - l.index = %d \n", l.index);
	}
	//if (l.adam){
	//    fread(l.m, sizeof(float), num, fp);
	//    fread(l.v, sizeof(float), num, fp);
	//}
	//if (l.c == 3) scal_cpu(num, 1./256, l.weights, 1);
	if (l.flipped)
	{
		transpose_matrix(l.weights, (l.c/l.groups)*l.size*l.size, l.n);
	}
	//if (l.binary) binarize_weights(l.weights, l.n, (l.c/l.groups)*l.size*l.size, l.weights);
#ifdef GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_convolutional_layer(l);
	}
#endif
}

void load_shortcut_weights(layer l, FILE *fp)
{
	TAT(TATPARMS);

	int num = l.nweights;
	int read_bytes;
	read_bytes = fread(l.weights, sizeof(float), num, fp);
	if (read_bytes > 0 && read_bytes < num)
	{
		printf("\n Warning: Unexpected end of weights-file! l.weights - l.index = %d \n", l.index);
	}
	//for (int i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
	//printf(" read_bytes = %d \n\n", read_bytes);
#ifdef GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_shortcut_layer(l);
	}
#endif
}

void load_implicit_weights(layer l, FILE *fp)
{
	TAT(TATPARMS);

	int num = l.nweights;
	int read_bytes;
	read_bytes = fread(l.weights, sizeof(float), num, fp);
	if (read_bytes > 0 && read_bytes < num)
	{
		printf("\n Warning: Unexpected end of weights-file! l.weights - l.index = %d \n", l.index);
	}
	//for (int i = 0; i < l.nweights; ++i) printf(" %f, ", l.weights[i]);
	//printf(" read_bytes = %d \n\n", read_bytes);
#ifdef GPU
	if (cfg_and_state.gpu_index >= 0)
	{
		push_implicit_layer(l);
	}
#endif
}

void load_weights_upto(network * net, const char * filename, int cutoff)
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

	if (cfg_and_state.is_verbose)
	{
		std::cout << "Loading weights from \"" << filename << "\"" << std::endl;
	}

#ifdef GPU
	if (net->gpu_index >= 0)
	{
		cuda_set_device(net->gpu_index);
	}
#endif

	FILE *fp = fopen(filename, "rb");
	if (!fp)
	{
		file_error(filename, DARKNET_LOC);
	}

	int major;
	int minor;
	int revision;
	fread(&major	, sizeof(int), 1, fp);
	fread(&minor	, sizeof(int), 1, fp);
	fread(&revision	, sizeof(int), 1, fp);

	if ((major * 10 + minor) >= 2)
	{
//		printf("\n seen 64");
		uint64_t iseen = 0;
		fread(&iseen, sizeof(uint64_t), 1, fp);
		*net->seen = iseen;
	}
	else
	{
//		printf("\n seen 32");
		uint32_t iseen = 0;
		fread(&iseen, sizeof(uint32_t), 1, fp);
		*net->seen = iseen;
	}

	*net->cur_iteration = get_current_batch(*net);
//	printf(", trained: %.0f K-images (%.0f Kilo-batches_64) \n", (float)(*net->seen / 1000), (float)(*net->seen / 64000));
	int transpose = (major > 1000) || (minor > 1000);

	size_t layers_with_weights = 0;

	for (int i = 0; i < net->n && i < cutoff; ++i)
	{
		layer & l = net->layers[i];
		if (l.dontload)
		{
			continue;
		}

		switch(l.type)
		{
			case CONVOLUTIONAL:
			{
				if (l.share_layer == NULL)
				{
					layers_with_weights ++;
					load_convolutional_weights(l, fp);
				}
				break;
			}
			case SHORTCUT:
			{
				if (l.nweights > 0)
				{
					layers_with_weights ++;
					load_shortcut_weights(l, fp);
				}
				break;
			}
			case IMPLICIT:
			{
				layers_with_weights ++;
				load_implicit_weights(l, fp);
				break;
			}
			case CONNECTED:
			{
				layers_with_weights ++;
				load_connected_weights(l, fp, transpose);
				break;
			}
			case BATCHNORM:
			{
				layers_with_weights ++;
				load_batchnorm_weights(l, fp);
				break;
			}
			case CRNN:
			{
				layers_with_weights ++;
				load_convolutional_weights(*(l.input_layer), fp);
				load_convolutional_weights(*(l.self_layer), fp);
				load_convolutional_weights(*(l.output_layer), fp);
				break;
			}
			case RNN:
			{
				layers_with_weights ++;
				load_connected_weights(*(l.input_layer), fp, transpose);
				load_connected_weights(*(l.self_layer), fp, transpose);
				load_connected_weights(*(l.output_layer), fp, transpose);
				break;
			}
			case GRU:
			{
				layers_with_weights ++;
				load_connected_weights(*(l.input_z_layer), fp, transpose);
				load_connected_weights(*(l.input_r_layer), fp, transpose);
				load_connected_weights(*(l.input_h_layer), fp, transpose);
				load_connected_weights(*(l.state_z_layer), fp, transpose);
				load_connected_weights(*(l.state_r_layer), fp, transpose);
				load_connected_weights(*(l.state_h_layer), fp, transpose);
				break;
			}
			case LSTM:
			{
				layers_with_weights ++;
				load_connected_weights(*(l.wf), fp, transpose);
				load_connected_weights(*(l.wi), fp, transpose);
				load_connected_weights(*(l.wg), fp, transpose);
				load_connected_weights(*(l.wo), fp, transpose);
				load_connected_weights(*(l.uf), fp, transpose);
				load_connected_weights(*(l.ui), fp, transpose);
				load_connected_weights(*(l.ug), fp, transpose);
				load_connected_weights(*(l.uo), fp, transpose);
				break;
			}
			case CONV_LSTM:
			{
				layers_with_weights ++;

				if (l.peephole)
				{
					load_convolutional_weights(*(l.vf), fp);
					load_convolutional_weights(*(l.vi), fp);
					load_convolutional_weights(*(l.vo), fp);
				}
				load_convolutional_weights(*(l.wf), fp);
				if (!l.bottleneck)
				{
					load_convolutional_weights(*(l.wi), fp);
					load_convolutional_weights(*(l.wg), fp);
					load_convolutional_weights(*(l.wo), fp);
				}
				load_convolutional_weights(*(l.uf), fp);
				load_convolutional_weights(*(l.ui), fp);
				load_convolutional_weights(*(l.ug), fp);
				load_convolutional_weights(*(l.uo), fp);
				break;
			}
			case LOCAL:
			{
				layers_with_weights ++;
				int locations = l.out_w*l.out_h;
				int size = l.size*l.size*l.c*l.n*locations;
				fread(l.biases, sizeof(float), l.outputs, fp);
				fread(l.weights, sizeof(float), size, fp);
#ifdef GPU
				if (cfg_and_state.gpu_index >= 0)
				{
					push_local_layer(l);
				}
#endif
				break;
			}
			default:
			{
				// this layer does not have weights to load
				continue;
			}
		}

		if (feof(fp))
		{
			Darknet::display_warning_msg("premature end-of-file reached while loading weights " + std::string(filename) + "\n");
			break;
		}
	}

	if (cfg_and_state.is_verbose)
	{
		std::cout << "Loaded weights for " << layers_with_weights << " of " << net->n << " layers from " << filename << std::endl;
	}

	fclose(fp);
}


void load_weights(network * net, const char * filename)
{
	TAT(TATPARMS);

	load_weights_upto(net, filename, net->n);
}


// load network & force - set batch size
network *load_network_custom(const char * cfg, const char * weights, int clear, int batch)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_verbose)
	{
		std::cout << "Loading configuration from \"" << cfg << "\"" << std::endl;
	}

	network* net = (network*)xcalloc(1, sizeof(network));
	*net = parse_network_cfg_custom(cfg, batch, 1);
	load_weights(net, weights);
	fuse_conv_batchnorm(*net);

	if (clear)
	{
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}

	return net;
}


// load network & get batch size from cfg-file
network *load_network(const char * cfg, const char * weights, int clear)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_verbose)
	{
		std::cout << "Loading configuration from \"" << cfg << "\"" << std::endl;
	}

	network* net = (network*)xcalloc(1, sizeof(network));
	*net = parse_network_cfg(cfg);
	load_weights(net, weights);

	/// @todo why do we not call fuse_conv_batchnorm() here?

	if (clear)
	{
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}

	return net;
}
