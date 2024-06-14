#include "Cfg.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


Darknet::CfgLine::CfgLine() :
	line_number(0),
	used(false)
{
	return;
}


Darknet::CfgLine::CfgLine(const std::string & l, const size_t ln, const std::string & lhs, const std::string & rhs) :
	line_number(ln),
	line(l),
	key(lhs),
	val(rhs),
	used(false)
{
	TAT(TATPARMS);

	if (val.size() > 0 and val.find_first_not_of("-.0123456789") == std::string::npos)
	{
		// looks like this might be a number!
		try
		{
			f = std::stof(val);
		}
		catch (...)
		{
			darknet_fatal_error(DARKNET_LOC, "line #%d: failed to parse numeric value from \"%s\"", line_number, l.c_str());
		}
	}

//	std::cout << "#" << line_number << ": line=\"" << line << "\" key=" << key << " val=" << val << std::endl;

	return;
};


Darknet::CfgLine::~CfgLine()
{
	TAT(TATPARMS);

	return;
}


std::string Darknet::CfgLine::debug() const
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss << line_number << ": key=" << key << " val=" << val;

	if (f.has_value())
	{
		ss << " f=" << f.value();
	}

	return ss.str();
}


Darknet::CfgSection::CfgSection(const std::string & l, const size_t ln) :
	type(Darknet::get_layer_from_name(l)),
	name(l),
	line_number(ln)
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgSection::~CfgSection()
{
	TAT(TATPARMS);

	return;
}


int Darknet::CfgSection::find_int(const std::string & key, const int default_value)
{
	auto iter = lines.find(key);
	if (iter != lines.end())
	{
		CfgLine & l = iter->second;
		if (l.f)
		{
			float iptr = 0.0f;
			float frac = std::modf(l.f.value(), &iptr);

			if (frac != 0.0f)
			{
				std::cout << "-> expected an integer for \"" << key << "\" on line #" << l.line_number << " but found \"" << l.f.value() << "\"" << std::endl;
			}

			const int i = static_cast<int>(iptr);

			if (cfg_and_state.is_verbose)
			{
				std::cout << "[" << name << "] #" << l.line_number << " " << key << "=" << i << std::endl;
			}

			l.used = true;
			return i;
		}
		else
		{
			darknet_fatal_error(DARKNET_LOC, "key \"%s\" on line #%ld should be numeric, but found \"%s\"", key.c_str(), l.line_number, l.val.c_str());
		}
	}

	if (cfg_and_state.is_verbose)
	{
		std::cout << "[" << name << "] #" << line_number << " DEFAULT " << key << "=" << default_value << std::endl;
	}

	return default_value;
}


float Darknet::CfgSection::find_float(const std::string & key, const float default_value)
{
	auto iter = lines.find(key);
	if (iter != lines.end())
	{
		CfgLine & l = iter->second;
		if (l.f)
		{
			const float f = l.f.value();

			if (cfg_and_state.is_verbose)
			{
				std::cout << "[" << name << "] #" << l.line_number << " " << key << "=" << f << std::endl;
			}

			l.used = true;
			return f;
		}
		else
		{
			darknet_fatal_error(DARKNET_LOC, "key \"%s\" on line #%ld should be numeric, but found \"%s\"", key.c_str(), l.line_number, l.val.c_str());
		}
	}

	if (cfg_and_state.is_verbose)
	{
		std::cout << "[" << name << "] #" << line_number << " DEFAULT " << key << "=" << default_value << std::endl;
	}

	return default_value;
}


std::string Darknet::CfgSection::find_str(const std::string & key, const std::string & default_value)
{
	auto iter = lines.find(key);
	if (iter != lines.end())
	{
		CfgLine & l = iter->second;

		if (cfg_and_state.is_verbose)
		{
			std::cout << "[" << name << "] #" << l.line_number << " " << key << "=" << l.val << std::endl;
		}

		l.used = true;
		return l.val;
	}

	if (cfg_and_state.is_verbose)
	{
		std::cout << "[" << name << "] #" << line_number << " DEFAULT " << key << "=" << default_value << std::endl;
	}

	return default_value;
}


Darknet::VFloat Darknet::CfgSection::find_float_array(const std::string & key)
{
	VFloat v;
	auto line = line_number;

	auto iter = lines.find(key);
	if (iter != lines.end())
	{
		CfgLine & l = iter->second;
		line = l.line_number;
		auto val = l.val;

		size_t pos = 0;
		while (pos < val.size())
		{
			const size_t digit = val.find_first_of("-.0123456789", pos);
			if (digit == std::string::npos)
			{
				// no numbers left to read
				break;
			}

			const size_t comma = val.find_first_not_of("-.0123456789", digit);
			std::string tmp;
			if (comma == std::string::npos)
			{
				tmp = val.substr(digit);
			}
			else
			{
				tmp = val.substr(digit, comma - digit);
			}

			try
			{
				const float f = std::stof(tmp);
				v.push_back(f);
			}
			catch(...)
			{
				break;
			}

			pos = comma;
		}
	}

	if (cfg_and_state.is_verbose)
	{
		std::cout << "[" << name << "] #" << line << " " << key << "=[";
		for (size_t idx = 0; idx < v.size(); idx ++)
		{
			if (idx > 0) std::cout << ", ";
			std::cout << v[idx];
		}
		std::cout << "]" << std::endl;
	}

	return v;
}


std::string Darknet::CfgSection::debug() const
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss << line_number << ": [" << get_name_from_layer(type) << "]" << std::endl;

	for (const auto & [key, line] : lines)
	{
		ss << line.debug() << std::endl;
	}

	return ss.str();
}


Darknet::CfgFile::CfgFile() :
	total_lines(0)
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgFile::CfgFile(const std::filesystem::path & fn) :
	CfgFile()
{
	TAT(TATPARMS);

	read(fn);

	return;
}


Darknet::CfgFile::~CfgFile()
{
	TAT(TATPARMS);

	return;
}


Darknet::CfgFile & Darknet::CfgFile::read(const std::filesystem::path & fn)
{
	TAT(TATPARMS);

	filename = fn;

	return read();
}


Darknet::CfgFile & Darknet::CfgFile::read()
{
	TAT(TATPARMS);

	total_lines = 0;
	sections.clear();

	if (not std::filesystem::exists(filename))
	{
		darknet_fatal_error(DARKNET_LOC, "config file does not exist: \"%s\"", filename.string());
	}

	filename = std::filesystem::canonical(filename);

	/* find lines such as these:
	 *
	 *		[net]
	 *		width=224
	 *		height=160 # this is a test
	 *		channels = 3
	 *		momentum=0.9
	 *
	 * ...and parse the following 3 things:
	 *
	 *		1) The name of the sections, such as "net" in "[net]"
	 *		2) The name of the key, such as "channels" in "channels=3"
	 *		3) The name of the value, such as "0.9" in "momentum=0.9"
	 */
	const std::regex rx(
		"^"				// start of line
		"\\["			// [
		"(.+)"			// group #1:  section name
		"\\]"			// ]
		"|"				// ...or...
		"^"				// start of line
		"[ \t]*"		// optional whitespace
		"("				// group #2
		"[^#= \t]+"		// key (everything up to #, =, or whitespace)
		")"				// end of group #2
		"[ \t]*"		// optional whitespace
		"="				// =
		"[ \t]*"		// optional whitespace
		"("				// group #3
		"[^#]+"			// value (everything up to #)
		")"				// end of group #3
		);

	std::ifstream ifs(filename);

	std::string line;
	while (std::getline(ifs, line))
	{
		total_lines ++;

		std::smatch matches;
		if (std::regex_match(line, matches, rx))
		{
			// line is either a section name, or a key-value pair -- check for a section name first

			const std::string section_name = matches.str(1);
			if (section_name.size() > 0)
			{
				CfgSection s(section_name, total_lines);
				sections.push_back(s);
			}
			else
			{
				// a section must exist prior to reading a key=val pair
				if (sections.empty())
				{
					darknet_fatal_error(DARKNET_LOC, "cannot add a configuration line without a section at line #%ld in %s", total_lines, filename.string().c_str());
				}

				const std::string key = convert_to_lowercase_alphanum(matches.str(2));
				const std::string val = matches.str(3);

				auto & s = sections.back();

				/* 2024-06-13:  As far as I know, each section should have unique keys.  The lines in each sections are stored in a
				 * map, meaning duplicates are not allowed.  If this is not the case and a particular layer can have duplicate keys
				 * (lines) then we'll have to change out the map for a vector, and fix how the "lines" processing works.  For now,
				 * do a quick check to see if this key already exists in this section, and abort loading the configuration.
				 */
				if (s.lines.count(key))
				{
					const auto iter = s.lines.find(key);
					darknet_fatal_error(DARKNET_LOC, "duplicate key \"%s\" in [%s] at lines #%ld and #%ld in %s", key.c_str(), s.name.c_str(), iter->second.line_number, total_lines, filename.string().c_str());
				}

				s.lines[key] = CfgLine(line, total_lines, key, val);
			}
		}
	}

	return *this;
}


std::string Darknet::CfgFile::debug() const
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss	<< "Filename: " << filename.string() << std::endl
		<< "Lines parsed: " << total_lines << std::endl;

	for (const auto & section : sections)
	{
		ss << section.debug() << std::endl;
	}

	return ss.str();
}


network * Darknet::CfgFile::create_network()
{
	TAT(TATPARMS);

	if (empty())
	{
		darknet_fatal_error(DARKNET_LOC, "darknet configuration cannot be created from an empty configuration");
	}

	// taken from:  load_network_custom()

	network * net = (network*)xcalloc(1, sizeof(network));

	// taken from:  parse_network_cfg_custom()
//	create_network(net, batch, 1);
	create_network(*net);
//	*net = parse_network_cfg_custom(cfg, batch, 1);

#if 0
	if (weights && weights[0] != 0)
	{
		load_weights(net, weights);
	}

	/// @todo What does this next line do!?  Why does load_network_custom() call this, but not load_network()?
	fuse_conv_batchnorm(*net);

	if (clear)
	{
		(*net->seen) = 0;
		(*net->cur_iteration) = 0;
	}
#endif

	return net;
}


namespace
{
	struct Size_Params
	{
		int batch;
		int inputs;
		int h;
		int w;
		int c;
		int index;
		int time_steps;
		int train;
		network net;
	};
}


network Darknet::CfgFile::create_network(network & net, int batch, int time_steps)
{
	TAT(TATPARMS);

	// taken from:  parse_network_cfg_custom()

	if (sections.empty())
	{
		darknet_fatal_error(DARKNET_LOC, "cannot create a network from empty configuration file \"%s\"", filename.string().c_str());
	}

	if (sections[0].type != ELayerType::NETWORK)
	{
		darknet_fatal_error(DARKNET_LOC, "first section in configuration should be [net] or [network], not [%s]", sections[0].name.c_str());
	}

	// this will overwrite whatever is in "net" with a bunch of new pointers;
	// the [net] is not a layer which is why we decrement
	net = make_network(sections.size() - 1);
	net.gpu_index = cfg_and_state.gpu_index;

	Size_Params params = {0};

	if (batch > 0)
	{
		// allocates memory for inference only
		params.train = 0;
	}
	else
	{
		// allocates memory for inference & training
		params.train = 1;
	}

	parse_net_section(0, net);


#if 0 // SC

	Size_Params params;

	for (const auto & section : sections)
	{
		switch (section.type)
		{
			case ELayerType::CONVOLUTIONAL:		{	l = parse_convolutional(options, params);							break;	}
			case ELayerType::LOCAL:				{	l = parse_local(options, params);									break;	}
			case ELayerType::ACTIVE:			{	l = parse_activation(options, params);								break;	}
			case ELayerType::RNN:				{	l = parse_rnn(options, params);										break;	}
			case ELayerType::GRU:				{	l = parse_gru(options, params);										break;	}
			case ELayerType::LSTM:				{	l = parse_lstm(options, params);									break;	}
			case ELayerType::CONV_LSTM:			{	l = parse_conv_lstm(options, params);								break;	}
			case ELayerType::HISTORY:			{	l = parse_history(options, params);									break;	}
			case ELayerType::CRNN:				{	l = parse_crnn(options, params);									break;	}
			case ELayerType::CONNECTED:			{	l = parse_connected(options, params);								break;	}
			case ELayerType::CROP:				{	l = parse_crop(options, params);									break;	}
			case ELayerType::IMPLICIT:			{	l = parse_implicit(options, params, net);							break;	}
			case ELayerType::DETECTION:			{	l = parse_detection(options, params);								break;	}
			case ELayerType::NORMALIZATION:		{	l = parse_normalization(options, params);							break;	}
			case ELayerType::BATCHNORM:			{	l = parse_batchnorm(options, params);								break;	}
			case ELayerType::MAXPOOL:			{	l = parse_maxpool(options, params);									break;	}
			case ELayerType::LOCAL_AVGPOOL:		{	l = parse_local_avgpool(options, params);							break;	}
			case ELayerType::REORG:				{	l = parse_reorg(options, params);									break;	}
			case ELayerType::REORG_OLD:			{	l = parse_reorg_old(options, params);								break;	}
			case ELayerType::AVGPOOL:			{	l = parse_avgpool(options, params);									break;	}
			case ELayerType::UPSAMPLE:			{	l = parse_upsample(options, params, net);							break;	}

			case ELayerType::COST:				{	l = parse_cost(options, params);			l.keep_delta_gpu = 1;	break;	}
			case ELayerType::REGION:			{	l = parse_region(options, params);			l.keep_delta_gpu = 1;	break;	}
			case ELayerType::YOLO:				{	l = parse_yolo(options, params);			l.keep_delta_gpu = 1;	break;	}
			case ELayerType::GAUSSIAN_YOLO:		{	l = parse_gaussian_yolo(options, params);	l.keep_delta_gpu = 1;	break;	}
			case ELayerType::CONTRASTIVE:		{	l = parse_contrastive(options, params);		l.keep_delta_gpu = 1;	break;	}

			case ELayerType::SOFTMAX:
			{
				l = parse_softmax(options, params);
				net.hierarchy = l.softmax_tree;
				l.keep_delta_gpu = 1;
				break;
			}
			case ELayerType::ROUTE:
			{
				l = parse_route(options, params);
				for (int k = 0; k < l.n; ++k)
				{
					net.layers[l.input_layers[k]].use_bin_output = 0;
					if (count >= last_stop_backward)
					{
						net.layers[l.input_layers[k]].keep_delta_gpu = 1;
					}
				}
				break;
			}
			case ELayerType::SHORTCUT:
			{
				l = parse_shortcut(options, params, net);
				net.layers[count - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				if (count >= last_stop_backward)
				{
					net.layers[l.index].keep_delta_gpu = 1;
				}
				break;
			}
			case ELayerType::SCALE_CHANNELS:
			{
				l = parse_scale_channels(options, params, net);
				net.layers[count - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				net.layers[l.index].keep_delta_gpu = 1;
				break;
			}
			case ELayerType::SAM:
			{
				l = parse_sam(options, params, net);
				net.layers[count - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				net.layers[l.index].keep_delta_gpu = 1;
				break;
			}
			case ELayerType::DROPOUT:
			{
				l = parse_dropout(options, params);
				l.output = net.layers[count-1].output;
				l.delta = net.layers[count-1].delta;
				#ifdef GPU
				l.output_gpu = net.layers[count-1].output_gpu;
				l.delta_gpu = net.layers[count-1].delta_gpu;
				l.keep_delta_gpu = 1;
				#endif
				break;
			}
			case ELayerType::EMPTY:
			{
				layer empty_layer = {(LAYER_TYPE)0};
				l = empty_layer;
				l.type = EMPTY;
				l.w = l.out_w = params.w;
				l.h = l.out_h = params.h;
				l.c = l.out_c = params.c;
				l.batch = params.batch;
				l.inputs = l.outputs = params.inputs;
				l.output = net.layers[count - 1].output;
				l.delta = net.layers[count - 1].delta;
				l.forward = empty_func;
				l.backward = empty_func;
				#ifdef GPU
				l.output_gpu = net.layers[count - 1].output_gpu;
				l.delta_gpu = net.layers[count - 1].delta_gpu;
				l.keep_delta_gpu = 1;
				l.forward_gpu = empty_func;
				l.backward_gpu = empty_func;
				#endif
				fprintf(stderr, "empty \n");
				break;
			}
			default:
			{
				darknet_fatal_error(DARKNET_LOC, "layer type not recognized: \"%s\"", s->type);
			}
		}
	}

	const auto & net_section = sections[0];

	if (batch > 0) params.train = 0;    // allocates memory for Inference only
	else params.train = 1;              // allocates memory for Inference & Training

	section *s = (section *)n->val;
	list *options = s->options;
	if(!is_network(s))
	{
		darknet_fatal_error(DARKNET_LOC, "first section must be [net] or [network]");
	}
	parse_net_options(options, &net);

	#ifdef GPU
	printf("net.optimized_memory = %d \n", net.optimized_memory);
	if (net.optimized_memory >= 2 && params.train)
	{
		pre_allocate_pinned_memory((size_t)1024 * 1024 * 1024 * 8);   // pre-allocate 8 GB CPU-RAM for pinned memory
	}
	#endif  // GPU

	params.h = net.h;
	params.w = net.w;
	params.c = net.c;
	params.inputs = net.inputs;
	if (batch > 0)					net.batch		= batch;
	if (time_steps > 0)				net.time_steps	= time_steps;
	if (net.batch < 1)				net.batch		= 1;
	if (net.time_steps < 1)			net.time_steps	= 1;
	if (net.batch < net.time_steps)	net.batch		= net.time_steps;
	params.batch		= net.batch;
	params.time_steps	= net.time_steps;
	params.net			= net;

	printf("mini_batch = %d, batch = %d, time_steps = %d, train = %d \n", net.batch, net.batch * net.subdivisions, net.time_steps, params.train);

	int last_stop_backward = -1;
	int avg_outputs = 0;
	int avg_counter = 0;
	float bflops = 0;
	size_t workspace_size = 0;
	size_t max_inputs = 0;
	size_t max_outputs = 0;
	int receptive_w = 1, receptive_h = 1;
	int receptive_w_scale = 1, receptive_h_scale = 1;
	const int show_receptive_field = option_find_float_quiet(options, "show_receptive_field", 0);

	n = n->next;
	int count = 0;
	free_section(s);

	// find l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
	node *n_tmp = n;
	int count_tmp = 0;
	if (params.train == 1)
	{
		while (n_tmp)
		{
			s = (section *)n_tmp->val;
			options = s->options;
			int stopbackward = option_find_int_quiet(options, "stopbackward", 0);
			if (stopbackward == 1)
			{
				last_stop_backward = count_tmp;
				printf("last_stop_backward = %d \n", last_stop_backward);
			}
			n_tmp = n_tmp->next;
			++count_tmp;
		}
	}

	int old_params_train = params.train;

	fprintf(stderr, "   layer   filters  size/strd(dil)      input                output\n");
	while(n)
	{
		params.train = old_params_train;
		if (count < last_stop_backward) params.train = 0;

		params.index = count;
		fprintf(stderr, "%4d ", count);
		s = (section *)n->val;
		options = s->options;
		layer l = { (LAYER_TYPE)0 };
		LAYER_TYPE lt = string_to_layer_type(s->type);
		if(lt == CONVOLUTIONAL){
			l = parse_convolutional(options, params);
		}else if(lt == LOCAL){
			l = parse_local(options, params);
		}else if(lt == ACTIVE){
			l = parse_activation(options, params);
		}else if(lt == RNN){
			l = parse_rnn(options, params);
		}else if(lt == GRU){
			l = parse_gru(options, params);
		}else if(lt == LSTM){
			l = parse_lstm(options, params);
		}else if (lt == CONV_LSTM) {
			l = parse_conv_lstm(options, params);
		}else if (lt == HISTORY) {
			l = parse_history(options, params);
		}else if(lt == CRNN){
			l = parse_crnn(options, params);
		}else if(lt == CONNECTED){
			l = parse_connected(options, params);
		}else if(lt == CROP){
			l = parse_crop(options, params);
		}else if(lt == COST){
			l = parse_cost(options, params);
			l.keep_delta_gpu = 1;
		}else if(lt == REGION){
			l = parse_region(options, params);
			l.keep_delta_gpu = 1;
		}else if (lt == YOLO) {
			l = parse_yolo(options, params);
			l.keep_delta_gpu = 1;
		}else if (lt == GAUSSIAN_YOLO) {
			l = parse_gaussian_yolo(options, params);
			l.keep_delta_gpu = 1;
		}else if(lt == DETECTION){
			l = parse_detection(options, params);
		}else if(lt == SOFTMAX){
			l = parse_softmax(options, params);
			net.hierarchy = l.softmax_tree;
			l.keep_delta_gpu = 1;
		}else if (lt == CONTRASTIVE) {
			l = parse_contrastive(options, params);
			l.keep_delta_gpu = 1;
		}else if(lt == NORMALIZATION){
			l = parse_normalization(options, params);
		}else if(lt == BATCHNORM){
			l = parse_batchnorm(options, params);
		}else if(lt == MAXPOOL){
			l = parse_maxpool(options, params);
		}else if (lt == LOCAL_AVGPOOL) {
			l = parse_local_avgpool(options, params);
		}else if(lt == REORG){
			l = parse_reorg(options, params);        }
			else if (lt == REORG_OLD) {
				l = parse_reorg_old(options, params);
			}else if(lt == AVGPOOL){
				l = parse_avgpool(options, params);
			}else if(lt == ROUTE){
				l = parse_route(options, params);
				int k;
				for (k = 0; k < l.n; ++k) {
					net.layers[l.input_layers[k]].use_bin_output = 0;
					if (count >= last_stop_backward)
						net.layers[l.input_layers[k]].keep_delta_gpu = 1;
				}
			}else if (lt == UPSAMPLE) {
				l = parse_upsample(options, params, net);
			}else if(lt == SHORTCUT){
				l = parse_shortcut(options, params, net);
				net.layers[count - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				if (count >= last_stop_backward)
					net.layers[l.index].keep_delta_gpu = 1;
			}else if (lt == SCALE_CHANNELS) {
				l = parse_scale_channels(options, params, net);
				net.layers[count - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				net.layers[l.index].keep_delta_gpu = 1;
			}
			else if (lt == SAM) {
				l = parse_sam(options, params, net);
				net.layers[count - 1].use_bin_output = 0;
				net.layers[l.index].use_bin_output = 0;
				net.layers[l.index].keep_delta_gpu = 1;
			} else if (lt == IMPLICIT) {
				l = parse_implicit(options, params, net);
			}else if(lt == DROPOUT){
				l = parse_dropout(options, params);
				l.output = net.layers[count-1].output;
				l.delta = net.layers[count-1].delta;
				#ifdef GPU
				l.output_gpu = net.layers[count-1].output_gpu;
				l.delta_gpu = net.layers[count-1].delta_gpu;
				l.keep_delta_gpu = 1;
				#endif
			}
			else if (lt == EMPTY) {
				layer empty_layer = {(LAYER_TYPE)0};
				l = empty_layer;
				l.type = EMPTY;
				l.w = l.out_w = params.w;
				l.h = l.out_h = params.h;
				l.c = l.out_c = params.c;
				l.batch = params.batch;
				l.inputs = l.outputs = params.inputs;
				l.output = net.layers[count - 1].output;
				l.delta = net.layers[count - 1].delta;
				l.forward = empty_func;
				l.backward = empty_func;
				#ifdef GPU
				l.output_gpu = net.layers[count - 1].output_gpu;
				l.delta_gpu = net.layers[count - 1].delta_gpu;
				l.keep_delta_gpu = 1;
				l.forward_gpu = empty_func;
				l.backward_gpu = empty_func;
				#endif
				fprintf(stderr, "empty \n");
			}
			else
			{
				darknet_fatal_error(DARKNET_LOC, "layer type not recognized: \"%s\"", s->type);
			}

			// calculate receptive field
			if(show_receptive_field)
			{
				int dilation = max_val_cmp(1, l.dilation);
				int stride = max_val_cmp(1, l.stride);
				int size = max_val_cmp(1, l.size);

				if (l.type == UPSAMPLE || (l.type == REORG))
				{

					l.receptive_w = receptive_w;
					l.receptive_h = receptive_h;
					l.receptive_w_scale = receptive_w_scale = receptive_w_scale / stride;
					l.receptive_h_scale = receptive_h_scale = receptive_h_scale / stride;

				}
				else {
					if (l.type == ROUTE) {
						receptive_w = receptive_h = receptive_w_scale = receptive_h_scale = 0;
						int k;
						for (k = 0; k < l.n; ++k) {
							layer route_l = net.layers[l.input_layers[k]];
							receptive_w = max_val_cmp(receptive_w, route_l.receptive_w);
							receptive_h = max_val_cmp(receptive_h, route_l.receptive_h);
							receptive_w_scale = max_val_cmp(receptive_w_scale, route_l.receptive_w_scale);
							receptive_h_scale = max_val_cmp(receptive_h_scale, route_l.receptive_h_scale);
						}
					}
					else
					{
						int increase_receptive = size + (dilation - 1) * 2 - 1;// stride;
						increase_receptive = max_val_cmp(0, increase_receptive);

						receptive_w += increase_receptive * receptive_w_scale;
						receptive_h += increase_receptive * receptive_h_scale;
						receptive_w_scale *= stride;
						receptive_h_scale *= stride;
					}

					l.receptive_w = receptive_w;
					l.receptive_h = receptive_h;
					l.receptive_w_scale = receptive_w_scale;
					l.receptive_h_scale = receptive_h_scale;
				}
				//printf(" size = %d, dilation = %d, stride = %d, receptive_w = %d, receptive_w_scale = %d - ", size, dilation, stride, receptive_w, receptive_w_scale);

				int cur_receptive_w = receptive_w;
				int cur_receptive_h = receptive_h;

				fprintf(stderr, "%4d - receptive field: %d x %d \n", count, cur_receptive_w, cur_receptive_h);
			}

			#ifdef GPU
			// futher GPU-memory optimization: net.optimized_memory == 2
			l.optimized_memory = net.optimized_memory;
			if (net.optimized_memory == 1 && params.train && l.type != DROPOUT) {
				if (l.delta_gpu) {
					cuda_free(l.delta_gpu);
					l.delta_gpu = NULL;
				}
			} else if (net.optimized_memory >= 2 && params.train && l.type != DROPOUT)
			{
				if (l.output_gpu) {
					cuda_free(l.output_gpu);
					//l.output_gpu = cuda_make_array_pinned(l.output, l.batch*l.outputs); // l.steps
					l.output_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
				}
				if (l.activation_input_gpu) {
					cuda_free(l.activation_input_gpu);
					l.activation_input_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
				}

				if (l.x_gpu) {
					cuda_free(l.x_gpu);
					l.x_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
				}

				// maximum optimization
				if (net.optimized_memory >= 3 && l.type != DROPOUT) {
					if (l.delta_gpu) {
						cuda_free(l.delta_gpu);
						//l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
						//printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
					}
				}

				if (l.type == CONVOLUTIONAL) {
					set_specified_workspace_limit(&l, net.workspace_size_limit);   // workspace size limit 1 GB
				}
			}
			#endif // GPU

			l.clip = option_find_float_quiet(options, "clip", 0);
			l.dynamic_minibatch = net.dynamic_minibatch;
			l.onlyforward = option_find_int_quiet(options, "onlyforward", 0);
			l.dont_update = option_find_int_quiet(options, "dont_update", 0);
			l.burnin_update = option_find_int_quiet(options, "burnin_update", 0);
			l.stopbackward = option_find_int_quiet(options, "stopbackward", 0);
			l.train_only_bn = option_find_int_quiet(options, "train_only_bn", 0);
			l.dontload = option_find_int_quiet(options, "dontload", 0);
			l.dontloadscales = option_find_int_quiet(options, "dontloadscales", 0);
			l.learning_rate_scale = option_find_float_quiet(options, "learning_rate", 1);
			option_unused(options);

			if (l.stopbackward == 1) printf(" ------- previous layers are frozen ------- \n");

			net.layers[count] = l;
		if (l.workspace_size > workspace_size) workspace_size = l.workspace_size;
		if (l.inputs > max_inputs) max_inputs = l.inputs;
		if (l.outputs > max_outputs) max_outputs = l.outputs;
		free_section(s);
		n = n->next;
		++count;
		if(n){
			if (l.antialiasing) {
				params.h = l.input_layer->out_h;
				params.w = l.input_layer->out_w;
				params.c = l.input_layer->out_c;
				params.inputs = l.input_layer->outputs;
			}
			else {
				params.h = l.out_h;
				params.w = l.out_w;
				params.c = l.out_c;
				params.inputs = l.outputs;
			}
		}
		if (l.bflops > 0) bflops += l.bflops;

		if (l.w > 1 && l.h > 1) {
			avg_outputs += l.outputs;
			avg_counter++;
		}
	}

	if (last_stop_backward > -1) {
		int k;
		for (k = 0; k < last_stop_backward; ++k) {
			layer l = net.layers[k];
			if (l.keep_delta_gpu) {
				if (!l.delta) {
					net.layers[k].delta = (float*)xcalloc(l.outputs*l.batch, sizeof(float));
				}
				#ifdef GPU
				if (!l.delta_gpu) {
					net.layers[k].delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
				}
				#endif
			}

			net.layers[k].onlyforward = 1;
			net.layers[k].train = 0;
		}
	}

	free_list(sections);

	#ifdef GPU
	if (net.optimized_memory && params.train)
	{
		int k;
		for (k = 0; k < net.n; ++k) {
			layer l = net.layers[k];
			// delta GPU-memory optimization: net.optimized_memory == 1
			if (!l.keep_delta_gpu) {
				const size_t delta_size = l.outputs*l.batch; // l.steps
				if (net.max_delta_gpu_size < delta_size) {
					net.max_delta_gpu_size = delta_size;
					if (net.global_delta_gpu) cuda_free(net.global_delta_gpu);
					if (net.state_delta_gpu) cuda_free(net.state_delta_gpu);
					assert(net.max_delta_gpu_size > 0);
					net.global_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
					net.state_delta_gpu = (float *)cuda_make_array(NULL, net.max_delta_gpu_size);
				}
				if (l.delta_gpu) {
					if (net.optimized_memory >= 3) {}
					else cuda_free(l.delta_gpu);
				}
				l.delta_gpu = net.global_delta_gpu;
			}
			else {
				if (!l.delta_gpu) l.delta_gpu = (float *)cuda_make_array(NULL, l.outputs*l.batch);
			}

			// maximum optimization
			if (net.optimized_memory >= 3 && l.type != DROPOUT) {
				if (l.delta_gpu && l.keep_delta_gpu) {
					//cuda_free(l.delta_gpu);   // already called above
					l.delta_gpu = cuda_make_array_pinned_preallocated(NULL, l.batch*l.outputs); // l.steps
					//printf("\n\n PINNED DELTA GPU = %d \n", l.batch*l.outputs);
				}
			}

			net.layers[k] = l;
		}
	}
	#endif

	set_train_only_bn(net); // set l.train_only_bn for all required layers

	net.outputs = get_network_output_size(net);
	net.output = get_network_output(net);
	avg_outputs = avg_outputs / avg_counter;
	printf("Total BFLOPS %5.3f \n", bflops);
	printf("avg_outputs = %d \n", avg_outputs);
	#ifdef GPU
	get_cuda_stream();
	//get_cuda_memcpy_stream();
	if (cfg_and_state.gpu_index >= 0)
	{
		int size = get_network_input_size(net) * net.batch;
		net.input_state_gpu = cuda_make_array(0, size);
		if (cudaSuccess == cudaHostAlloc((void**)&net.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped)) net.input_pinned_cpu_flag = 1;
		else {
			cudaGetLastError(); // reset CUDA-error
			net.input_pinned_cpu = (float*)xcalloc(size, sizeof(float));
		}

		// pre-allocate memory for inference on Tensor Cores (fp16)
		*net.max_input16_size = 0;
		*net.max_output16_size = 0;
		if (net.cudnn_half) {
			*net.max_input16_size = max_inputs;
			CHECK_CUDA(cudaMalloc((void **)net.input16_gpu, *net.max_input16_size * sizeof(short))); //sizeof(half)
			*net.max_output16_size = max_outputs;
			CHECK_CUDA(cudaMalloc((void **)net.output16_gpu, *net.max_output16_size * sizeof(short))); //sizeof(half)
		}

		if (workspace_size)
		{
			std::cout << "Allocating workspace to transfer between CPU and GPU:  " << size_to_IEC_string(workspace_size) << std::endl;

			net.workspace = cuda_make_array(0, workspace_size / sizeof(float) + 1);
		}
		else
		{
			printf("Allocating workspace:  %s\n", size_to_IEC_string(workspace_size));
			net.workspace = (float*)xcalloc(1, workspace_size);
		}
	}
	#else
	if (workspace_size)
	{
		printf("Allocating workspace:  %s\n", size_to_IEC_string(workspace_size));
		net.workspace = (float*)xcalloc(1, workspace_size);
	}
	#endif

	LAYER_TYPE lt = net.layers[net.n - 1].type;
	if (lt == YOLO || lt == REGION || lt == DETECTION)
	{
		if (net.w % 32 != 0 ||
			net.h % 32 != 0 ||
			net.w < 32      ||
			net.h < 32      )
		{
			darknet_fatal_error(DARKNET_LOC, "width=%d and height=%d in cfg file must be divisible by 32 for YOLO networks", net.w, net.h);
		}
	}

#endif // SC

	return net;
}


Darknet::CfgFile & Darknet::CfgFile::parse_net_section(const size_t section_idx, network & net)
{
	TAT(TATPARMS);

	Darknet::CfgSection & s = sections.at(section_idx);

	net.max_batches = s.find_int("max_batches", 0);
	net.batch = s.find_int("batch",1);
	net.learning_rate = s.find_float("learning_rate", .001);
	net.learning_rate_min = s.find_float("learning_rate_min", .00001);
	net.batches_per_cycle = s.find_int("sgdr_cycle", net.max_batches);
	net.batches_cycle_mult = s.find_int("sgdr_mult", 2);
	net.momentum = s.find_float("momentum", .9);
	net.decay = s.find_float("decay", .0001);
	const int subdivs = s.find_int("subdivisions",1);
	net.time_steps = s.find_int("time_steps",1);
	net.track = s.find_int("track", 0);
	net.augment_speed = s.find_int("augment_speed", 2);
	net.init_sequential_subdivisions = net.sequential_subdivisions = s.find_int("sequential_subdivisions", subdivs);
	if (net.sequential_subdivisions > subdivs)
	{
		net.init_sequential_subdivisions = net.sequential_subdivisions = subdivs;
	}
	net.try_fix_nan = s.find_int("try_fix_nan", 0);
	net.batch /= subdivs;          // mini_batch
	const int mini_batch = net.batch;
	net.batch *= net.time_steps;  // mini_batch * time_steps
	net.subdivisions = subdivs;    // number of mini_batches

	net.weights_reject_freq = s.find_int("weights_reject_freq", 0);
	net.equidistant_point = s.find_int("equidistant_point", 0);
	net.badlabels_rejection_percentage = s.find_float("badlabels_rejection_percentage", 0);
	net.num_sigmas_reject_badlabels = s.find_float("num_sigmas_reject_badlabels", 0);
	net.ema_alpha = s.find_float("ema_alpha", 0);
	*net.badlabels_reject_threshold = 0;
	*net.delta_rolling_max = 0;
	*net.delta_rolling_avg = 0;
	*net.delta_rolling_std = 0;
	*net.seen = 0;
	*net.cur_iteration = 0;
	*net.cuda_graph_ready = 0;
	net.use_cuda_graph = s.find_int("use_cuda_graph", 0);
	net.loss_scale = s.find_float("loss_scale", 1);
	net.dynamic_minibatch = s.find_int("dynamic_minibatch", 0);
	net.optimized_memory = s.find_int("optimized_memory", 0);
	net.workspace_size_limit = (size_t)1024*1024 * s.find_float("workspace_size_limit_MB", 1024);  // 1024 MB by default

	net.adam = s.find_int("adam", 0);
	if (net.adam)
	{
		net.B1 = s.find_float("B1", .9);
		net.B2 = s.find_float("B2", .999);
		net.eps = s.find_float("eps", .000001);
	}

	net.h = s.find_int("height",0);
	net.w = s.find_int("width",0);
	net.c = s.find_int("channels",0);
	net.inputs = s.find_int("inputs", net.h * net.w * net.c);
	net.max_crop = s.find_int("max_crop",net.w*2);
	net.min_crop = s.find_int("min_crop",net.w);
	net.flip = s.find_int("flip", 1);
	net.blur = s.find_int("blur", 0);
	net.gaussian_noise = s.find_int("gaussian_noise", 0);
	net.mixup = s.find_int("mixup", 0);
	int cutmix = s.find_int("cutmix", 0);
	int mosaic = s.find_int("mosaic", 0);
	if (mosaic && cutmix)
	{
		net.mixup = 4;
	}
	else if (cutmix)
	{
		net.mixup = 2;
	}
	else if (mosaic)
	{
		net.mixup = 3;
	}
	net.letter_box = s.find_int("letter_box", 0);
	net.mosaic_bound = s.find_int("mosaic_bound", 0);
	net.contrastive = s.find_int("contrastive", 0);
	net.contrastive_jit_flip = s.find_int("contrastive_jit_flip", 0);
	net.contrastive_color = s.find_int("contrastive_color", 0);
	net.unsupervised = s.find_int("unsupervised", 0);
	if (net.contrastive && mini_batch < 2)
	{
		darknet_fatal_error(DARKNET_LOC, "mini_batch size (batch/subdivisions) should be higher than 1 for contrastive loss");
	}

	net.label_smooth_eps = s.find_float("label_smooth_eps", 0.0f);
	net.resize_step = s.find_float("resize_step", 32);
	net.attention = s.find_int("attention", 0);
	net.adversarial_lr = s.find_float("adversarial_lr", 0);
	net.max_chart_loss = s.find_float("max_chart_loss", 20.0);

	net.angle = s.find_float("angle", 0);
	net.aspect = s.find_float("aspect", 1);
	net.saturation = s.find_float("saturation", 1);
	net.exposure = s.find_float("exposure", 1);
	net.hue = s.find_float("hue", 0);
	net.power = s.find_float("power", 4);

	if (!net.inputs && !(net.h && net.w && net.c))
	{
		darknet_fatal_error(DARKNET_LOC, "no input parameters supplied");
	}

	const std::string policy = s.find_str("policy", "constant");
	net.policy = (
			policy == "random"		? RANDOM	:
			policy == "poly"		? POLY		:
			policy == "constant"	? CONSTANT	:
			policy == "step"		? STEP		:
			policy == "exp"			? EXP		:
			policy == "sigmoid"		? SIG		:
			policy == "steps"		? STEPS		:
			policy == "sgdr"		? SGDR		: CONSTANT);

	net.burn_in = s.find_int("burn_in", 0);

#ifdef GPU
	if (net.gpu_index >= 0)
	{
		char device_name[1024];
		int compute_capability = get_gpu_compute_capability(net.gpu_index, device_name);
#ifdef CUDNN_HALF
		if (compute_capability >= 700)
		{
			net.cudnn_half = 1;
		}
		else
		{
			net.cudnn_half = 0;
		}
#endif// CUDNN_HALF
		fprintf(stderr, " %d : compute_capability = %d, cudnn_half = %d, GPU: %s \n", net.gpu_index, compute_capability, net.cudnn_half, device_name);
	}
	else
	{
		fprintf(stderr, " GPU isn't used \n");
	}
#endif// GPU

	if (net.policy == STEP)
	{
		net.step = s.find_int("step", 1);
		net.scale = s.find_float("scale", 1);
	}
	else if (net.policy == STEPS || net.policy == SGDR)
	{
		auto steps		= s.find_float_array("steps");
		auto scales		= s.find_float_array("scales");
		auto seq_scales	= s.find_float_array("seq_scales");

		if (net.policy == STEPS && (steps.empty() || scales.empty()))
		{
			darknet_fatal_error(DARKNET_LOC, "STEPS policy must have steps and scales in cfg file");
		}

		// make sure all arrays are the same size
		int n = steps.size();
		scales.resize(n);
		seq_scales.resize(n);

		net.num_steps	= n;
		net.steps		= (int*)xcalloc(n, sizeof(int));
		net.scales		= (float*)xcalloc(n, sizeof(float));
		net.seq_scales	= (float*)xcalloc(n, sizeof(float));

		for (auto i = 0; i < steps.size(); i ++)
		{
			net.steps[i]		= static_cast<int>(steps[i]);
			net.scales[i]		= scales[i];
			net.seq_scales[i]	= seq_scales[i];
		}
	}
	else if (net.policy == EXP)
	{
		net.gamma = s.find_float("gamma", 1);
	}
	else if (net.policy == SIG)
	{
		net.gamma = s.find_float("gamma", 1);
		net.step = s.find_int("step", 1);
	}
	else if (net.policy == POLY || net.policy == RANDOM)
	{
		//net.power = s.find_float("power", 1);
	}

	return *this;
}
