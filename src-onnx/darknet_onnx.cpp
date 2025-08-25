/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#include "darknet_onnx.hpp"


/** @file
 * Convert Darknet/YOLO .cfg and .weights files to .onnx files.
 *
 * @warning This code should not be trusted.  It was written in the summer of 2025 by Stephane Charette without a full
 * understanding of either the Darknet/YOLO internals, nor any reasonable understanding of the ONNX internals.  I
 * obtained some sample .onnx files from several sources on the web, and attempted to reverse engineer how these .onnx
 * files may have (!?) been put together from Darknet/YOLO weights.  I appologize for the cases where things are not
 * yet working as expected, and if it does happen to work I regret to say it is likely a mix of luck and happy
 * coincidences.
 *
 * Over time, I'm hoping other people will show up to help shine light in the dark corners, or provide me with more
 * configurations and weights that are broken to help make this tool work better.  As you'll no doubt see in the code
 * below, there are many places where I've left some "todo" comments in an attempt to document code that needs to be
 * fixed.
 *
 * Stephane Charette, 2025-08-18.
 */


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();


	size_t padding_len = 2;


	static std::string format_name(const size_t idx, const std::string & name)
	{
		TAT(TATPARMS);
		std::stringstream ss;
		ss << std::setfill('0') << std::setw(padding_len) << (idx) << "_" << name;
		return ss.str();
	}


	static std::string format_name(const size_t idx, const Darknet::ELayerType & type)
	{
		TAT(TATPARMS);
		return format_name(idx, Darknet::to_string(type));
	}


	static std::string format_name(const size_t idx, const Darknet::Layer & l)
	{
		TAT(TATPARMS);
		return format_name(idx, Darknet::to_string(l.type));
	}


	static std::string format_weights(const size_t idx, const Darknet::Layer & l, const std::string & name)
	{
		TAT(TATPARMS);
		return format_name(idx, l) + "_" + name;
	}

	static std::string ir_date_lookup(const int ir)
	{
		TAT(TATPARMS);

		switch (ir)
		{
			// check these dates, they came from ChatGPT
			case 3:		return "Aug 2017";
			case 4:		return "Jan 2019";
			case 5:		return "Mar 2019";
			case 6:		return "Sep 2019";
			case 7:		return "May 2020";
			case 8:		return "Jul 2021";
			case 9:		return "May 2023";
			case 10:	return "Mar 2024";
			case 11:	return "May 2025";
		}
		return "unknown";
	}

	static std::string ops_date_lookup(const int ops)
	{
		TAT(TATPARMS);

		switch (ops)
		{
			// check these dates, they came from ChatGPT
			case 8:		return "Apr 2018";
			case 9:		return "Dec 2018";
			case 10:	return "Apr 2019";
			case 11:	return "Sep 2019";
			case 12:	return "May 2020";
			case 13:	return "Aug 2020";
			case 14:	return "Aug 2020";
			case 15:	return "Jul 2021";
			case 16:	return "Feb 2022";
			case 17:	return "Jun 2022";
			case 18:	return "Dec 2022";
			case 19:	return "May 2023";
			case 20:	return "Oct 2023";
			case 21:	return "Mar 2024";
			case 22:	return "Jan 2025";
			case 23:	return "May 2025";
		}
		return "unknown";
	}
}


void Darknet::ONNXExport::log_handler(google::protobuf::LogLevel level, const char * filename, int line, const std::string & message)
{
	TAT(TATPARMS);

	*cfg_and_state.output << "Protocol buffer error detected:"
		<< " level="	<< level
		<< " fn="		<< filename
		<< " line="		<< line
		<< " msg="		<< message
		<< std::endl;

	if (level == google::protobuf::LOGLEVEL_ERROR or
		level == google::protobuf::LOGLEVEL_FATAL)
	{
		throw std::runtime_error("cannot continue due to unexpected protocol buffer error: " + message);
	}

	return;
}


Darknet::ONNXExport::ONNXExport(const std::filesystem::path & cfg_filename, const std::filesystem::path & weights_filename, const std::filesystem::path & onnx_filename) :
	cfg_fn(cfg_filename),
	weights_fn(weights_filename),
	onnx_fn(onnx_filename),
	opset_version(18),	// - Upsample is until v9 (Mar 2019)
						// - Resize begins at v10 (May 2019)
						// - Mish begins at v18 (Dec 2022)
						//
						// Technically, we could use v10 for YOLOv4-tiny and YOLOv4-tiny-3L,
						// and only fall back to v18 when YOLOv4 (full) is used, but at this
						// point it is easier to settle on a single ONNX version to support.
						//
						// Also see initialize_model() where the IR version is set.
	graph(nullptr),
	fuse_batchnorm(true)
{
	TAT(TATPARMS);
	*cfg_and_state.output											<< std::endl
		<< "Darknet/YOLO ONNX Export Tool"							<< std::endl
		<< "-> configuration ........ " << cfg_fn		.string()	<< std::endl
		<< "-> weights .............. " << weights_fn	.string()	<< std::endl
		<< "-> onnx output .......... " << onnx_fn		.string()	<< std::endl
		;

	google::protobuf::SetLogHandler(&Darknet::ONNXExport::log_handler);

	GOOGLE_PROTOBUF_VERIFY_VERSION;

	if (not std::filesystem::exists(onnx_fn))
	{
		// see if we can create the .onnx file
		std::ofstream ofs(onnx_fn, std::ios::binary);
		ofs << std::endl;
	}

	// delete the .onnx file to ensure we have write access
	std::error_code ec;
	const bool success = std::filesystem::remove(onnx_fn, ec);
	if (not success)
	{
		throw std::runtime_error("failed to delete file " + onnx_fn.string() + ": " + ec.message());
	}

	return;
}


Darknet::ONNXExport::~ONNXExport()
{
	TAT(TATPARMS);

	google::protobuf::ShutdownProtobufLibrary();

	free_network(cfg.net);

	return;
}


Darknet::ONNXExport & Darknet::ONNXExport::load_network()
{
	TAT(TATPARMS);

	// force the verbose logging to get the nice colour output when the network is loaded
	const bool original_verbose_flag = cfg_and_state.is_verbose;
	Darknet::set_verbose(true);

	// the following block of code is taken from load_network_custom() and parse_network_cfg_custom()
	cfg.read(cfg_fn);
	cfg.create_network(1, 1);
	load_weights(&cfg.net, weights_fn.string().c_str());

	if (cfg_and_state.is_set("dontfuse"))
	{
		// do not fuse the mean/scale/var
		fuse_batchnorm = false;
	}
	else
	{
		// fuse the mean, scale, and variance into the weights
		fuse_batchnorm = true;
		fuse_conv_batchnorm(cfg.net);
		calculate_binary_weights(&cfg.net);
	}

	// restore the verbose flag
	if (not original_verbose_flag)
	{
		Darknet::set_verbose(original_verbose_flag);
	}

	// tiny and tiny-3L have less than 100 sections, so pad 2 characters,
	// while yolo-full have over 100 sections, so pad 3 characters
	padding_len = std::to_string(cfg.sections.size()).size();

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::display_summary()
{
	TAT(TATPARMS);

	*cfg_and_state.output
		<< "-> doc string ........... " << model.doc_string()					<< std::endl
		<< "-> type name ............ " << model.GetTypeName()					<< std::endl
		<< "-> domain ............... " << model.domain()						<< std::endl
//		<< "-> metadata props size .. " << model.metadata_props_size()			<< std::endl
//		<< "-> training info size ... " << model.training_info_size()			<< std::endl
//		<< "-> functions size ....... " << model.functions_size()				<< std::endl
//		<< "-> configuration size ... " << model.configuration_size()			<< std::endl
		<< "-> producer name ........ " << model.producer_name()				<< std::endl
		<< "-> producer version ..... " << model.producer_version() << " "		<< Darknet::in_colour(Darknet::EColour::kDarkGrey, "[built " __DATE__ "]") << std::endl
		<< "-> model version ........ " << model.model_version()				<< std::endl
		<< "-> batchnorm fused ...... " << (fuse_batchnorm		? "yes" : "no")	<< Darknet::in_colour(Darknet::EColour::kDarkGrey, " [toggle with -fuse or -dontfuse]") << std::endl
//		<< "-> has graph ............ " << (model.has_graph()	? "yes" : "no")	<< std::endl
		<< "-> graph input size ..... " << graph->input_size()	<< " "			<< Darknet::in_colour(Darknet::EColour::kDarkGrey, input_string)	<< std::endl
		<< "-> graph output size .... " << graph->output_size()	<< " "			<< Darknet::in_colour(Darknet::EColour::kDarkGrey, output_string)	<< std::endl
		<< "-> graph node size ...... " << graph->node_size()					<< std::endl
		<< "-> graph initializers ... " << graph->initializer_size()			<< std::endl
		<< "-> ir version ........... " << model.ir_version()	<< " "			<< Darknet::in_colour(Darknet::EColour::kDarkGrey, "[" + ir_date_lookup(model.ir_version()) + "]") << std::endl
		<< "-> opset version ........ ";

	for (const auto & opset : model.opset_import())
	{
		if (opset.domain().empty() == false)
		{
			*cfg_and_state.output << opset.domain() << ":";
		}
		*cfg_and_state.output << opset.version() << " " << Darknet::in_colour(Darknet::EColour::kDarkGrey, "[" + ops_date_lookup(opset.version()) + "]") << " ";
	}
	*cfg_and_state.output << std::endl;

	for (const auto & [key, val] : number_of_floats_exported)
	{
		*cfg_and_state.output << "-> exported " << key << " ";
		if (key.size() < 12)
		{
			*cfg_and_state.output << std::string(12 - key.size(), '.');
		}

		// add a comma every 3rd digit to make it easier to read
		std::string str = std::to_string(val);
		size_t pos = str.size();
		while (pos > 3)
		{
			pos -= 3;
			str.insert(pos, ",");
		}

		*cfg_and_state.output << " " << sizeof(float) << " x " << str << " " << Darknet::in_colour(Darknet::EColour::kDarkGrey, "[" + size_to_IEC_string(sizeof(float) * val) + "]") << std::endl;
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::initialize_model()
{
	TAT(TATPARMS);

	// IR = Intermediate Representation, related to versioning
	// https://github.com/onnx/onnx/blob/main/docs/IR.md
	// https://github.com/onnx/onnx/blob/main/docs/Versioning.md
	// 2019_9_19 aka "6" is the last version prior to introducing training
//	model.set_ir_version(onnx::Version::IR_VERSION_2019_9_19);	// == 6
	model.set_ir_version(onnx::Version::IR_VERSION_2023_5_5);	// == 9 (Mish was introduced in Dec 2022)
//	model.set_ir_version(onnx::Version::IR_VERSION_2025_05_12); // == 11

	// The name of the framework or tool used to generate this model.
	model.set_producer_name("Darknet/YOLO ONNX Export Tool");

	// The version of the framework or tool used to generate this model.
	model.set_producer_version(DARKNET_VERSION_SHORT);

	// We use reverse domain names as name space indicators.
	/// @todo We need a command-line parameter for this field.
	model.set_domain("ai.darknetcv");

	// The version of the graph encoded.
	/// @todo We need a command-line parameter for this field.
	model.set_model_version(1);

	// A human-readable documentation for this attribute. Markdown is allowed.
	std::time_t tt = std::time(nullptr);
	char buffer[50];
	std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %z", std::localtime(&tt));
	model.set_doc_string(
		"ONNX generated from Darknet/YOLO neural network " +
		cfg_fn.filename().string() +
		" (" + std::to_string(cfg.sections.size()) + " layers)"
		" and " +
		weights_fn.filename().string() +
		" (" + size_to_IEC_string(std::filesystem::file_size(weights_fn)) + ")"
		" on " +
		buffer + ".");

	auto opset = model.add_opset_import();
	opset->set_domain(""); // empty string means use the default ONNX domain
	opset->set_version(opset_version); // "Upsample" is only supported up to v9, and "Resize" requires a minimum of v10.  MISH requires v18.

	graph = new onnx::GraphProto();
	graph->set_name(weights_fn.stem().string());
	model.set_allocated_graph(graph);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_input_output_dimensions(onnx::ValueInfoProto * proto, const std::string & name, const int v1, const int v2, const int v3, const int v4, const size_t line_number)
{
	TAT(TATPARMS);

	/* For example:
	 *
		input {
			name: "000_net"
			type {
				tensor_type {
					elem_type: 1				# tensor of floats
					shape {						# NCHW=...
						dim { dim_value: 1 }	# batch size
						dim { dim_value: 3 }	# channels
						dim { dim_value: 160 }	# height
						dim { dim_value: 224 }	# width
				}
			}
		}
	}
	*/

	proto->set_name(name);
	proto->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(line_number));

	auto type = new onnx::TypeProto();
	proto->set_allocated_type(type);

	auto tensor_type = new onnx::TypeProto_Tensor();
	type->set_allocated_tensor_type(tensor_type);
	tensor_type->set_elem_type(onnx::TensorProto::FLOAT);

	auto shape = new onnx::TensorShapeProto();
	tensor_type->set_allocated_shape(shape);

	// n
	auto dim = shape->add_dim();
	dim->set_dim_value(v1);

	// other values are optional

	if (v2 >= 0)
	{
		// c
		dim = shape->add_dim();
		dim->set_dim_value(v2);

		if (v3 >= 0)
		{
			// h
			dim = shape->add_dim();
			dim->set_dim_value(v3);

			if (v4 >= 0)
			{
				// w
				dim = shape->add_dim();
				dim->set_dim_value(v4);
			}
		}
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_input_frame()
{
	TAT(TATPARMS);

	auto input = graph->add_input();

	const int b = 1;
	const int c = cfg.net.c;
	const int h = cfg.net.h;
	const int w = cfg.net.w;

	populate_input_output_dimensions(input, "frame", b, c, h, w, cfg.network_section.line_number);

	input->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(cfg.network_section.line_number) + " [w=" + std::to_string(w) + ", h=" + std::to_string(h) + ", c=" + std::to_string(c) + "]");

	input_string =
		"[B=" + std::to_string(b) +
		" C=" + std::to_string(c) +
		" H=" + std::to_string(h) +
		" W=" + std::to_string(w) +
		"]";

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_output()
{
	TAT(TATPARMS);

	// look for all the YOLO output layers

	for (int idx = 0; idx < cfg.net.n; idx ++)
	{
		const auto & l = cfg.net.layers[idx];
		const auto & section = cfg.sections[idx];
		if (section.type == Darknet::ELayerType::YOLO)
		{
			auto output = graph->add_output();
			populate_input_output_dimensions(output, format_name(idx, l), 1, l.c, l.h, l.w, section.line_number);

			output->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(idx) + "]");

			output_string += "[" + output->name() + "] ";
		}
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_nodes()
{
	TAT(TATPARMS);

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md

	// prime what we'll use as our first input name
	most_recent_output_per_index[-1] = "frame";

	for (size_t index = 0; index < cfg.sections.size(); index ++)
	{
		auto & section = cfg.sections[index];

		*cfg_and_state.output
			<< "\r"
			"-> processing layer ..... " << index+1 << "/" << cfg.sections.size() << " [" << section.name << "] on line #" << section.line_number << "          " << std::flush;
		if (cfg_and_state.is_verbose)
		{
			*cfg_and_state.output << std::endl;
		}

		switch(section.type)
		{
			case Darknet::ELayerType::CONVOLUTIONAL:
			{
				add_node_conv(index, section);
				break;
			}
			case Darknet::ELayerType::ROUTE:
			{
				// check to see if we're splitting the output, or concatenating two previous outputs
				const bool route_is_split = (section.find_int_array("layers").size() == 1);

				if (route_is_split)
				{
					add_node_route_split(index, section);
				}
				else
				{
					add_node_route_concat(index, section);
				}
				break;
			}
			case Darknet::ELayerType::MAXPOOL:
			{
				add_node_maxpool(index, section);
				break;
			}
			case Darknet::ELayerType::UPSAMPLE:
			{
				add_node_resize(index, section);
				break;
			}
			case Darknet::ELayerType::YOLO:
			{
				add_node_yolo(index, section);
				break;
			}
			case Darknet::ELayerType::SHORTCUT:
			{
				// shortcut is in YOLOv4 (full), not in tiny or tiny-3L
				add_node_shortcut(index, section);
				break;
			}
			default:
			{
				throw std::invalid_argument("layer type " + Darknet::to_string(section.type) + " in " + cfg_fn.string() + ":" + std::to_string(section.line_number) + " is not supported in this version of the Darknet/YOLO ONNX export tool.");
			}
		}
	}

	*cfg_and_state.output << std::endl;

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_conv(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const auto name = format_name(index, section.type);

	auto node = graph->add_node();
	node->set_name(name);
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << node->name() << std::endl;
	}
	node->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(index) + "]");
	node->set_op_type("Conv");
	node->add_input(most_recent_output_per_index[index - 1]);
	node->add_input(name + "_weights");
	node->add_output(name);
	most_recent_output_per_index[index] = name;

	const int dilation		= 1;
	const int stride		= section.find_int("stride"	, 2);
	const int kernel_size	= section.find_int("size"	, 3);
	const int pad			= [&]() -> int
	{
		int i = 0;
		if (section.find_int("pad", 0))
		{
			i = dilation * (kernel_size - 1) / 2;
		}
		return i;
	}();

	auto attrib = node->add_attribute();
	attrib->set_name("pads");
	attrib->add_ints(pad);
	attrib->add_ints(pad);
	attrib->add_ints(pad);
	attrib->add_ints(pad);
	attrib->set_type(onnx::AttributeProto::INTS);

	attrib = node->add_attribute();
	attrib->set_name("dilations");
	attrib->add_ints(dilation);
	attrib->add_ints(dilation);
	attrib->set_type(onnx::AttributeProto::INTS);

	attrib = node->add_attribute();
	attrib->set_name("group");
	attrib->set_i(1); // default for group is already "1"
	attrib->set_type(onnx::AttributeProto::INT);

	attrib = node->add_attribute();
	attrib->set_name("kernel_shape");
	attrib->add_ints(kernel_size);
	attrib->add_ints(kernel_size);
	attrib->set_type(onnx::AttributeProto::INTS);

	attrib = node->add_attribute();
	attrib->set_name("strides");
	attrib->add_ints(stride);
	attrib->add_ints(stride);
	attrib->set_type(onnx::AttributeProto::INTS);

	const auto & l = cfg.net.layers[index];

	// loosely based on load_convolutional_weights()
	populate_graph_initializer(l.weights, l.nweights, index, l, "weights");

	if (section.find_int("batch_normalize", 0) and not fuse_batchnorm)
	{
		// insert a batch normalization node
		add_node_bn(index, section);
	}
	else
	{
		// bias must be added in only 1 place -- either in BN, or here in the Conv if BN isn't used
		node->add_input(name + "_bias");
		populate_graph_initializer(l.biases, l.n, index, l, "bias");
	}

	check_activation(index, section);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_shortcut(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const auto name = format_name(index, section.type);

	auto node = graph->add_node();
	node->set_name(name);
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << node->name() << std::endl;
	}
	node->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(index) + "]");
	node->set_op_type("Add");
	node->add_input(most_recent_output_per_index[index - 1]);

	int layer_to_add = section.find_int("from");
	// if the index is positive, then we have an absolute value; otherwise it is relative to the current index
	if (layer_to_add < 0)
	{
		layer_to_add += index;
	}
	node->add_input(most_recent_output_per_index[layer_to_add]);
	node->add_output(name);
	most_recent_output_per_index[index] = name;

#if 0 /// @todo V5: unused?  Do we have weights for shortcuts?
	const auto & l = cfg.net.layers[index];
	populate_graph_initializer(l.weights, l.nweights, index, l, "weights");
#endif

	check_activation(index, section);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::check_activation(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	// when outputing certain nodes such as "Conv" or "Add", if it uses activation=... then we may need to also output an activation node
	const auto & activation = section.find_str("activation", "linear");
	switch(cfg.net.layers[index].activation)
	{
		case LEAKY:
		case RELU:
		case MISH:
		{
			add_node_activation(index, section);
			break;
		}
		case LINEAR:
		{
			// nothing to do
			break;
		}
		default:
		{
			throw std::invalid_argument("activation type " + std::to_string(cfg.net.layers[index].activation) + " in " + cfg_fn.string() + ":" + std::to_string(section.line_number) + " is not supported in this version of the ONNX export tool");
		}
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_activation(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const auto name = format_name(index, section.type) + "_activation";

	auto node = graph->add_node();
	node->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(index) + "]");
	node->set_name(name);
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << node->name() << std::endl;
	}
	node->add_input(most_recent_output_per_index[index]);
	node->add_output(name);
	most_recent_output_per_index[index] = name;

	const auto & activation = cfg.net.layers[index].activation;
	if (activation == LEAKY)
	{
		node->set_op_type("LeakyRelu");

		// with Darknet, alpha is hard-coded to be 0.1f
		auto attrib = node->add_attribute();
		attrib->set_name("alpha");
		attrib->set_f(0.1f);
		attrib->set_type(onnx::AttributeProto::FLOAT);
	}
	else if (activation == RELU)
	{
		node->set_op_type("Relu");
	}
	else if (activation == MISH)
	{
		// YOLOv4 (full), not in tiny or tiny-3L

		// note that "Mish" as an operator was only introduced in opset 18+
		if (opset_version < 18)
		{
			auto opset = model.mutable_opset_import(0);
			opset->set_version(18);
			opset_version = 18;

			*cfg_and_state.output << "-> WARNING .............. " << Darknet::in_colour(Darknet::EColour::kYellow, "The use of \"mish\" in \"" + name + "\" requires the opset version to be increased to " + std::to_string(opset->version()) + ".") << std::endl;
		}

		node->set_op_type("Mish");

		// if we need to support this in an earlier version of the ONNX opset, then we'd need to split it out into 4 nodes:
		//
		//		1) exp(x)
		//		2) log(1+...)
		//		3) tanh(...)
		//		4) x * ...
	}
	else
	{
		throw std::invalid_argument("activation type " + std::to_string(activation) + " in " + cfg_fn.string() + ":" + std::to_string(section.line_number) + " is unexpected");
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_route_split(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const auto name = format_name(index, section.type);

	auto node = graph->add_node();
	node->set_op_type("Split");
	node->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(index) + "]");
	node->set_name(name);
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << node->name() << std::endl;
	}

	int layer_to_split = section.find_int("layers");
	// if the index is positive, then we have an absolute value; otherwise it is relative to the current index
	if (layer_to_split < 0)
	{
		layer_to_split += index;
	}

	node->add_input(most_recent_output_per_index[layer_to_split]);
	node->add_output(name + "_A");
	node->add_output(name + "_B");
	most_recent_output_per_index[index] = name + "_A";

	auto attrib = node->add_attribute();
	attrib->set_name("axis"); // "Which axis to split on" https://github.com/onnx/onnx/blob/main/docs/Changelog.md#attributes-88
	attrib->set_i(1); // 1 == concat on channel axis
	attrib->set_type(onnx::AttributeProto::INT);

	if (opset_version < 13) // this was removed at v13
	{
		const auto & l = cfg.net.layers[layer_to_split];
		// split:  "length of each output" https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Split-2
		//
		// the size we need is half of the input layer
		const int split = l.n / 2; ///< @todo V5: is this logic correct?  This is only a guess as to how this works.

		attrib = node->add_attribute();
		attrib->set_name("split");
		attrib->add_ints(split);
		attrib->add_ints(split);
		attrib->set_type(onnx::AttributeProto::INTS);
	}

	if (opset_version >= 18) // this was introduced at v18
	{
		attrib = node->add_attribute();
		attrib->set_name("num_outputs");
		attrib->set_i(2);
		attrib->set_type(onnx::AttributeProto::INT);
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_route_concat(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const auto name = format_name(index, section.type);

	auto node = graph->add_node();
	node->set_op_type("Concat");
	node->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(index) + "]");
	node->set_name(name);
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << node->name() << std::endl;
	}
	node->add_output(name);
	most_recent_output_per_index[index] = name;

	// get the layers we're combining to use as our input
	for (int idx : section.find_int_array("layers"))
	{
		// if the index is positive, then we have an absolute value; otherwise it is relative to the current index
		if (idx < 0)
		{
			idx += index;
		}

		node->add_input(most_recent_output_per_index[idx]);
	}

	auto attrib = node->add_attribute();
	attrib->set_name("axis"); // "Which axis to split on" https://github.com/onnx/onnx/blob/main/docs/Changelog.md#attributes-88
	attrib->set_i(1); // 1 == concat on channel axis
	attrib->set_type(onnx::AttributeProto::INT);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_maxpool(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const auto name = format_name(index, section.type);

	auto node = graph->add_node();
	node->set_op_type("MaxPool");
	node->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(index) + "]");
	node->set_name(name);
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << node->name() << std::endl;
	}
	node->add_input(most_recent_output_per_index[index - 1]);
	node->add_output(name);
	most_recent_output_per_index[index] = name;

	const int dilation		= 1;
	const int stride		= section.find_int("stride"	, 2);
	const int kernel_size	= section.find_int("size"	, 3);
	const int pad			= [&]() -> int
	{
		int i = 0;
		if (section.find_int("pad", 0))
		{
			i = dilation * (kernel_size - 1) / 2;
		}
		return i;
	}();

	auto attrib = node->add_attribute();
	attrib->set_name("pads");
	attrib->add_ints(pad);
	attrib->add_ints(pad);
	attrib->add_ints(pad);
	attrib->add_ints(pad);
	attrib->set_type(onnx::AttributeProto::INTS);

	attrib = node->add_attribute();
	attrib->set_name("kernel_shape");
	attrib->add_ints(kernel_size);
	attrib->add_ints(kernel_size);
	attrib->set_type(onnx::AttributeProto::INTS);

	attrib = node->add_attribute();
	attrib->set_name("dilations");
	attrib->add_ints(dilation);
	attrib->add_ints(dilation);
	attrib->set_type(onnx::AttributeProto::INTS);

	attrib = node->add_attribute();
	attrib->set_name("strides");
	attrib->add_ints(stride);
	attrib->add_ints(stride);
	attrib->set_type(onnx::AttributeProto::INTS);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_yolo(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const auto name = format_name(index, section.type);

	auto node = graph->add_node();
	node->set_op_type("Identity"); //"YOLO");
//	node->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(index) + "]");
	node->set_name(name);
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << node->name() << std::endl;
	}
	node->add_input(most_recent_output_per_index[index - 1]);
	node->add_output(name);
	most_recent_output_per_index[index] = name;

#if 1
	node->set_doc_string(
		cfg_fn.filename().string() +
		" line #" + std::to_string(section.line_number) +
		" [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(index) + "]"
		" anchors="	+ section.find_str("anchors") +
		" mask="	+ section.find_str("mask"	) +
		" classes="	+ section.find_str("classes"));
#else

	// These are non-standard attributes.
	// They cannot be added as it breaks any validation checks that users attempt to run on the .onnx output file.
	// This is also why this node was changed to be "Identity" instead of using a custom "YOLO" node type.

	auto attrib = node->add_attribute();
	attrib->set_name("anchors");
	attrib->set_type(onnx::AttributeProto::FLOATS);
	for (const float & anchor : section.find_float_array("anchors"))
	{
		attrib->add_floats(anchor);
	}

	attrib = node->add_attribute();
	attrib->set_name("mask");
	attrib->set_type(onnx::AttributeProto::INTS);
	for (const int & mask : section.find_int_array("mask"))
	{
		attrib->add_ints(mask);
	}

	attrib = node->add_attribute();
	attrib->set_name("classes");
	attrib->set_type(onnx::AttributeProto::INT);
	attrib->set_i(section.find_int("classes"));
#endif

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_resize(const size_t index, Darknet::CfgSection & section) // aka "Upsample"
{
	TAT(TATPARMS);

	const auto name = format_name(index, section.type);

	if (opset_version < 10)
	{
		throw std::runtime_error("op type \"Resize\" required for node " + name + " needs opset >= 10, but opset is currently set to " + std::to_string(opset_version) + ".");
	}

	auto node = graph->add_node();
	node->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(index) + "]");
	node->set_op_type("Resize"); // this requires opset v10 or newer (prior to v10, it was called "Upsample")
	node->set_name(name);
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << node->name() << std::endl;
	}
	node->add_input(most_recent_output_per_index[index - 1]);
	node->add_input(""); // ROI (unused)
	node->add_input(name + "_scales");
	node->add_output(name);
	most_recent_output_per_index[index] = name;

	auto attrib = node->add_attribute();
	attrib->set_name("mode");
	attrib->set_type(onnx::AttributeProto::STRING);
	attrib->set_s("nearest"); // "nearest" or "linear" or "cubic"

	attrib = node->add_attribute();
	attrib->set_name("nearest_mode");
	attrib->set_type(onnx::AttributeProto::STRING);
	attrib->set_s("round_prefer_floor"); // "round_prefer_floor" or "round_prefer_ceil", or "floor", or "ceil"

	attrib = node->add_attribute();
	attrib->set_name("antialias");
	attrib->set_type(onnx::AttributeProto::INT);
	attrib->set_i(0); // 0=disable antialias

	attrib = node->add_attribute();
	attrib->set_name("coordinate_transformation_mode");
	attrib->set_type(onnx::AttributeProto::STRING);
	attrib->set_s("asymmetric");

	const float stride = section.find_float("stride", 2.0f);
	float f[4] = {1.0f, 1.0f, stride, stride};

	const auto & l = cfg.net.layers[index];
	populate_graph_initializer(f, 4, index, l, "scales", true);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_bn(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const auto name = format_name(index, section.type) + "_bn";

	auto node = graph->add_node();
	node->set_op_type("BatchNormalization");
	node->set_name(name);
	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << node->name() << std::endl;
	}
	node->add_input(most_recent_output_per_index[index]);
	node->add_input(name + "_scale");
	node->add_input(name + "_bias");
	node->add_input(name + "_mean");
	node->add_input(name + "_variance");
	node->add_output(name);
	most_recent_output_per_index[index] = name;

	auto attrib = node->add_attribute();
	attrib->set_name("epsilon");
	attrib->set_type(onnx::AttributeProto::FLOAT);
	attrib->set_f(0.00001f);

	attrib = node->add_attribute();
	attrib->set_name("momentum");
	attrib->set_type(onnx::AttributeProto::FLOAT);
	attrib->set_f(0.99f);

	const auto & l = cfg.net.layers[index];

	populate_graph_initializer(l.biases				, l.n, index, l, "bn_bias"		); // note this one also exists in "Conv" when BN is disabled
	populate_graph_initializer(l.scales				, l.n, index, l, "bn_scale"		);
	populate_graph_initializer(l.rolling_mean		, l.n, index, l, "bn_mean"		);
	populate_graph_initializer(l.rolling_variance	, l.n, index, l, "bn_variance"	);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_initializer(const float * f, const size_t n, const size_t idx, const Darknet::Layer & l, const std::string & name, const bool simple)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_trace and (f == nullptr or n == 0))
	{
		*cfg_and_state.output << "   " << format_weights(idx, l, name) << ": f=" << (void*)f << " n=" << n << std::endl;
	}
	else if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "-> " << format_weights(idx, l, name) << ": exporting " << n << " " << name << std::endl;
	}

	onnx::TensorProto * initializer = graph->add_initializer();
	initializer->set_data_type(onnx::TensorProto::FLOAT);
	initializer->set_name(format_weights(idx, l, name));
	initializer->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(cfg.sections[idx].line_number) + " [" + Darknet::to_string(l.type) + ", layer #" + std::to_string(idx) + ", " + std::to_string(n) + " x " + name + "]");

	/** @todo V5 2025-08-13:  This is black magic!  I actually have no idea how the DIMS work.  I saw some example
	 * Darknet/YOLO weights converted to ONNX and attempted to figure out the patern.  While this seems to work for
	 * the few examples I have, I would be extremely happy if someone can point out to me exactly how this works so
	 * I can implement it correctly!
	 */

	if (f == nullptr or n == 0)
	{
		initializer->add_dims(0);
	}
	else
	{
		if (simple)
		{
			initializer->add_dims(n);
		}
		else
		{
			// "l.n" is always the first dimension
			initializer->add_dims(l.n);

			if (n > l.n and not simple)
			{
				// must be dealing with weights

				const int div = std::max(1, l.size); // prevent division-by-zero

				initializer->add_dims(n / l.n / div / div);
				initializer->add_dims(div);
				initializer->add_dims(div);
			}
		}

		#if 0
			// important assumption:  pointer must be to little-endian float32
			initializer->set_raw_data(f, n * sizeof(float));
		#else
			// no need to worry about ARM and big/little endian byte order with this method
			for (size_t i = 0; i < n; i ++)
			{
				initializer->add_float_data(f[i]);
			}
		#endif

		// get the last part of the name to use as a key; for example, "002_conv_weights" returns a key of "weights"
		std::string key = name;
		auto pos = key.rfind("_");
		if (pos != std::string::npos)
		{
			key.erase(0, pos + 1);
		}
		number_of_floats_exported[key] += n;
//		std::cout << "==> name=" << name << ", key[" << key << "]=" << number_of_floats_exported[key] << std::endl;
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::build_model()
{
	TAT(TATPARMS);

	populate_graph_input_frame();
	populate_graph_output();
	populate_graph_nodes();

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::save_output_file()
{
	TAT(TATPARMS);

	std::ofstream ofs(onnx_fn, std::ios::binary);
	const bool success = model.SerializeToOstream(&ofs);
	if (not success)
	{
		throw std::runtime_error("failed to save ONNX output file " + onnx_fn.string());
	}
	ofs.close();

	*cfg_and_state.output
		<< "-> onnx saved to ........ " << Darknet::in_colour(Darknet::EColour::kBrightCyan, onnx_fn.string()) << " "
		<< Darknet::in_colour(Darknet::EColour::kDarkGrey, "[" + size_to_IEC_string(std::filesystem::file_size(onnx_fn)) + "]")					<< std::endl
		<< "-> WARNING .............. " << Darknet::in_colour(Darknet::EColour::kYellow, "This Darknet/YOLO ONNX Export Tool is experimental.")	<< std::endl
		<< "-> done!"																															<< std::endl;

	return *this;
}
