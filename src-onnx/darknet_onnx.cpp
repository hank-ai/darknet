/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#include "darknet_onnx.hpp"


/** @file
 * Convert Darknet/YOLO .cfg and .weights files to .onnx files.
 *
 * @warning This code cannot be trusted.  It was written in the summer of 2025 by Stephane Charette without a full
 * understanding of either the Darknet/YOLO internals, nor any reasonable understanding of the ONNX internals.  I
 * obtained some sample .onnx files from several sources on the web, and attempted to reverse engineer how these .onnx
 * files may have (!?) been put together from Darknet/YOLO weights.  I appologize for the cases where things are not
 * yet working as expected, and if it does happen to work I regret to say it is likely a mix of luck and happy
 * coincidences.
 *
 * Over time, I'm hoping other people will show up to help shine light in the dark corners, or provide me with more
 * configurations and weights that are broken to help make this tool work better.  But as you'll no doubt see in the
 * code below, there are many places where I've left some "todo" comments in an attempt to document code that needs to
 * be fixed.
 *
 * Stephane Charette, 2025-08-18.
 */


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();


	static std::string format_name(const size_t idx, const std::string & name)
	{
		TAT(TATPARMS);
		std::stringstream ss;
		ss << std::setfill('0') << std::setw(3) << (idx) << "_" << name;
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
	graph(nullptr)
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
	fuse_conv_batchnorm(cfg.net);
	calculate_binary_weights(&cfg.net);

	// restore the verbose flag
	if (not original_verbose_flag)
	{
		Darknet::set_verbose(original_verbose_flag);
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::display_summary()
{
	TAT(TATPARMS);

	*cfg_and_state.output
		<< "-> type name ............ " << model.GetTypeName()					<< std::endl
		<< "-> domain ............... " << model.domain()						<< std::endl
//		<< "-> metadata props size .. " << model.metadata_props_size()			<< std::endl
//		<< "-> training info size ... " << model.training_info_size()			<< std::endl
//		<< "-> functions size ....... " << model.functions_size()				<< std::endl
//		<< "-> configuration size ... " << model.configuration_size()			<< std::endl
		<< "-> producer name ........ " << model.producer_name()				<< std::endl
		<< "-> producer version ..... " << model.producer_version()				<< std::endl
		<< "-> doc string ........... " << model.doc_string()					<< std::endl
		<< "-> has graph ............ " << (model.has_graph() ? "yes" : "no")	<< std::endl
		<< "-> graph input size ..... " << graph->input_size()					<< std::endl
		<< "-> graph output size .... " << graph->output_size()					<< std::endl
		<< "-> graph node size ...... " << graph->node_size()					<< std::endl
		<< "-> graph initializers ... " << graph->initializer_size()			<< std::endl
		<< "-> model version ........ " << model.model_version()				<< std::endl
		<< "-> ir version ........... " << model.ir_version()					<< std::endl
		<< "-> opset version ........ ";

	for (const auto & opset : model.opset_import())
	{
		if (opset.domain().empty() == false)
		{
			*cfg_and_state.output << opset.domain() << ":";
		}
		*cfg_and_state.output << opset.version() << " ";
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

		*cfg_and_state.output << " " << sizeof(float) << " x " << str << " (" << size_to_IEC_string(sizeof(float) * val) << ")" << std::endl;
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
	model.set_ir_version(onnx::Version::IR_VERSION_2019_9_19); // == 6
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
	model.set_doc_string("ONNX generated from Darknet/YOLO neural network files " + cfg_fn.filename().string() + " and " + weights_fn.filename().string() + " on " + buffer + ".");

	auto opset = model.add_opset_import();
	opset->set_domain(""); // empty string means use the default ONNX domain
	opset->set_version(9); // beware of v10 or higher since op "Upsample" was deprecated in v10

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
					elem_type: 1				# what does this mean?
					shape {
						dim { dim_value: 1 }	# number
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
	tensor_type->set_elem_type(1); /// @todo V5: what does "1" mean here?

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


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_input_000_net()
{
	TAT(TATPARMS);

	auto input = graph->add_input();

	populate_input_output_dimensions(input, format_name(0, "net"), 1, cfg.net.c, cfg.net.h, cfg.net.w, cfg.network_section.line_number);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_input()
{
	TAT(TATPARMS);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_output()
{
	TAT(TATPARMS);

	// look for all the YOLO output layers

	for (int idx = 0; idx < cfg.net.n; idx ++)
	{
		const auto & l = cfg.net.layers[idx];
		if (l.type == Darknet::ELayerType::YOLO)
		{
			auto output = graph->add_output();
			populate_input_output_dimensions(output, format_name(idx, l), 1, l.c, l.h, l.w, cfg.sections[idx].line_number);
		}
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_nodes()
{
	TAT(TATPARMS);

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md
	const std::map<Darknet::ELayerType, std::string> op =
	{
		{Darknet::ELayerType::CONVOLUTIONAL	, "Conv"	},
		{Darknet::ELayerType::ROUTE			, "Split"	},	// or "Concat", depending on the purpose of [route] -- see below
		{Darknet::ELayerType::MAXPOOL		, "MaxPool"	},
		{Darknet::ELayerType::UPSAMPLE		, "Upsample"},	// beware, this was deprecated starting with onnx operator set version 10
	};

	std::string most_recent_node_output = "000_net";

	for (size_t index = 0; index < cfg.sections.size(); index ++)
	{
		auto & section = cfg.sections[index];

		const auto current_node_name		= format_name(index, section.type);
		const auto doc_string				= cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + "]";

		const bool route_is_concat 			= section.type == Darknet::ELayerType::ROUTE and not	section.exists("groups");
		const bool route_is_split			= section.type == Darknet::ELayerType::ROUTE and		section.exists("groups");

		const bool previous_layer_is_yolo	= index > 0 and cfg.sections[index - 1].type == Darknet::ELayerType::YOLO;
		const bool next_layer_is_yolo		= index < (cfg.sections.size() - 1) and cfg.sections[index + 1].type == Darknet::ELayerType::YOLO;

		if (section.type == Darknet::ELayerType::YOLO)
		{
			// skip to the next layer
			continue;
		}

		if (section.type == Darknet::ELayerType::ROUTE and previous_layer_is_yolo)
		{
			/// @todo V5: why do we skip to the next layer in this case?  Is this correct?
			continue;
		}

		auto node = graph->add_node();
		node->set_name(current_node_name);
		node->set_doc_string(doc_string);

		if (op.count(section.type) != 1)
		{
			throw std::invalid_argument("layer type " + Darknet::to_string(section.type) + " does not have a known operator");
		}

		if (route_is_concat)
		{
			node->set_op_type("Concat");
			// get the layers we're combining to figure out the input names to use
			for (int input_layer_index : section.find_int_array("layers"))
			{
				// if the index is positive, then we have an absolute value; otherwise it is relative to the current index
				if (input_layer_index < 0)
				{
					input_layer_index += index;
				}

				auto input_name = format_name(input_layer_index, cfg.sections[input_layer_index].type);
				if (cfg.sections[input_layer_index].type == Darknet::ELayerType::CONVOLUTIONAL)
				{
					input_name += "_lrelu";
				}
				node->add_input(input_name);
			}
		}
		else
		{
			node->set_op_type(op.at(section.type));
			node->add_input(most_recent_node_output);

			if (section.type == Darknet::ELayerType::UPSAMPLE)
			{
				node->add_input(current_node_name + "_scale");
			}
		}

		if (route_is_split)
		{
			// create a 2nd output when we have a route (split)
			node->add_output(current_node_name + "_dummy0");
		}

		node->add_output(current_node_name);
		most_recent_node_output = current_node_name;

		if (section.type == Darknet::ELayerType::MAXPOOL)
		{
			/// @todo V5 more hard-coded attributes that I need to better understand
			auto attrib = node->add_attribute();
			attrib->set_name("auto_pad");
			attrib->set_s("SAME_UPPER"); // note UPPER while CONV uses LOWER?
			attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
//			attrib->set_doc_string("???");
		}

		if (section.type == Darknet::ELayerType::CONVOLUTIONAL)
		{
			/// @todo V5 more hard-coded attributes that I need to better understand
			auto attrib = node->add_attribute();
			attrib->set_name("auto_pad");
			attrib->set_s("SAME_LOWER"); // note LOWER while MAXPOOL uses UPPER?
			attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
//			attrib->set_doc_string("???");

			attrib = node->add_attribute();
			attrib->set_name("dilations"); // layer->dilation ?
			attrib->add_ints(1);
			attrib->add_ints(1);
			attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
//			attrib->set_doc_string("???");
		}
		if (section.type == Darknet::ELayerType::CONVOLUTIONAL or section.type == Darknet::ELayerType::MAXPOOL)
		{
			auto attrib = node->add_attribute();
			attrib->set_name("kernel_shape");
			attrib->add_ints(section.find_int("size", 3));
			attrib->add_ints(section.find_int("size", 3));
			attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
//			attrib->set_doc_string("???");

			attrib = node->add_attribute();
			attrib->set_name("strides");
			attrib->add_ints(section.find_int("stride", 2));
			attrib->add_ints(section.find_int("stride", 2));
			attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
//			attrib->set_doc_string("???");
		}

		if (section.type == Darknet::ELayerType::ROUTE)
		{
			/// @todo V5 more hard-coded attributes that I need to better understand
			auto attrib = node->add_attribute();
			attrib->set_name("axis"); // "Which axis to split on" https://github.com/onnx/onnx/blob/main/docs/Changelog.md#attributes-88
			attrib->set_i(1);
			attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
//			attrib->set_doc_string("???");

			if (route_is_split)
			{
				int layer_to_split = section.find_int("layers");
				// if the index is positive, then we have an absolute value; otherwise it is relative to the current index
				if (layer_to_split < 0)
				{
					layer_to_split += index;
				}
				const auto & l = cfg.net.layers[layer_to_split];

				// split:  "length of each output" https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Split-2
				//
				// the size we need is half of the input layer
				const int split = l.n / 2; ///< @todo V5: is this logic correct?  This is only a guess as to how this works.

				attrib = node->add_attribute();
				attrib->set_name("split");
				attrib->add_ints(split);
				attrib->add_ints(split);
				attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
//				attrib->set_doc_string("???");
			}
		}

		if (section.type == Darknet::ELayerType::UPSAMPLE)
		{
			auto attrib = node->add_attribute();
			attrib->set_name("mode");
			attrib->set_s("nearest"); // default is "nearest"
			attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
		}

		// does this layer have weights?
		const auto & l = cfg.net.layers[index];
		if (l.weights != nullptr and l.nweights > 0)
		{
			node->add_input(current_node_name + "_weights");
			if (next_layer_is_yolo)
			{
				node->add_input(current_node_name + "_bias");
			}
			else
			{

				// what about batch normalize (bn)?

				auto node2 = graph->add_node();
				node2->set_doc_string(doc_string);
				node2->set_op_type("BatchNormalization");
				node2->add_input(most_recent_node_output);
				const auto bn_name = current_node_name + "_bn";
				node2->set_name(bn_name);
				node2->add_output(bn_name);
				most_recent_node_output = bn_name;

				/// @todo V5 why is this if() statement wrong?
//				if (l.batch_normalize and not l.dontloadscales)
				{
					node2->add_input(current_node_name + "_scale"	);
					node2->add_input(current_node_name + "_bias"	);
					node2->add_input(current_node_name + "_mean"	);
					node2->add_input(current_node_name + "_variance");

					/// @todo V5: hard-coded attributes...need to understand these and how to generate them correctly
					auto attrib = node2->add_attribute();
					attrib->set_name("epsilon");
					attrib->set_f(0.00001f); // 1e-05
					attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
//					attrib->set_doc_string("???");

					attrib = node2->add_attribute();
					attrib->set_name("momentum");
					attrib->set_f(0.99f);
					attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
//					attrib->set_doc_string("???");
				}
			}
		}

		/// @todo V5: this next block should not be hard-coded
		if (section.type == Darknet::ELayerType::CONVOLUTIONAL and not next_layer_is_yolo) /// @todo check for type, such as lrelu
		{
			auto node2 = graph->add_node();
			node2->set_op_type("LeakyRelu");
			const auto name = format_name(index, section.type) + "_lrelu";
			node2->set_doc_string(doc_string);
			node2->set_name(name);
			node2->add_input(most_recent_node_output);
			node2->add_output(name);
			most_recent_node_output = name;

			auto attrib = node2->add_attribute();
			attrib->set_name("alpha");
			attrib->set_f(0.1f);
			attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
		}

#if 0 /// @todo need to revisit this to determine exactly what is needed
		for (const auto & [key, line] : section.lines)
		{
			auto attrib = node->add_attribute();
			attrib->set_name(line.key);
			attrib->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(line.line_number) + ": " + line.line);

			if (line.f)
			{
				// is this a float or an int?
				const int	i = line.f.value();
				const float	f = line.f.value();
				if (i == f and line.val.find(".") == std::string::npos)
				{
					// must be an int
					attrib->set_i(i);
					attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INT);
				}
				else
				{
					// must be a float
					attrib->set_f(f);
					attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOAT);
				}
			}
			else
			{
				// check to see if we have a float array or int array, such as:
				//
				//		layers=-6,-1
				// or:
				//		scales=0.5, 0.4, 0.3, 0.2, 0.1

				if (line.val.length() > 0 and line.val.find_first_not_of(" -.,0123456789") == std::string::npos)
				{
					// array of ints, or array of floats?
					if (line.val.find(".") == std::string::npos)
					{
						// no decimal point, must be an array of INT
						for (const auto & i : section.find_int_array(key))
						{
							attrib->add_ints(i);
						}
						attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_INTS);
					}
					else
					{
						// we have at least 1 decimal point, must be an array of FLOAT
						for (const auto & f : section.find_float_array(key))
						{
							attrib->add_floats(f);
						}
						attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_FLOATS);
					}
				}
				else
				{
					// must be a plain old string
					attrib->set_s(line.val);
					attrib->set_type(onnx::AttributeProto_AttributeType::AttributeProto_AttributeType_STRING);
				}
			}
		}
#endif
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_initializer(const float * f, const size_t n, const size_t idx, const Darknet::Layer & l, const std::string & name)
{
	TAT(TATPARMS);

	if (f == nullptr or n == 0)
	{
		if (cfg_and_state.is_trace)
		{
			*cfg_and_state.output << "   " << format_weights(idx, l, name) << ": f=" << (void*)f << " n=" << n << std::endl;
		}
		return *this;
	}

	if (cfg_and_state.is_trace)
	{
		*cfg_and_state.output << "=> " << format_weights(idx, l, name) << ": exporting " << n << " " << name << std::endl;
	}

	onnx::TensorProto * initializer = graph->add_initializer();

	/** @todo V5 2025-08-13:  This is black magic!  I actually have no idea how the DIMS work.  I saw some example
	 * Darknet/YOLO weights converted to ONNX and attempted to figure out the patern.  While this seems to work for
	 * the few examples I have, I would be extremely happy if someone can point out to me exactly how this works so
	 * I can implement it correctly!
	 */

	// "l.n" is always the first dimension
	initializer->add_dims(l.n);
	if (n > l.n)
	{
		// must be dealing with weights

		const int div = std::max(1, l.size); // prevent division-by-zero

		initializer->add_dims(n / l.n / div / div);
		initializer->add_dims(div);
		initializer->add_dims(div);
	}

	initializer->set_data_type(onnx::TensorProto::FLOAT);
	initializer->set_name(format_weights(idx, l, name));
	initializer->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(cfg.sections[idx].line_number) + " [" + Darknet::to_string(l.type) + ", " + std::to_string(n) + " " + name + "]");

	for (size_t i = 0; i < n; i ++)
	{
		initializer->add_float_data(f[i]);
	}

	// get the last part of the name to use as a key; for example, "002_conv_weights" returns a key of "weights"
	std::string key = name;
	auto pos = key.rfind("_");
	if (pos != std::string::npos)
	{
		key.erase(0, pos + 1);
	}
	number_of_floats_exported[key] += n;

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_initializers()
{
	TAT(TATPARMS);

	// define several helper lambdas to export floats

	const auto export_convolutional = [&](Darknet::Layer & l, const int idx, const std::string & name) -> void
	{
		TAT(TATPARMS);

		// loosely based on load_convolutional_weights()
		const bool flag = l.batch_normalize and not l.dontloadscales;
//		Darknet::display_warning_msg("Layer #" + std::to_string(idx) + ": export of \"" + Darknet::to_string(l.type) + "\"" " from line #" + std::to_string(cfg.sections[idx].line_number) + " is untested.\n");
		if (true) populate_graph_initializer(l.biases			, l.n			, idx, l, name + "bias"		);
		if (true) populate_graph_initializer(l.weights			, l.nweights	, idx, l, name + "weights"	);
		if (flag) populate_graph_initializer(l.scales			, l.n			, idx, l, name + "scale"	);
		if (flag) populate_graph_initializer(l.rolling_mean		, l.n			, idx, l, name + "mean"		);
		if (flag) populate_graph_initializer(l.rolling_variance	, l.n			, idx, l, name + "variance"	);
	};

	const auto export_connected = [&](Darknet::Layer & l, const int idx, const std::string & name) -> void
	{
		TAT(TATPARMS);

		// loosely based on load_connected_weights()
		const bool flag = l.batch_normalize and not l.dontloadscales;
		Darknet::display_warning_msg("Layer #" + std::to_string(idx) + ": export of \"" + Darknet::to_string(l.type) + "\"" " from line #" + std::to_string(cfg.sections[idx].line_number) + " is untested.\n");
		if (true) populate_graph_initializer(l.biases			, l.outputs				, idx, l, name + "bias"		);
		if (true) populate_graph_initializer(l.weights			, l.outputs * l.inputs	, idx, l, name + "weights"	);
		if (flag) populate_graph_initializer(l.scales			, l.outputs				, idx, l, name + "scale"	);
		if (flag) populate_graph_initializer(l.rolling_mean		, l.outputs				, idx, l, name + "mean"		);
		if (flag) populate_graph_initializer(l.rolling_variance	, l.outputs				, idx, l, name + "variance"	);
	};

	// look through all the layers and export the values from the ones that have weights, biases, etc.
	for (int idx = 0; idx < cfg.net.n; idx ++)
	{
		auto & l = cfg.net.layers[idx];

		// similar switch() statement to the one in load_weights_upto(), see weights.cpp for details
		switch(l.type)
		{
			case Darknet::ELayerType::CONVOLUTIONAL:
			{
				if (l.share_layer == nullptr)
				{
					export_convolutional(l, idx, "");
				}
				break;
			}
			case Darknet::ELayerType::SHORTCUT:
			{
				if (l.nweights > 0)
				{
					Darknet::display_warning_msg("Layer #" + std::to_string(idx) + ": export of \"" + Darknet::to_string(l.type) + "\"" " from line #" + std::to_string(cfg.sections[idx].line_number) + " is untested.\n");
					populate_graph_initializer(l.weights, l.nweights, idx, l, "weights");
				}
				break;
			}
			case Darknet::ELayerType::CONNECTED:
			{
				export_connected(l, idx, "");
				break;
			}
			case Darknet::ELayerType::CRNN:
			{
				export_convolutional(*l.input_layer	, idx, "input_"	);
				export_convolutional(*l.self_layer	, idx, "self_"	);
				export_convolutional(*l.output_layer, idx, "output_");
				break;
			}
			case Darknet::ELayerType::RNN:
			{
				export_connected(*l.input_layer	, idx, "input_"	);
				export_connected(*l.self_layer	, idx, "self_"	);
				export_connected(*l.output_layer, idx, "output_");
				break;
			}
			case Darknet::ELayerType::LSTM:
			{
				export_connected(*l.wf, idx, "wf_");
				export_connected(*l.wi, idx, "wi_");
				export_connected(*l.wg, idx, "wg_");
				export_connected(*l.wo, idx, "wo_");
				export_connected(*l.uf, idx, "uf_");
				export_connected(*l.ui, idx, "ui_");
				export_connected(*l.ug, idx, "ug_");
				export_connected(*l.uo, idx, "uo_");
				break;
			}
			default:
			{
				// this layer does not have weights to load
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "   " << format_weights(idx, l, "none") << ": no weights" << std::endl;
				}
				break;
			}
		}
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::build_model()
{
	TAT(TATPARMS);

	populate_graph_input_000_net();
	populate_graph_input();
	populate_graph_output();
	populate_graph_nodes();
	populate_graph_initializers();

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
		<< "-> onnx saved to ........ " << onnx_fn.string()
		<< " (" << size_to_IEC_string(std::filesystem::file_size(onnx_fn)) << ")"
		<< std::endl
		<< "-> WARNING .............. " << Darknet::in_colour(Darknet::EColour::kYellow, "This Darknet/YOLO ONNX Export Tool is experimental.") << std::endl
		<< "-> done!" << std::endl;

	return *this;
}
