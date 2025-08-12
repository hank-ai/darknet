#include "darknet_onnx.hpp"


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
		<< "-> opset import size .... " << model.opset_import_size()			<< std::endl
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
		<< "-> ir version ........... " << model.ir_version()					<< std::endl
		<< "-> domain ............... " << model.domain()						<< std::endl
		<< "-> model version ........ " << model.model_version()				<< std::endl
		;

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
	opset->set_version(9); // beware of v10 or higher since op "Upsample" was deprecated

	graph = new onnx::GraphProto();
	graph->set_name(weights_fn.stem().string());
	model.set_allocated_graph(graph);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::set_doc_string(onnx::ValueInfoProto * proto, const size_t line_number)
{
	TAT(TATPARMS);

	if (proto and line_number > 0)
	{
		proto->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(line_number));
	}

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
	set_doc_string(proto, line_number);

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
	// https://github.com/onnx/onnx/blob/main/docs/Operators.md
	const std::map<Darknet::ELayerType, std::string> op =
	{
		{Darknet::ELayerType::CONVOLUTIONAL	, "Conv"	},
		{Darknet::ELayerType::ROUTE			, "Split"	},	// or "Concat"?
		{Darknet::ELayerType::MAXPOOL		, "MaxPool"	},
		{Darknet::ELayerType::UPSAMPLE		, "Upsample"},	// beware, this was deprecated starting with onnx operator set version 10
		{Darknet::ELayerType::YOLO			, "unknown"	},
	};

	for (size_t index = 0; index < cfg.sections.size(); index ++)
	{
		auto & section = cfg.sections[index];

		auto node = graph->add_node();
		node->set_name(format_name(index, section.type));

		if (op.count(section.type) != 1)
		{
			throw std::invalid_argument("layer type " + Darknet::to_string(section.type) + " does not have a known operator");
		}

		node->set_op_type(op.at(section.type));

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
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_initializers()
{
	TAT(TATPARMS);

	// look for all the layers with weights and biases

	for (int idx = 0; idx < cfg.net.n; idx ++)
	{
		const auto & l = cfg.net.layers[idx];

		bool load = false;



		/* ***************************** */
		/* TODO TODO TODO TODO TODO TODO */
		/* ***************************** */

		// similar switch() statement to the one in load_weights_upto()
		switch(l.type)
		{
			case Darknet::ELayerType::CONVOLUTIONAL:
			{
				if (l.share_layer == NULL)
				{
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> layer #" << idx << " (" << Darknet::to_string(l.type) << "): exporting convolutional weights" << std::endl;
					}
					//load_convolutional_weights(l, fp);
					load = true;
				}
				break;
			}
			case Darknet::ELayerType::SHORTCUT:
			{
				if (l.nweights > 0)
				{
					if (cfg_and_state.is_trace)
					{
						*cfg_and_state.output << "=> layer #" << idx << " (" << Darknet::to_string(l.type) << "): exporting shortcut weights" << std::endl;
					}
					//bytes_read += load_shortcut_weights(l, fp);
					load = true;
				}
				break;
			}
			case Darknet::ELayerType::CONNECTED:
			{
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << idx << " (" << Darknet::to_string(l.type) << "): exporting connected weights" << std::endl;
				}
				//bytes_read += load_connected_weights(l, fp, transpose);
				load = true;
				break;
			}
			case Darknet::ELayerType::CRNN:
			{
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << idx << " (" << Darknet::to_string(l.type) << "): exporting convolutional weights" << std::endl;
				}
//				bytes_read += load_convolutional_weights(*(l.input_layer)	, fp);
//				bytes_read += load_convolutional_weights(*(l.self_layer)	, fp);
//				bytes_read += load_convolutional_weights(*(l.output_layer)	, fp);
				load = true;
				break;
			}
			case Darknet::ELayerType::RNN:
			{
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << idx << " (" << Darknet::to_string(l.type) << "): exporting connected weights" << std::endl;
				}
//				bytes_read += load_connected_weights(*(l.input_layer)	, fp, transpose);
//				bytes_read += load_connected_weights(*(l.self_layer)	, fp, transpose);
//				bytes_read += load_connected_weights(*(l.output_layer)	, fp, transpose);
				load = true;
				break;
			}
			case Darknet::ELayerType::LSTM:
			{
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << idx << " (" << Darknet::to_string(l.type) << "): exporting connected weights" << std::endl;
				}
//				bytes_read += load_connected_weights(*(l.wf), fp, transpose);
//				bytes_read += load_connected_weights(*(l.wi), fp, transpose);
//				bytes_read += load_connected_weights(*(l.wg), fp, transpose);
//				bytes_read += load_connected_weights(*(l.wo), fp, transpose);
//				bytes_read += load_connected_weights(*(l.uf), fp, transpose);
//				bytes_read += load_connected_weights(*(l.ui), fp, transpose);
//				bytes_read += load_connected_weights(*(l.ug), fp, transpose);
//				bytes_read += load_connected_weights(*(l.uo), fp, transpose);
				load = true;
				break;
			}
			default:
			{
				// this layer does not have weights to load
				if (cfg_and_state.is_trace)
				{
					*cfg_and_state.output << "=> layer #" << idx << " (" << Darknet::to_string(l.type) << "): no weights to export" << std::endl;
				}
				break;
			}
		}

		/* ***************************** */
		/* TODO TODO TODO TODO TODO TODO */
		/* ***************************** */






		if (load)
		{
			onnx::TensorProto * initializer = graph->add_initializer();
			initializer->add_dims(32);
			initializer->add_dims(1);
			initializer->set_data_type(onnx::TensorProto::FLOAT);
			initializer->add_float_data(0.783f);
			initializer->add_float_data(0.784f);
			initializer->add_float_data(0.785f);
			initializer->add_float_data(0.786f);
			initializer->set_name(format_name(idx, l));
			initializer->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(cfg.sections[idx].line_number) + " [" + Darknet::to_string(l.type) + "]");
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
		<< std::endl;

	return *this;
}
