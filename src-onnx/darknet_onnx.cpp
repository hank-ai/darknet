/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2025 Stephane Charette
 */

#include "darknet_onnx.hpp"
#include "darknet_node.hpp"


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
 *
 * Update:
 *
 * Post-processing for YOLO layers was added for Darknet v5.1.  ONNX is still a beast, but I'm feeling much more
 * confident about the export tool than I was a few months ago.  I would no longer say "this code should not be
 * trusted" but instead would caution readers and debuggers to not assume the code is doing the right thing in all
 * cases since many assumptions have been made.  If you find a problem, please let me know or submit a PR.
 *
 * Stephane Charette, 2025-11-24.
 */


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();


	static std::string ir_date_lookup(const int ir)
	{
		TAT(TATPARMS);

		switch (ir)
		{
			// check these dates, they came from ChatGPT
			case 1:		return "Oct 2017";
			case 2:		return "Oct 2017";
			case 3:		return "Nov 2017";
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
			case 5:		return "Dec 2017";
			case 6:		return "Mar 2018";
			case 7:		return "May 2018";
			case 8:		return "Sep 2018";
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


Darknet::ONNXExport::ONNXExport(const std::filesystem::path & cfg_filename, const std::filesystem::path & weights_filename, const std::filesystem::path & onnx_filename) :
	cfg_fn(cfg_filename),
	weights_fn(weights_filename),
	onnx_fn(onnx_filename),
	opset_version(1), // see initialize_model()
	graph(nullptr),
	bit_size(32),
	fuse_batchnorm(true),
	postprocess_boxes(true)
{
	TAT(TATPARMS);
	*cfg_and_state.output											<< std::endl
		<< "Darknet/YOLO ONNX Export Tool"							<< std::endl
		<< "-> configuration ........ " << cfg_fn		.string()	<< std::endl
		<< "-> weights .............. " << weights_fn	.string()	<< std::endl
		<< "-> onnx output .......... " << onnx_fn		.string()	<< std::endl
		;

	GOOGLE_PROTOBUF_VERIFY_VERSION;

	if (not std::filesystem::exists(onnx_fn))
	{
		// see if we can create the .onnx file
		std::ofstream ofs(onnx_fn, std::ios::binary);
		ofs << std::endl;
	}

	// delete the output .onnx file to ensure we have write access
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

	if (cfg_and_state.is_set("noboxes"))
	{
		postprocess_boxes = false;
	}

	if (cfg_and_state.is_set("fp32"))
	{
		bit_size = 32;
	}
	if (cfg_and_state.is_set("fp16"))
	{
		bit_size = 16;
	}

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
		<< "-> post-process boxes ... " << (postprocess_boxes	? "yes" : "no")	<< Darknet::in_colour(Darknet::EColour::kDarkGrey, " [toggle with -boxes or -noboxes]") << std::endl
		<< "-> has graph ............ " << (model.has_graph()	? "yes" : "no")	<< std::endl
		<< "-> graph name ........... " << graph->name()						<< std::endl
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

	const auto colour = (bit_size < 32 ? Darknet::EColour::kBrightWhite : Darknet::EColour::kNormal);
	*cfg_and_state.output << std::endl
		<< "-> exported bit size .... "
		<< Darknet::in_colour(colour, std::to_string(bit_size) + "-bit floats")
		<< Darknet::in_colour(Darknet::EColour::kDarkGrey, " [toggle with -fp16 or -fp32]")
		<< std::endl;

	const std::set<std::string> exports_that_use_16_bit_floats =
	{
		"bias",
		"mean",
		"scale", // but not "scales"!
		"variance",
		"weights"
	};

	for (const auto & [key, val] : number_of_floats_exported)
	{
		*cfg_and_state.output << "-> exported " << key << " ";
		if (key.size() < 12)
		{
			*cfg_and_state.output << std::string(12 - key.size(), '.');
		}

		size_t float_size = sizeof(float);
		if (bit_size == 16 and exports_that_use_16_bit_floats.count(key))
		{
			float_size /= 2;
		}

		// add a comma every 3rd digit to make it easier to read
		std::string str = std::to_string(val);
		size_t pos = str.size();
		while (pos > 3)
		{
			pos -= 3;
			str.insert(pos, ",");
		}

		*cfg_and_state.output << " " << float_size << " bytes x " << str << " " << Darknet::in_colour(Darknet::EColour::kDarkGrey, "[" + size_to_IEC_string(float_size * val) + "]") << std::endl;
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::initialize_model()
{
	TAT(TATPARMS);

	if (bit_size != 32 and
		bit_size != 16)
	{
		// eventually I'd like to support INT8 quantization as well, but for now all we have is FP16 and FP32
		throw std::runtime_error("FP16 and FP32 are supported, but bit size is currently set to " + std::to_string(bit_size) + " which is not supported");
	}

	/* Quickly look through the configuration to see which ONNX opset we should be using:
	 *
	 *		Default:
	 *			- opset 5, Dec 2017
	 *
	 *		YOLOv4-tiny and YOLOv4-tiny-3L:
	 *			- [upsample] needs "Resize" introduced in opset 10
	 *
	 *		YOLOv4-full:
	 *			- MISH activation needs opset 18 introduced in December 2022
	 */
	opset_version = 5;
	for (const auto & section : cfg.sections)
	{
		if (section.type == Darknet::ELayerType::UPSAMPLE)
		{
			// "Resize" needs at least opset 10
			if (opset_version < 10)
			{
				Darknet::display_warning_msg("Opset 10 needed due to [upsample] at line #" + std::to_string(section.line_number) + ".\n");
				opset_version = 10;
			}
		}

		for (const auto & [key, line] : section.lines)
		{
			if (key == "activation" and line.val == "mish")
			{
				// "Mish" needs at least opset 18
				if (opset_version < 18)
				{
					Darknet::display_warning_msg("Opset 18 needed due to \"mish\" at line #" + std::to_string(line.line_number) + ".\n");
					opset_version = 18;
				}
			}
		}
	}

	// IR = Intermediate Representation, related to versioning
	// https://github.com/onnx/onnx/blob/main/docs/IR.md
	// https://github.com/onnx/onnx/blob/main/docs/Versioning.md
	// 2019_9_19 aka "6" is the last version prior to introducing training
	model.set_ir_version(onnx::Version::IR_VERSION_2019_3_18);	// == 5
//	model.set_ir_version(onnx::Version::IR_VERSION_2019_9_19);	// == 6
//	model.set_ir_version(onnx::Version::IR_VERSION_2023_5_5);	// == 9 (Mish was introduced in Dec 2022)
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
	const size_t number_of_classes = cfg.net.details->class_names.size();
	std::time_t tt = std::time(nullptr);
	char buffer[50];
	std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S %z", std::localtime(&tt));
	model.set_doc_string(
		"ONNX generated from Darknet/YOLO neural network " +
		cfg_fn.filename().string() +
		" (" +
		std::to_string(number_of_classes) + " class" + (number_of_classes == 1 ? "" : "es") + ", " +
		std::to_string(cfg.sections.size()) + " layers" +
		") and " +
		weights_fn.filename().string() +
		" (" + size_to_IEC_string(std::filesystem::file_size(weights_fn)) + ")"
		" on " +
		buffer + ".");

	auto opset = model.add_opset_import();
	opset->set_domain(""); // empty string means use the default ONNX domain
	opset->set_version(opset_version);

	graph = new onnx::GraphProto();
	graph->set_name(cfg_fn.stem().string());
	model.set_allocated_graph(graph);

	Node::graph = graph;
	Node::cfg_filename = cfg_fn.filename().string();

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

	if (bit_size == 32)
	{
		tensor_type->set_elem_type(onnx::TensorProto::FLOAT);
	}
	else
	{
		tensor_type->set_elem_type(onnx::TensorProto::FLOAT16);
	}

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

	input->set_doc_string(
		cfg_fn.filename().string() +
		" line #"	+ std::to_string(cfg.network_section.line_number) +
		" [b=1,"
		" c="		+ std::to_string(c) + ","
		" h="		+ std::to_string(h) + ","
		" w="		+ std::to_string(w) + "]"
		);

	input_string =
		"[\"frame\""
		" B=" + std::to_string(b) +
		" C=" + std::to_string(c) +
		" H=" + std::to_string(h) +
		" W=" + std::to_string(w) +
		"]";

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_YOLO_output()
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
			populate_input_output_dimensions(output, Node::get_output_for_layer_index(idx), 1, l.c, l.h, l.w, section.line_number);
			output->set_doc_string(cfg_fn.filename().string() + " line #" + std::to_string(section.line_number) + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(idx) + "]");
			output_string += "[\"" + output->name() + "\"] ";
		}
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_nodes()
{
	TAT(TATPARMS);

	// https://github.com/onnx/onnx/blob/main/docs/Operators.md

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
				const auto number_of_layers = section.find_int_array("layers").size();

				if (number_of_layers == 1)
				{
					if (section.exists("groups") or section.exists("group_id"))
					{
						// we're doing a channel slice
						add_node_route_slice(index, section);
					}
					else
					{
						// reference a specific layer but effectively a no-op
						add_node_route_identity(index, section);
					}
				}
				else if (number_of_layers >= 2)
				{
					add_node_route_concat(index, section);
				}
				else
				{
					throw std::invalid_argument(cfg_fn.string() + ": " + Darknet::to_string(section.type) + " on line #" + std::to_string(section.line_number) + ": number of layers is not supported: " + std::to_string(number_of_layers));
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

	Node node(section);
	node.type("Conv")
		.add_input(-1)
		.add_input("_weights")
		.add_attribute_INT("group"			, 1							)
		.add_attribute_INTS("pads"			, {pad, pad, pad, pad}		)
		.add_attribute_INTS("dilations"		, {dilation, dilation}		)
		.add_attribute_INTS("kernel_shape"	, {kernel_size, kernel_size})
		.add_attribute_INTS("strides"		, {stride, stride}			);

	const auto & l = cfg.net.layers[index];

	// loosely based on load_convolutional_weights()
	populate_graph_initializer(l.weights, l.nweights, l, node.name + "_weights");

	if (section.find_int("batch_normalize", 0) and not fuse_batchnorm)
	{
		// insert a batch normalization node
		add_node_bn(index, section);
	}
	else
	{
		// bias must be added in only 1 place -- either in BN, or here in the Conv if BN isn't used
		node.add_input("_bias");
		populate_graph_initializer(l.biases, l.n, l, node.name + "_bias");
	}

	check_activation(index, section);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_shortcut(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	Node node(section);
	node.type("Add").add_input(index - 1).add_input(section.find_int("from"));

#if 0 /// @todo V5: unused?  Do we have weights for shortcuts?
	const auto & l = cfg.net.layers[index];
	populate_graph_initializer(l.weights, l.nweights, l, node.name + "_weights");
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
		case LOGISTIC:
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

	const std::string previous_output = Node::get_output_for_layer_index(index);

	Node node(section, "_activation");
	node.add_input(previous_output);

	const auto & activation = cfg.net.layers[index].activation;
	if (activation == LEAKY)
	{
		node.type("LeakyRelu").add_attribute_FLOAT("alpha", 0.1f); // with Darknet, alpha is hard-coded to be 0.1f
	}
	else if (activation == LOGISTIC)
	{
		node.type("Sigmoid");
	}
	else if (activation == RELU)
	{
		node.type("Relu");
	}
	else if (activation == MISH)
	{
		// YOLOv4 (full), not in tiny or tiny-3L

		// note that "Mish" as an operator was only introduced in opset 18+
		if (opset_version < 18)
		{
			throw std::runtime_error("op type \"Mish\" required for node " + section.name + " needs opset >= 18, but opset is currently set to " + std::to_string(opset_version) + ".");
		}

		node.type("Mish");

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


Darknet::ONNXExport & Darknet::ONNXExport::add_node_route_identity(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	/** @todo verify what ChatGPT has to say about this:
	 *
	 * Many Darknet-to-ONNX converters mistakenly convert single-layer route layers into Split or Slice nodes due to:
	 *
	 *	1. Overgeneralized conversion logic (trying to handle all cases with the same code)
	 *	2. Misinterpretation of what route does
	 *	3. Incorrect assumptions about data layout
	 *	4. Legacy from older or broken conversion scripts
	 *
	 * In reality, single-layer route is just a pass-through, but some tools try to “process” it unnecessarily.
	 *
	 * ...and:
	 *
	 * In ONNX, the Identity operator is functionally equivalent to a no-op (no operation) on the tensor data.
	 * It simply passes the input through to the output unchanged.
	 */

	// if we get here, then we should have no "groups" or "group_id"
	if (section.exists("groups") or section.exists("group_id"))
	{
		throw std::runtime_error("route identity layer at line #" + std::to_string(section.line_number) + " should not have 'groups' or 'group_id'");
	}

#if 0
	Node node(section, "_identity");
	node.type("Identity").add_input(section.find_int("layers"));
#else
	// We don't need to create a node for this.  Simply take the output we need, and store a reference to it in the map
	// as if a real node existed.  Then when the consumer goes to use it as input, they'll be grabbing the right output.

	int route_index = section.find_int("layers");
	if (route_index < 0)
	{
		route_index += section.index;
	}
	const std::string output = Node::get_output_for_layer_index(route_index);
	Node::output_per_layer_index[section.index] = output;
#endif

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_route_slice(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	Node node(section, "_slice");

	const int groups	= section.find_int("groups"		, 1);
	const int group_id	= section.find_int("group_id"	, 0);
	if (groups == 1 and cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << node.name << ": splitting into 1 group is the same as a NO-OP.  This node could be eliminated." << std::endl;
	}
	if (groups < 1)
	{
		throw std::invalid_argument(node.name + ": invalid number of groups (" + std::to_string(groups) + ")");
	}
	if (group_id < 0 or group_id >= groups)
	{
		throw std::invalid_argument(node.name + ": invalid group number (" + std::to_string(group_id) + ")");
	}

	node.doc_append(
		" groups="		+ std::to_string(groups) +
		" group_id="	+ std::to_string(group_id));

	// this is a slice, so we'll always have a *single* layer, not an array of ints like on concat nodes
	int layer_to_slice = section.find_int("layers");
	if (layer_to_slice < 0)
	{
		layer_to_slice += index;
	}

	const auto & l = cfg.net.layers[layer_to_slice];
	const int channels = l.out_c;

	// The inputs we're going to need for the Slice node are:
	//
	//	- starts (group_id * channels_per_group)
	//	- ends (start + channels_per_group)
	//	- axes (1)
	//	- steps (1)
	//
	const int channels_per_group	= channels / groups;
	const int starts				= group_id * channels_per_group;
	const int ends					= starts + channels_per_group;

	const std::map<std::string, int> constants =
	{
		{node.name + "_1_starts", starts},
		{node.name + "_2_ends"	, ends},
		{node.name + "_3_axes"	, 1},
		{node.name + "_4_steps"	, 1}
	};
	for (const auto & [key, val] : constants)
	{
		onnx::TensorProto * initializer = graph->add_initializer();
		initializer->set_name(key);
		initializer->set_doc_string(node.node->doc_string());
		initializer->set_data_type(onnx::TensorProto::INT32);
		initializer->add_dims(1);
		initializer->add_int32_data(val);
	}

	node.type("Slice")
		.add_input(layer_to_slice)
		.add_input("_1_starts"	)
		.add_input("_2_ends"	)
		.add_input("_3_axes"	)
		.add_input("_4_steps"	);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_route_concat(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	Node node(section, "_concat");
	node.type("Concat").add_attribute_INT("axis", 1); // axis 1 == concat on channel axis

	// get the layers we're combining to use as our input
	for (int idx : section.find_int_array("layers"))
	{
		node.add_input(idx);
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_maxpool(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const int dilation		= 1;
	const int stride		= section.find_int("stride"	, 2);
	const int kernel_size	= section.find_int("size"	, 3);
	const int pad			= dilation * (kernel_size - 1) / 2;

	Node node(section);
	node.type("MaxPool").add_input(index - 1)//.add_input("_weights")
		.add_attribute_INT("ceil_mode"		, 0							)
		.add_attribute_INTS("pads"			, {pad, pad, pad, pad}		)
		.add_attribute_INTS("dilations"		, {dilation, dilation}		)
		.add_attribute_INTS("kernel_shape"	, {kernel_size, kernel_size})
		.add_attribute_INTS("strides"		, {stride, stride}			);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_yolo(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	if (postprocess_boxes)
	{
		// don't bother to output the YOLO layers since we're going to output the full post-processing of boxes instead
		return *this;
	}

	const auto strip_spaces =
			[](const std::string & s) -> std::string
			{
				std::string out;
				for (const auto & c : s)
				{
					if (not std::isspace(c))
					{
						out += c;
					}
				}
				return out;
			};

	static size_t yolo_counter = 0;

	Node node(section);
	node.type("Identity")
		.add_input(-1)
		.set_output("yolo_" + std::to_string(yolo_counter ++))
		.doc_append(
			" classes="	+ strip_spaces(section.find_str("classes"	)) +
			" mask="	+ strip_spaces(section.find_str("mask"		)) +
			" anchors="	+ strip_spaces(section.find_str("anchors"	)));

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_resize(const size_t index, Darknet::CfgSection & section) // aka "Upsample"
{
	TAT(TATPARMS);

	if (opset_version < 10)
	{
		throw std::runtime_error("op type \"Resize\" required for node " + section.name + " needs opset >= 10, but opset is currently set to " + std::to_string(opset_version) + ".");
	}

	Node node(section);
	node.type("Resize");

	// at opset 10, Resize only takes 2 inputs; roi wasn't added until opset 11
	if (opset_version < 11)
	{
		node.add_input(-1).add_input("_scales").add_attribute_STR("mode", "nearest"); // introduced in opset 10
	}
	else
	{
		node.add_input(-1).add_input("_roi").add_input("_scales")
			.add_attribute_STR("mode"							, "nearest"				) // introduced in opset 10
			.add_attribute_STR("nearest_mode"					, "round_prefer_floor"	) // introduced in opset 11
			.add_attribute_STR("coordinate_transformation_mode"	, "asymmetric"			); // introduced in opset 11
	}

	if (opset_version >= 18)
	{
		node.add_attribute_INT("antialias", 0); // 0=disable antialias -- introduced in opset 18
	}

	const auto & l = cfg.net.layers[index];
	const float stride = section.find_float("stride", 2.0f);
	float f[4] = {1.0f, 1.0f, stride, stride};
	populate_graph_initializer(f, 4, l, node.name + "_scales", true);

	if (opset_version >= 11)
	{
		// even though the RoI isn't used, we still need to provide a dummy (empty) tensor
		populate_graph_initializer(nullptr, 0, l, node.name + "_roi", true);
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::add_node_bn(const size_t index, Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const std::string previous_output = Node::get_output_for_layer_index(index);

	Node node(section, "_bn");
	node.type("BatchNormalization")
		.add_input(previous_output)
		.add_input("_bn_scale"		)
		.add_input("_bn_bias"		)
		.add_input("_bn_mean"		)
		.add_input("_bn_variance"	)
		.add_attribute_FLOAT("epsilon", 0.00001f)
		.add_attribute_FLOAT("momentum", 0.99f);

	const auto & l = cfg.net.layers[index];

	populate_graph_initializer(l.biases				, l.n, l, node.name + "_bn_bias"		); // note this one also exists in "Conv" when BN is disabled
	populate_graph_initializer(l.scales				, l.n, l, node.name + "_bn_scale"		);
	populate_graph_initializer(l.rolling_mean		, l.n, l, node.name + "_bn_mean"		);
	populate_graph_initializer(l.rolling_variance	, l.n, l, node.name + "_bn_variance"	);

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_initializer(const float * f, const size_t n, const Darknet::Layer & l, const std::string & name, const bool simple)
{
	TAT(TATPARMS);

	if (cfg_and_state.is_trace and (f == nullptr or n == 0))
	{
		*cfg_and_state.output << "   " << name << ": f=" << (void*)f << " n=" << n << std::endl;
	}
	else if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "-> " << name << ": exporting " << n << " " << name << "                " << std::endl;
	}

	onnx::TensorProto * initializer = graph->add_initializer();
	initializer->set_name(name);
	initializer->set_doc_string("initializer for " + Darknet::to_string(l.type) + ", " + std::to_string(n) + " x " + name + "]");

	/** @todo V5 2025-08-13:  This is black magic!  I actually have no idea how the DIMS work.  I saw some example
	 * Darknet/YOLO weights converted to ONNX and attempted to figure out the patern.  While this seems to work for
	 * the few examples I have, I would be extremely happy if someone can point out to me exactly how this works so
	 * I can implement it correctly!
	 */

	bool must_convert = false;

	if (f == nullptr or n == 0)
	{
		initializer->add_dims(0);
	}
	else
	{
		if (simple)
		{
			// for example, "scales" from Resize nodes uses this
			initializer->add_dims(n);
		}
		else
		{
			// must be dealing with weights, bias, scale, etc.
			if (bit_size < 32)
			{
				must_convert = true;
			}

			// "l.n" is always the first dimension
			initializer->add_dims(l.n);

			if (n > l.n and not simple)
			{
				const int div = std::max(1, l.size); // prevent division-by-zero

				initializer->add_dims(n / l.n / div / div);
				initializer->add_dims(div);
				initializer->add_dims(div);
			}
		}

		if (must_convert)
		{
			initializer->set_data_type(onnx::TensorProto::FLOAT16);

			std::vector<std::uint16_t> v;
			v.reserve(n);
			for (size_t i = 0; i < n; i ++)
			{
				v.push_back(Darknet::convert_to_fp16(f[i]));
			}
			initializer->set_raw_data(v.data(), v.size() * sizeof(std::uint16_t));
		}
		else
		{
			initializer->set_data_type(onnx::TensorProto::FLOAT);

			// no need to worry about ARM and big/little endian byte order with this method
			for (size_t i = 0; i < n; i ++)
			{
				initializer->add_float_data(f[i]);
			}
		}

		// get the last part of the name to use as a key; for example, "N6_L2_conv_weights" returns a key of "weights"
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
	populate_graph_nodes();

	if (postprocess_boxes)
	{
		// this outputs the full set of nodes, including "confs" and "boxes"
		populate_graph_postprocess();
	}
	else
	{
		// this only outputs up to the YOLO nodes
		populate_graph_YOLO_output();
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::save_output_file()
{
	TAT(TATPARMS);

#if 0
	// onnx::ModelProto::SerializeToOstream() is not available with older versions of Protocol Buffers, such as when using Ubuntu 20.04.
	std::ofstream ofs(onnx_fn, std::ios::binary);
	const bool success = model.SerializeToOstream(&ofs);
	if (not success)
	{
		throw std::runtime_error("failed to save ONNX output file " + onnx_fn.string());
	}
	ofs.close();
#else
	const std::string str = model.SerializeAsString();
	if (str.empty())
	{
		throw std::runtime_error("failed to serialize ONNX output to " + onnx_fn.string());
	}
	std::ofstream(onnx_fn, std::ios::binary) << str;
#endif

	*cfg_and_state.output
		<< "-> onnx saved to ........ " << Darknet::in_colour(Darknet::EColour::kBrightCyan, onnx_fn.string()) << " "
		<< Darknet::in_colour(Darknet::EColour::kDarkGrey, "[" + size_to_IEC_string(std::filesystem::file_size(onnx_fn)) + "]")					<< std::endl
//		<< "-> WARNING .............. " << Darknet::in_colour(Darknet::EColour::kYellow, "This Darknet/YOLO ONNX Export Tool is experimental.")	<< std::endl
		<< "-> done!"																															<< std::endl;

	return *this;
}


/* ********************* */
/* POST-PROCESSING BOXES */
/* ********************* */


Darknet::VStr Darknet::ONNXExport::postprocess_yolo_slice_and_concat(Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	const int number_of_classes	= section.find_int("classes");
	const int number_of_masks	= section.find_int_array("mask").size();
	if (number_of_masks != 3)
	{
		throw std::runtime_error("line #" + std::to_string(section.line_number) + ": the ONNX export tool expected 3 masks per YOLO head, not " + std::to_string(number_of_masks));
	}

	/* We'll use the LEGO project as an example, which has 5 classes.  So each anchor feature vector looks like this:
	 *
	 *		[ 0=tx, 1=ty, 2=tw, 3=th, 4=to, 5=class0, 6=class1, 7=class2, 8=class3, 9=class4 ]
	 *
	 * We need to split things as follows:
	 *
	 *		1) tx and ty are grouped together
	 *		2) tw and th are grouped together
	 *		3) objectness is processed by itself
	 *		4) all class probabilities are grouped together
	 *
	 * So for each anchor we'll have a "Split" node.  That means with 3 anchors, we'll have 12 ONNX "Split" nodes.
	 * (3 anchors * 4 things = 12 nodes)
	 *
	 * Once we're done splitting everything, then we concat the nodes back into 4 groups.  The final node names (and output
	 * names) will look similar to this:
	 *
	 *		1) 37_yolo_concat_tx_ty
	 *		2) 37_yolo_concat_tw_th
	 *		3) 37_yolo_concat_obj
	 *		4) 37_yolo_concat_class
	 */
	struct Splits
	{
		std::string name;
		int start_index;
		int end_index;
	};
	const std::vector<Splits> v =
	{
		{"tx_ty", 0, 2					},
		{"tw_th", 2, 2					},
		{"obj"	, 4, 1					},
		{"class", 5, number_of_classes	}	// probability for each individual class
	};

	Darknet::VStr results;

	for (const auto & split : v)
	{
		Darknet::VStr outputs;

		for (int mask = 0; mask < number_of_masks; mask ++)
		{
			const int starts	= split.start_index + mask * (5 + number_of_classes); // 5 refers to the 5 fields always present:  tx, ty, tw, th, and objectness
			const int ends		= split.end_index + starts;
			const auto doc		= "post-processing: " + split.name + " [" + Darknet::to_string(section.type) + ", layer #" + std::to_string(section.index) + ", mask=" + std::to_string(mask) + ", classes=" + std::to_string(number_of_classes) + ", start=" + std::to_string(starts) + "]";

			Node node(section, "_slice_" + std::to_string(starts));
			node.type("Slice").doc_append(" mask=" + std::to_string(mask) + ", classes=" + std::to_string(number_of_classes) + ", start=" + std::to_string(starts));
			outputs.push_back(node.output);

			const std::map<std::string, int> initializers =
			{
				{node.name + "_1_starts", starts},
				{node.name + "_2_ends"	, ends	},
				{node.name + "_3_axes"	, 1		},
				{node.name + "_4_steps"	, 1		}
			};
			for (const auto & [key, val] : initializers)
			{
				onnx::TensorProto * initializer = graph->add_initializer();
				initializer->set_data_type(onnx::TensorProto::INT32);
				initializer->set_name(key);
				initializer->set_doc_string(doc);
				initializer->add_dims(1);
				initializer->add_int32_data(val);
			}

			node.add_input(-1)
				.add_input("_1_starts"	)
				.add_input("_2_ends"	)
				.add_input("_3_axes"	)
				.add_input("_4_steps"	);
		}

		// now we re-combine all similar outputs from each anchor

		Node node(section, "_concat_" + split.name);
		node.type("Concat").add_attribute_INT("axis", 1); // 1 == concat on channel axis
		for (const auto & out : outputs)
		{
			node.add_input(out);
		}

		results.push_back(node.output);
	}

	return results;
}


Darknet::ONNXExport & Darknet::ONNXExport::postprocess_yolo_tx_ty(Darknet::CfgSection & section, const Darknet::VStr & v, Darknet::VStr & output_names)
{
	TAT(TATPARMS);

	// This assumes that postprocess_yolo_slice_and_concat() has already run, and we have the "concat_tx_ty" node ready to use.

	for (const auto & name : v)
	{
		if (name.find("_concat_tx_ty") == std::string::npos)
		{
			continue;
		}

		// sigmoid brings the values into a range between zero and one

		Node sigmoid(section, "_sigmoid_tx_ty");
		sigmoid.type("Sigmoid").add_input(name);

		/* ...but instead of [0, 1], we actually want [-0.025, 1.025].  So we need to scale up by 1.05,
		* and then subtract half to ensure the center points are still located at the right place.
		*
		*			normalized_offset = sigmoid(tx) * 1.05 - 0.025
		*
		* ChatGPT says:
		*
		*			This is an affine transform that “stretches” and “recenters” the sigmoid output,
		*			giving a bit of extra range outside the cell.
		*/
		const float variance = 0.05f;
		Node const_1_050(section, 1.0f + variance, bit_size);
		Node const_0_025(section, variance / 2.0f, bit_size);

		// multiply by 1.05
		Node mul(section, "_mul_tx_ty");
		mul.type("Mul").add_input(sigmoid.output).add_input(const_1_050.output);

		// shift by -0.025
		Node sub(section, "_sub_tx_ty");
		sub.type("Sub").add_input(mul.output).add_input(const_0_025.output);

		output_names.push_back(sub.output);
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::postprocess_yolo_tw_th(Darknet::CfgSection & section, const Darknet::VStr & v, Darknet::VStr & output_names)
{
	TAT(TATPARMS);

	// This assumes that postprocess_yolo_slice_and_concat() has already run, and we have the "concat_tw_th" node ready to use.

	for (const auto & name : v)
	{
		if (name.find("_concat_tw_th") == std::string::npos)
		{
			continue;
		}

		Node node(section, "_exp_tw_th");
		node.type("Exp").add_input(name);

		output_names.push_back(node.output);
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::postprocess_yolo_to(Darknet::CfgSection & section, const Darknet::VStr & v, Darknet::VStr & output_names)
{
	TAT(TATPARMS);

	// This assumes that postprocess_yolo_slice_and_concat() has already run, and we have the "concat_obj" node ready to use.

	const auto & l					= cfg.net.layers[section.index];
	const int number_of_masks		= section.find_int_array("mask").size();
	const int size					= number_of_masks * l.w * l.h;
	VInt reshape_vector				= {1, size};

	for (const auto & name : v)
	{
		if (name.find("_concat_obj") == std::string::npos)
		{
			continue;
		}

		const auto reshape_const_1 = Node(section, reshape_vector).output;
		Node reshape1(section, "_reshape_obj");
		reshape1.type("Reshape").add_input(name).add_input(reshape_const_1);

		Node sigmoid(section, "_sigmoid_obj");
		sigmoid.type("Sigmoid").add_input(reshape1.output);

		reshape_vector.push_back(1);
		const auto reshape_const_2 = Node(section, reshape_vector).output;
		Node reshape2(section, "_reshape_obj");
		reshape2.type("Reshape").add_input(sigmoid.output).add_input(reshape_const_2);

		output_names.push_back(reshape2.output);
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::postprocess_yolo_class(Darknet::CfgSection & section, const Darknet::VStr & v, Darknet::VStr & output_names)
{
	TAT(TATPARMS);

	// This assumes that postprocess_yolo_slice_and_concat() has already run, and we have the "concat_class" node ready to use.

	const auto & l					= cfg.net.layers[section.index];
	const int number_of_classes		= section.find_int("classes");
	const int number_of_masks		= section.find_int_array("mask").size();
	VInt reshape_vector				= {1, number_of_masks, number_of_classes, l.w * l.h};

	for (const auto & name : v)
	{
		if (name.find("_concat_class") == std::string::npos)
		{
			continue;
		}

		const auto const_shape_1 = Node(section, reshape_vector).output;
		Node reshape1(section, "_reshape_class");
		reshape1.type("Reshape").add_input(name).add_input(const_shape_1);

		Node transpose(section, "_transpose_class");
		transpose.type("Transpose").add_input(reshape1.output).add_attribute_INTS("perm", {0, 1, 3, 2});

		reshape_vector = {1, number_of_masks * l.w * l.h, number_of_classes};
		const auto const_shape_2 = Node(section, reshape_vector).output;
		Node reshape2(section, "_reshape_class");
		reshape2.type("Reshape").add_input(transpose.output).add_input(const_shape_2);

		Node sigmoid(section, "_sigmoid_class");
		sigmoid.type("Sigmoid").add_input(reshape2.output);

		output_names.push_back(sigmoid.output);
	}

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::postprocess_yolo_confs(const Darknet::VStr & output_obj, const Darknet::VStr & output_class)
{
	TAT(TATPARMS);

	// every YOLO layer has "classes" and "objectness" which are combined together to produce the final confidence values

	Darknet::VStr v;

	int number_of_classes		= 0;
	int number_of_boxes			= 0;
	int number_of_yolo_layers	= 0;

	// start at 1 since we reference "idx - 1" below which would be a problem if idx was zero
	// (and YOLO cannot be first layer anyway!)
	for (int idx = 1; idx < cfg.net.n; idx ++)
	{
		auto & section = cfg.sections[idx];
		if (section.type != Darknet::ELayerType::YOLO)
		{
			continue;
		}

		const std::string & classes_name	= output_class	.at(number_of_yolo_layers);
		const std::string & objectness_name	= output_obj	.at(number_of_yolo_layers);
		number_of_yolo_layers ++;

		number_of_classes = section.find_int("classes");
		const int number_of_masks = section.find_int_array("mask").size();
		const auto & l = cfg.net.layers[idx - 1];
		number_of_boxes += (l.w * l.h * number_of_masks);

		// first combine the objectness with the classes
		Node node(section, "_mul_confs");
		node.type("Mul").add_input(classes_name).add_input(objectness_name);

		// remember this multiplication, because then we take those from *each* of the YOLO layers
		v.push_back(node.output);
	}

	// this is where we concatenate all of the multiplications to create the final "confidence" output

	Node node("confs");
	node.type("Concat").set_output("confs").add_attribute_INT("axis", 1); // 1 == concat on channel axis
	for (const auto & input : v)
	{
		node.add_input(input);
	}

	// this is one of the 2 outputs (also see "boxes")
	//
	// The middle value is the sum of prediction boxes across all YOLO layers.

	std::stringstream ss;
	ss << "Output of confidence values (objectness x probability) for all " << number_of_boxes << " prediction boxes";
	if (number_of_classes < 2)
	{
		ss << ".";
	}
	else if (number_of_classes == 2)
	{
		ss << " and both classes.";
	}
	else
	{
		ss << " and " << number_of_classes << " classes.";
	}

	if (number_of_yolo_layers > 1)
	{
		ss << " This output is a combination of ";
		if (number_of_yolo_layers == 2)
		{
			ss << " both YOLO layers.";
		}
		else
		{
			ss << " all " << number_of_yolo_layers << " layers.";
		}
	}

	auto output = graph->add_output();
	populate_input_output_dimensions(output, "confs", 1, number_of_boxes, number_of_classes);
	output->set_doc_string(ss.str());
	node.doc(ss.str());

	output_string += "[\"confs\"] ";

	return *this;
}


Darknet::VStr Darknet::ONNXExport::postprocess_yolo_boxes(const Darknet::VStr & output_tx_ty, const Darknet::VStr & output_tw_th)
{
	TAT(TATPARMS);

	// ...why 6?  Where does this magic value come from?
	const int magic_6			= 6;
	int number_of_yolo_layers	= 0;

	VStr results;

	for (int idx = 1; idx < cfg.net.n; idx ++)
	{
		auto & section = cfg.sections[idx];
		if (section.type != Darknet::ELayerType::YOLO)
		{
			continue;
		}

		// we found a YOLO layer

		const std::string & tx_ty_name = output_tx_ty.at(number_of_yolo_layers);
		const std::string & tw_th_name = output_tw_th.at(number_of_yolo_layers);
		number_of_yolo_layers ++;

		Darknet::VStr outputs;
		const auto & l = cfg.net.layers[idx];

		// first we deal with tx_ty...

		for (int i = 0; i < magic_6; i ++)
		{
			Node slice(section, "_slice_" + std::to_string(i) + "_tx_ty");
			slice.type("Slice").add_input(tx_ty_name);

			const std::map<std::string, int> initializers =
			{
				{slice.name + "_1_starts"	, i},
				{slice.name + "_2_ends"		, i + 1},
				{slice.name + "_3_axes"		, 1},
				{slice.name + "_4_steps"	, 1}
			};
			for (const auto & [key, val] : initializers)
			{
				onnx::TensorProto * initializer = graph->add_initializer();
				initializer->set_data_type(onnx::TensorProto::INT32);
				initializer->set_name(key);
				initializer->add_dims(1);
				initializer->add_int32_data(val);
				slice.add_input(key);
			}

			onnx::TensorProto tensor;
			tensor.set_data_type(bit_size == 32 ? onnx::TensorProto_DataType_FLOAT : onnx::TensorProto_DataType_FLOAT16);
			tensor.add_dims(1);
			tensor.add_dims(1);
			tensor.add_dims(l.h);
			tensor.add_dims(l.w);

			// the vector only gets used when dealing with FP16
			std::vector<std::uint16_t> v;

			for (int h = 0; h < l.h; h ++)
			{
				for (int w = 0; w < l.w; w ++)
				{
					if (i % 2 == 0)
					{
						// X values
						if (bit_size == 32)
						{
							tensor.add_float_data(w);
						}
						else
						{
							v.push_back(Darknet::convert_to_fp16(w));
						}
					}
					else
					{
						// Y values
						if (bit_size == 32)
						{
							tensor.add_float_data(h);
						}
						else
						{
							v.push_back(Darknet::convert_to_fp16(h));
						}
					}
				}
			}
			if (not v.empty())
			{
				tensor.set_raw_data(v.data(), v.size() * sizeof(std::uint16_t));
			}

			Node constants(section, "_const_tx_ty");
			constants.type("Constant");
			auto attr = constants.node->add_attribute();
			attr->set_name("value");
			attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
			*attr->mutable_t() = tensor;

			Node add(section, "_add_" + std::to_string(i) + "_tx_ty");
			add.type("Add").add_input(slice.output).add_input(constants.output);

			outputs.push_back(add.output);
		}

		// then we deal with tw_th...

		const auto masks		= section.find_int_array("mask");		// e.g., "3, 4, 5"
		const auto anchors		= section.find_float_array("anchors");	// e.g., "8, 8, 10, 10, 15, 12, 41, 41, 48, 47, 73, 70"
		const float stride_x	= cfg.net.w / l.w;
		const float stride_y	= cfg.net.h / l.h;

		// So with that LEGO example, the values of interest to this YOLO head are "41, 41, 48, 47, 73, 70"  (width followed by height)

		if (masks.size() * 2 != magic_6)
		{
			throw std::runtime_error(cfg_fn.string() + ": " + Darknet::to_string(section.type) + " on line #" + std::to_string(section.line_number) + ": the ONNX export tool expected 6 values (3 masks) per YOLO head, not " + std::to_string(masks.size()));
		}

		Darknet::VFloat multiplier;
		for (const auto & mask : masks)
		{
			multiplier.push_back(anchors[mask * 2 + 0] / stride_x);
			multiplier.push_back(anchors[mask * 2 + 1] / stride_y);
		}

		for (int i = 0; i < magic_6; i ++)
		{
			Node slice(section, "_slice_" + std::to_string(i) + "_tw_th");
			slice.type("Slice").add_input(tw_th_name);

			const std::map<std::string, int> initializers =
			{
				{slice.name + "_1_starts"	, i},
				{slice.name + "_2_ends"		, i + 1},
				{slice.name + "_3_axes"		, 1},
				{slice.name + "_4_steps"	, 1}
			};
			for (const auto & [key, val] : initializers)
			{
				onnx::TensorProto * initializer = graph->add_initializer();
				initializer->set_data_type(onnx::TensorProto::INT32);
				initializer->set_name(key);
				initializer->add_dims(1);
				initializer->add_int32_data(val);
				slice.add_input(key);
			}

			onnx::TensorProto tensor;
			if (bit_size < 32)
			{
				tensor.set_data_type(onnx::TensorProto_DataType_FLOAT16);
				std::vector<std::uint16_t> v;
				v.push_back(Darknet::convert_to_fp16(multiplier[i]));
				tensor.set_raw_data(v.data(), v.size() * sizeof(std::uint16_t));
			}
			else
			{
				tensor.set_data_type(onnx::TensorProto_DataType_FLOAT);
				tensor.add_float_data(multiplier[i]);
			}
			Node constants(section, "_const_" + std::to_string(i) + "_tw_th");
			constants.type("Constant");
			auto attr = constants.node->add_attribute();
			attr->set_name("value");
			attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
			*attr->mutable_t() = tensor;

			Node mul(section, "_mul_" + std::to_string(i) + "_tw_th");
			mul.type("Mul").add_input(slice.output).add_input(constants.output);

			outputs.push_back(mul.output);
		}

		// now make 2 groups, each with 6 inputs
		for (const std::string name : {"lhs", "rhs"})
		{
			Node concat(section, "_concat_" + name);
			concat.type("Concat").add_attribute_INT("axis", 1);

			// only take every 2nd output, based on whether or not we're starting with "even" (zero) or "odd" (one)
			for (size_t i = (name == "lhs" ? 0 : 1); i < outputs.size(); i += 2)
			{
				concat.add_input(outputs[i]);
			}

			onnx::TensorProto tensor;
			const float f = (name == "lhs" ? l.w : l.h);
			if (bit_size == 32)
			{
				tensor.set_data_type(onnx::TensorProto_DataType_FLOAT);
				tensor.add_float_data(f);
			}
			else
			{
				std::vector<std::uint16_t> v;
				v.push_back(Darknet::convert_to_fp16(f));
				tensor.set_data_type(onnx::TensorProto_DataType_FLOAT16);
				tensor.set_raw_data(v.data(), v.size() * sizeof(std::uint16_t));
			}
			Node constants(section, "_const_" + name);
			constants.type("Constant");
			auto attr = constants.node->add_attribute();
			attr->set_name("value");
			attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
			*attr->mutable_t() = tensor;

			Node div(section, "_div_" + name);
			div.type("Div").add_input(concat.output).add_input(constants.output);

			results.push_back(div.output);
		}
	}

	return results;
}


Darknet::ONNXExport & Darknet::ONNXExport::postprocess_yolo_boxes(const Darknet::VStr & v)
{
	TAT(TATPARMS);

	/* "v" should be set to the even-and-odd Div output, such as:
	 *
	 *		- N162_L30_yolo_div_lhs
	 *		- N165_L30_yolo_div_rhs
	 *		- N204_L37_yolo_div_lhs
	 *		- N207_L37_yolo_div_rhs
	 */

	int number_of_yolo_layers = 0;
	int number_of_boxes = 0;

	VStr results;

	for (int idx = 1; idx < cfg.net.n; idx ++)
	{
		auto & section = cfg.sections[idx];
		if (section.type != Darknet::ELayerType::YOLO)
		{
			continue;
		}

		// we found a YOLO layer

		VStr outputs_sub;
		VStr outputs_add;
		const std::string & lhs = v.at(number_of_yolo_layers * 2 + 0); // N162_L30_yolo_div_lhs
		const std::string & rhs = v.at(number_of_yolo_layers * 2 + 1); // N165_L30_yolo_div_rhs
		number_of_yolo_layers ++;
		const auto & l = cfg.net.layers[idx];
		const int number_of_masks = section.find_int_array("mask").size();
		const int size = number_of_masks * l.w * l.h;
		number_of_boxes += size;
		VInt reshape_vector = {1, size, 1};

		for (const auto & name : {lhs, rhs})
		{
			Node slice1(section, "_slice");
			const std::map<std::string, int> constants1 =
			{
				{slice1.name + "_1_starts"	, 0},
				{slice1.name + "_2_ends"	, 3},
				{slice1.name + "_3_axes"	, 1},
				{slice1.name + "_4_steps"	, 1}
			};
			for (const auto & [key, val] : constants1)
			{
				onnx::TensorProto * initializer = graph->add_initializer();
				initializer->set_name(key);
				initializer->set_doc_string(slice1.node->doc_string());
				initializer->set_data_type(onnx::TensorProto::INT32);
				initializer->add_dims(1);
				initializer->add_int32_data(val);
			}
			slice1
				.type("Slice")
				.add_input(name)
				.add_input("_1_starts"	)
				.add_input("_2_ends"	)
				.add_input("_3_axes"	)
				.add_input("_4_steps"	);

			const auto reshape1_const = Node(section, reshape_vector).output;
			Node reshape1(section, "_reshape");
			reshape1.type("Reshape").add_input(slice1.output).add_input(reshape1_const);

			Node slice2(section, "_slice");
			const std::map<std::string, int64_t> constants2 =
			{
				{slice2.name + "_1_starts"	, 3},
				{slice2.name + "_2_ends"	, std::numeric_limits<int64_t>::max()},
				{slice2.name + "_3_axes"	, 1},
				{slice2.name + "_4_steps"	, 1}
			};
			for (const auto & [key, val] : constants2)
			{
				onnx::TensorProto * initializer = graph->add_initializer();
				initializer->set_name(key);
				initializer->set_doc_string(slice1.node->doc_string());
				initializer->set_data_type(onnx::TensorProto::INT64);
				initializer->add_dims(1);
				initializer->add_int64_data(val);
			}
			slice2
			.type("Slice")
			.add_input(name)
			.add_input("_1_starts"	)
			.add_input("_2_ends"	)
			.add_input("_3_axes"	)
			.add_input("_4_steps"	);

			const auto reshape2_const = Node(section, reshape_vector).output;
			Node reshape2(section, "_reshape");
			reshape2.type("Reshape").add_input(slice2.output).add_input(reshape2_const);

			Node const_half(section, 0.5f, bit_size);
			Node mul2(section, "_mul");
			mul2.type("Mul").add_input(reshape2.output).add_input(const_half.output);

			Node sub(section, "_sub");
			sub.type("Sub").add_input(reshape1.output).add_input(mul2.output);

			Node add(section, "_add");
			add.type("Add").add_input(sub.output).add_input(reshape2.output);

			outputs_sub.push_back(sub.output);
			outputs_add.push_back(add.output);
		}

		Node concat(section, "_concat_boxes");
		concat.type("Concat").add_attribute_INT("axis", 2); // 2 == concat on ...?
		for (const auto & sub : outputs_sub)
		{
			concat.add_input(sub);
		}
		for (const auto & add : outputs_add)
		{
			concat.add_input(add);
		}

		reshape_vector.push_back(4);
		const auto reshape_const = Node(section, reshape_vector).output;
		Node reshape(section, "_reshape_boxes");
		reshape.type("Reshape").add_input(concat.output).add_input(reshape_const);

		results.push_back(reshape.output);
	}

	// last node -- output the final "boxes"

	Node node("boxes");
	node.type("Concat").set_output("boxes").add_attribute_INT("axis", 1); // 1 == concat on channel axis
	for (const auto & input : results)
	{
		node.add_input(input);
	}

	// this is the 2nd output from the network (also see "confs")

	std::stringstream ss;
	ss << "Output of 4 box coordinates for all " << number_of_boxes << " prediction boxes.";
	if (number_of_yolo_layers > 1)
	{
		ss << " This output is a combination of ";
		if (number_of_yolo_layers == 2)
		{
			ss << " both YOLO layers.";
		}
		else
		{
			ss << " all " << number_of_yolo_layers << " layers.";
		}
	}

	auto output = graph->add_output();
	populate_input_output_dimensions(output, "boxes", 1, number_of_boxes, 1, 4);
	output->set_doc_string(ss.str());
	node.doc(ss.str());

	output_string += "[\"boxes\"] ";

	return *this;
}


Darknet::ONNXExport & Darknet::ONNXExport::populate_graph_postprocess()
{
	TAT(TATPARMS);

	if (not postprocess_boxes)
	{
		// ...how did we get here if we're not doing the post-processing of boxes?
		throw std::runtime_error("post-processing of boxes is disabled");
	}

	VStr output_tx_ty;
	VStr output_tw_th;
	VStr output_obj;
	VStr output_class;

	for (int idx = 0; idx < cfg.net.n; idx ++)
	{
		auto & section = cfg.sections[idx];
		if (section.type == Darknet::ELayerType::YOLO)
		{
			const auto results = postprocess_yolo_slice_and_concat(section);
			postprocess_yolo_tx_ty	(section, results, output_tx_ty	);
			postprocess_yolo_tw_th	(section, results, output_tw_th	);
			postprocess_yolo_to		(section, results, output_obj	);
			postprocess_yolo_class	(section, results, output_class	);
		}
	}

	// every output vector should have exactly the same number of items, which should match the number of YOLO heads in the network
	if (output_tx_ty.size() != output_tw_th.size() or
		output_tx_ty.size() != output_obj.size() or
		output_tx_ty.size() != output_class.size())
	{
		throw std::runtime_error("output vectors don't match"
			" (tx+ty="	+ std::to_string(output_tx_ty.size()) +
			", tw+th="	+ std::to_string(output_tw_th.size()) +
			", obj="	+ std::to_string(output_obj.size()) +
			", class="	+ std::to_string(output_class.size()) +
			")");
	}

	// every objectness and class chains are combined together to get the confidence values "confs"
	postprocess_yolo_confs(output_obj, output_class);

	// last thing we need to do is deal with "boxes"
	const auto results = postprocess_yolo_boxes(output_tx_ty, output_tw_th);
	postprocess_yolo_boxes(results);

	return *this;
}
