/* Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
 * Copyright 2024-2026 Stephane Charette
 */

#include "darknet_node.hpp"


namespace
{
	static auto & cfg_and_state = Darknet::CfgAndState::get();
}


onnx::GraphProto *	Darknet::Node::graph		= nullptr;
std::string			Darknet::Node::cfg_filename	= "unknown";
size_t				Darknet::Node::node_counter	= 0;
Darknet::MIdStr		Darknet::Node::output_per_layer_index;


Darknet::Node::Node(const std::string & n) :
	node(nullptr),
	layer_index(0),
	counter(node_counter ++)
{
	TAT(TATPARMS);

	init(n);

	return;
}


Darknet::Node::Node(const Darknet::CfgSection & section, const std::string & append) :
	node(nullptr),
	layer_index(0),
	counter(node_counter ++)
{
	TAT(TATPARMS);

	layer_index = section.index;

	std::stringstream ss;
	ss << "L" << (section.index) << "_" << section.name << append;
	init(ss.str());

	doc(section);

	return;
}


Darknet::Node::Node(const Darknet::CfgSection & section, const float f, const size_t bit_size) :
	Node(section, "_constant")
{
	TAT(TATPARMS);

	onnx::TensorProto tensor;
	tensor.set_name(name);
	tensor.set_doc_string(node->doc_string());
	if (bit_size == 32 or bit_size == 8)
	{
		tensor.set_data_type(onnx::TensorProto_DataType_FLOAT);
		tensor.add_float_data(f);
	}
	else if (bit_size == 16)
	{
		std::vector<std::uint16_t> v;
		v.push_back(Darknet::convert_to_fp16(f));
		tensor.set_data_type(onnx::TensorProto_DataType_FLOAT16);
		tensor.set_raw_data(v.data(), v.size() * sizeof(std::uint16_t));
	}
	else
	{
		throw std::invalid_argument(name + ": cannot add float=" + std::to_string(f) + " with unsupported size_bits=" + std::to_string(bit_size));
	}

	type("Constant");

	auto attr = node->add_attribute();
	attr->set_name("value");
	attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
	*attr->mutable_t() = tensor;

	return;
}


Darknet::Node::Node(const Darknet::CfgSection & section, const Darknet::VInt & v) :
	Node(section, "_constant")
{
	TAT(TATPARMS);

	onnx::TensorProto tensor;
	tensor.set_name(name);
	tensor.set_data_type(onnx::TensorProto_DataType_INT64);
	tensor.add_dims(v.size());
	for (const auto & i : v)
	{
		tensor.add_int64_data(i);
	}
	tensor.set_doc_string(node->doc_string());

	type("Constant");

	auto attr = node->add_attribute();
	attr->set_name("value");
	attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
	*attr->mutable_t() = tensor;

	return;
}


Darknet::Node::~Node()
{
	TAT(TATPARMS);

	return;
}


Darknet::Node & Darknet::Node::init(const std::string & n)
{
	TAT(TATPARMS);

	if (node == nullptr)
	{
		node = graph->add_node();
	}

	name = "N" + std::to_string(counter) + "_" + n;
	node->set_name(name);
	set_output();
	doc(name);

	if (cfg_and_state.is_verbose)
	{
		*cfg_and_state.output << "=> " << name << std::endl;
	}

	return *this;
}


Darknet::Node & Darknet::Node::doc(const std::string & str)
{
	TAT(TATPARMS);

	node->set_doc_string(str);

	return *this;
}


Darknet::Node & Darknet::Node::doc(const Darknet::CfgSection & section)
{
	TAT(TATPARMS);

	std::stringstream ss;
	ss << cfg_filename << " line #" << section.line_number << " [" << section.name << ", layer #" << section.index << "]";

	if (layer_index == 0)
	{
		layer_index = section.index;
	}

	return doc(ss.str());
}


Darknet::Node & Darknet::Node::doc_append(const std::string & str)
{
	TAT(TATPARMS);

	std::string doc_string = node->doc_string();
	doc_string += str;

	return doc(doc_string);
}


Darknet::Node & Darknet::Node::type(const std::string & type)
{
	TAT(TATPARMS);

	node->set_op_type(type);

	return *this;
}


Darknet::Node & Darknet::Node::add_input(int idx)
{
	TAT(TATPARMS);

	if (idx == -1 and layer_index == 0)
	{
		// special case for the very first node in the graph which needs to take input from "frame"
		return add_input("frame");
	}

	// if the index is positive, then we have an absolute value; otherwise it is relative to the current index
	if (idx < 0)
	{
		idx += layer_index;
	}

	if (output_per_layer_index.count(idx) == 0)
	{
		throw std::runtime_error("cannot find output for layer #" + std::to_string(idx) + " when attempting to add an input for node " + name);
	}

	const auto & input = output_per_layer_index.at(idx);

	return add_input(input);
}


Darknet::Node & Darknet::Node::add_input(const std::string & input)
{
	TAT(TATPARMS);

	if (input.empty())
	{
		throw std::runtime_error("name cannot be blank when adding input to node " + name);
	}

	if (input[0] == '_')
	{
		// for example, if input is "_weights" then we append this to the end of the node's name
		node->add_input(name + input);
	}
	else
	{
		node->add_input(input);
	}

	return *this;
}


Darknet::Node & Darknet::Node::set_output(const std::string & out)
{
	TAT(TATPARMS);

	// as far as I know we only ever have 1 output per node, so clear any previous output name
	node->clear_output();

	if (out.empty())
	{
//		output = "N" + std::to_string(counter) + "_L" + std::to_string(layer_index) + "_output";
//		output = name + "_output";
		output = name;
	}
	else
	{
		output = out;
	}

	node->add_output(output);
	output_per_layer_index[layer_index] = output;

	return *this;
}


Darknet::Node & Darknet::Node::add_attribute_INT(const std::string & key, const int val)
{
	TAT(TATPARMS);

	if (key.empty())
	{
		throw std::invalid_argument("key name cannot be empty when adding a new INT attribute to " + name);
	}

	onnx::AttributeProto * attrib = node->add_attribute();
	attrib->set_name(key);
	attrib->set_type(onnx::AttributeProto::INT);
	attrib->set_i(val);

	return *this;
}


Darknet::Node & Darknet::Node::add_attribute_INTS(const std::string & key, const Darknet::VInt & val)
{
	TAT(TATPARMS);

	if (key.empty())
	{
		throw std::invalid_argument("key name cannot be empty when adding a new INTS attribute to " + name);
	}

	onnx::AttributeProto * attrib = node->add_attribute();
	attrib->set_name(key);
	attrib->set_type(onnx::AttributeProto::INTS);
	for (const int & i : val)
	{
		attrib->add_ints(i);
	}

	return *this;
}


Darknet::Node & Darknet::Node::add_attribute_STR(const std::string & key, const std::string & val)
{
	TAT(TATPARMS);

	if (key.empty())
	{
		throw std::invalid_argument("key name cannot be empty when adding a new STR attribute to " + name);
	}

	onnx::AttributeProto * attrib = node->add_attribute();
	attrib->set_name(key);
	attrib->set_type(onnx::AttributeProto::STRING);
	attrib->set_s(val);

	return *this;
}


Darknet::Node & Darknet::Node::add_attribute_FLOAT(const std::string & key, const float val)
{
	TAT(TATPARMS);

	if (key.empty())
	{
		throw std::invalid_argument("key name cannot be empty when adding a new FLOAT attribute to " + name);
	}

	onnx::AttributeProto * attrib = node->add_attribute();
	attrib->set_name(key);
	attrib->set_type(onnx::AttributeProto::FLOAT);
	attrib->set_f(val);

	return *this;
}


std::string Darknet::Node::get_output_for_layer_index(const int idx)
{
	TAT(TATPARMS);

	if (output_per_layer_index.count(idx) == 0)
	{
		throw std::runtime_error("cannot find output for layer #" + std::to_string(idx));
	}

	return output_per_layer_index.at(idx);
}
