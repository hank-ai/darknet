#!/bin/python3

# Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
# Copyright 2025 Stephane Charette
#
# Display some information on nodes, inputs, outputs, and attributes.
# Call the script with a single .onnx filename.


import sys
import onnx  # sudo apt-get install python3-onnx


def get_shape(value_info):
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(dim.dim_param)
        else:
            shape.append(dim.dim_value)
    return shape


def get_tensor_shape_by_name(model, name):
    # Four places in the model where we may find what we want:
    #   1) graph.input
    #   2) graph.output
    #   3) graph.value_info
    #   4) graph.initializer

    for input in model.graph.input:
        if input.name == name:
            return get_shape(input)

    for output in model.graph.output:
        if output.name == name:
            return get_shape(output)

    for value_info in model.graph.value_info:
        if value_info.name == name:
            return get_shape(value_info)

    for initializer in model.graph.initializer:
        if initializer.name == name:
            return list(initializer.dims)

    return None


if len(sys.argv) != 2:
    print("Must specify one .onnx filename to check.")
    exit(1)

try:
    # best case scenario is if we can use shape_inference to get all the dimensions
    try:
        model = onnx.shape_inference.infer_shapes(model=onnx.load(sys.argv[1]), check_type=True, strict_mode=True)
    except Exception as e:
        print("Shape inference failed with the following exception:")
        print(f"{e}")
        print("Trying again to load the model directly.  Some dimensions may not be available.")
        model = onnx.load(sys.argv[1])

    # display a few bits of information before we go through the nodes
    print(f"Filename ............. {sys.argv[1]}")
    print(f"Doc string ........... {model.doc_string}")
    print(f"Domain ............... {model.domain}")
    print(f"Producer name ........ {model.producer_name}")
    print(f"Producer version ..... {model.producer_version}")
    print(f"Model version ........ {model.model_version}")
    print(f"Graph input size ..... {len(model.graph.input)}")
    print(f"Graph output size .... {len(model.graph.output)}")
    print(f"Graph node size ...... {len(model.graph.node)}")
    print(f"Graph initializers ... {len(model.graph.initializer)}")
    print(f"IR version ........... {model.ir_version}")

    for opset in model.opset_import:
        domain = opset.domain + ":" if opset.domain else ""
        print(f"Opset version ........ {domain}{opset.version}")

    for node in model.graph.node:
        print(f"\n{node.name:30}  OP: {node.op_type:35} DOC: {node.doc_string}")
        for input_name in node.input:
            shape = get_tensor_shape_by_name(model, input_name)
            print(f"                                IN: {input_name:35} DIM: {shape}")
        for output_name in node.output:
            shape = get_tensor_shape_by_name(model, output_name)
            print(f"                               OUT: {output_name:35} DIM: {shape}")
        if len(node.attribute) > 0:
            print("                            ATTRIB: ", end="")
            for attrib in node.attribute:
                print(f"{attrib.name}=", end="")
                if attrib.type == onnx.AttributeProto.FLOAT:
                    print(f"{attrib.f:.5f} ", end="")
                elif attrib.type == onnx.AttributeProto.INT:
                    print(f"{attrib.i} ", end="")
                elif attrib.type == onnx.AttributeProto.FLOATS:
                    print(f"{attrib.floats} ", end="")
                elif attrib.type == onnx.AttributeProto.INTS:
                    print(f"{attrib.ints} ", end="")
                elif attrib.type == onnx.AttributeProto.STRING:
                    print(f"{attrib.s.decode('utf-8')} ", end="")
                elif attrib.type == onnx.AttributeProto.TENSOR:
                    print(f"{onnx.numpy_helper.to_array(attrib.t)} (TENSOR) ", end="")
                else:
                    print(f"... (type #{attrib.type}) ", end="")
            print("")

    print("")

    for node in model.graph.input:
        shape = get_tensor_shape_by_name(model, node.name)
        if len(shape) == 4:
            # can we assume the shape is BCHW?
            print(f"Model input node:         {node.name:10} W={shape[3]:4} H={shape[2]:4} C={shape[1]:3} B={shape[0]:1}    {shape}")
        else:
            print(f"Model input node:         {node.name:10}     {shape}")

    for node in model.graph.output:
        shape = get_tensor_shape_by_name(model, node.name)
        if len(shape) == 4:
            # can we assume the shape is BCHW?
            print(f"Model output node:        {node.name:10} W={shape[3]:4} H={shape[2]:4} C={shape[1]:3} B={shape[0]:1}    {shape}")
        else:
            print(f"Model output node:        {node.name:10}     {shape}")
    exit(0)

except Exception as e:
    print(f"Exception: {e}")
    exit(2)
