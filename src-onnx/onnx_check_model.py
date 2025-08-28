#!/bin/python3

# Darknet/YOLO:  https://codeberg.org/CCodeRun/darknet
# Copyright 2025 Stephane Charette
#
# Check the given file to see if it is a valid ONNX model.
# Call the script with a single .onnx filename.


import sys
import onnx  # sudo apt-get install python3-onnx


if len(sys.argv) != 2:
    print("Must specify one .onnx filename to check.")
    exit(1)

try:
    onnx.checker.check_model(sys.argv[1], full_check=True)
    print("ONNX model is valid.")
except Exception as e:
    print(f"ONNX model validation failed: {e}")
