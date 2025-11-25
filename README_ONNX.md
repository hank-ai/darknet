# Notice

> [!CAUTION]
> **This is not the primary "readme" file!**  Please start with [README.md](README.md#Building).  This file contains an optional subset of the instructions which is specific to ONNX export.

# ONNX

The ONNX export tool is optional in the Darknet/YOLO codebase.  To build and use this tool, you also need to install support for Google Protocol Buffers prior to building and installing Darknet/YOLO.

On Ubuntu Linux, this is done using:

    sudo apt-get install libprotobuf-dev protobuf-compiler

On Windows, this is done using:

    .\vcpkg.exe install protobuf:x64-windows

> [!WARNING]
> Need confirmation that the Windows command is correct.

Once this additional dependency has been installed, please continue with the usual Darknet/YOLO build steps as described in the [README.md](README.md#Building).

> [!TIP]
> The CMake flags `-DDARKNET_TRY_ONNX=ON` and `-DDARKNET_TRY_ONNX=OFF` can be used to skip building the ONNX tool.  The default in v5.0 was `-DDARKNET_TRY_ONNX=OFF`.  In v5.1, the default was changed to `ON`.

As of November 2025, the Darknet/YOLO ONNX export tool has only been tested with the following *stock* configuration files:

- YOLOv4-tiny.cfg
- YOLOv4-tiny-3l.cfg
- YOLOv4.cfg

> [!TIP]
> Software developers wanting more information on the ONNX process should see [the instructions on using `onnx.proto3`](src-onnx/onnx.proto3.txt).
