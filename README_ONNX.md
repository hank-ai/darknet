# Notice

> [!CAUTION]
> **This is not the primary "readme" file!**  Please start with [README.md](README.md#Building).  This file contains an optional subset of the instructions which is specific to ONNX export.

# ONNX

The ONNX export tool is optional in the Darknet/YOLO codebase.  To build and use this tool, you also need to install support for the Google Protocol Buffers.

On Ubuntu Linux, this is done using:

    sudo apt-get install libprotobuf-dev

On Windows, this is done using:

    .\vcpkg.exe install protobuf:x64-windows

> [!WARNING]
> Need confirmation that the Windows command is correct.

Once this additional dependency has been installed, please continue with the usual Darknet/YOLO build steps as described in the [README.md](README.md#Building).

As of August 2025, the Darknet/YOLO ONNX export tool has only been tested with the following *stock* configuration files:

- YOLOv4-tiny.cfg
- YOLOv4-tiny-3l.cfg
- YOLOv4.cfg

> [!TIP]
> Software developers wanting more information on the ONNX process should see [the instructions on using `onnx.proto3`](src-onnx/onnx.proto3.pb.txt).
