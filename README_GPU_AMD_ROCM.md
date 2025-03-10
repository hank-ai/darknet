# Notice

**This is not the primary "readme" file!**  Please start with [README.md](README.md#Building).  This file contains an optional subset of the instructions which is specific to AMD GPUs.

# AMD GPUs

If you have a modern GPU made by AMD which is supported by ROCm and HIP, then Darknet can be built to take advantage of the GPU to process images and video frames.  This will make Darknet/YOLO run much faster.

If you'd like to prevent the Darknet build process from attempting to detect AMD, ROCm, and HIP, you can define `DARKNET_TRY_ROCM=OFF` like this when running CMake:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DDARKNET_TRY_CUDA=OFF ..
```

## Linux

The AMD ROCm and HIP software must be installed **prior** to running `cmake` for Darknet/YOLO.  This is because as part of Darknet's CMake process, it will attempt to identify your GPU and the necessary ROCm and HIP files.

* If you install ROCm and HIP after having already built Darknet/YOLO, you'll need to delete your `src/darknet/build/CMakeCache.txt` file to force CMake to re-detect ROCm, HIP, your GPU, and the necessary files.
* Visit <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html> to download and install ROCm and HIP.


## Windows

TBD.

* https://rocm.docs.amd.com/projects/install-on-windows/en/latest/
