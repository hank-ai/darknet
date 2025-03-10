# Notice

**This is not the primary "readme" file!**  Please start with [README.md](README.md#Building).  This file contains an optional subset of the instructions which is specific to NVIDIA GPUs.

# NVIDIA GPUs

If you have a modern GPU made by NVIDIA which is CUDA-capable, then Darknet can be built to take advantage of the GPU to process images and video frames.  This will make Darknet/YOLO run much faster.

The NVIDIA GPUs which are supported by CUDA and cuDNN include those from the Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Ada Lovelace, Hopper, and Blackwell lines.  If uncertain about whether your GPU is supported with CUDA and cuDNN, please find your GPU [on this NVIDIA page](https://developer.nvidia.com/cuda-gpus).

If you'd like to prevent the Darknet build process from attempting to detect NVIDIA and CUDA, you can define `DARKNET_TRY_CUDA=OFF` like this when running CMake:

```sh
cmake -DCMAKE_BUILD_TYPE=Release -DDARKNET_TRY_CUDA=OFF ..
```

## Linux

The NVIDIA CUDA and cuDNN software must be installed **prior** to running `cmake` for Darknet/YOLO.  This is because as part of Darknet's CMake process, it will attempt to identify your GPU and the necessary CUDA and cuDNN files.

* If you install CUDA and cuDNN after having already built Darknet/YOLO, you'll need to delete your `src/darknet/build/CMakeCache.txt` file to force CMake to re-detect CUDA, cuDNN, your GPU, and the necessary files.
* Visit <https://developer.nvidia.com/cuda-downloads> to download and install CUDA.
* Visit <https://developer.nvidia.com/rdp/cudnn-download> or <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#cudnn-package-manager-installation-overview> to download and install cuDNN.
* Once you install CUDA make sure you can run both `nvcc` and `nvidia-smi`.  You may have to [modify your `PATH` variable](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#mandatory-actions).

## Windows

On Windows, the NVIDIA CUDA and cuDNN software must be installed **after** installing Visual Studio.  If you install or upgrade Visual Studio after, then you must re-install CUDA and cuDNN.  This is because part of the NVIDIA software is installed within the Visual Studio folders.

* If you install CUDA and cuDNN after having already built Darknet/YOLO, you'll need to delete your `c:/src/darknet/build/CMakeCache.txt` file to force CMake to re-detect CUDA, cuDNN, your GPU, and the necessary files.
* Visit <https://developer.nvidia.com/cuda-downloads> to download and install CUDA.
* Visit <https://developer.nvidia.com/rdp/cudnn-download> or <https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#download-windows> to download and install cuDNN.
* Remember to reboot once you've installed the NVIDIA driver and software.
* After you reboot, make sure you can run both `nvcc.exe` and `nvidia-smi.exe`.  You may have to modify your `PATH` variable.
* Once you download cuDNN, unzip and copy the bin, include, and lib directories into `C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/[version]/`.  You may need to overwrite some files.
* CUDA **must** be installed **after** Visual Studio.  If you re-install or upgrade Visual Studio, remember to also re-install NVIDIA CUDA and cuDNN before re-building Darknet.

### Additional Information for Windows

While building Darknet, if you get an error about some missing CUDA or cuDNN DLLs such as `cublas64_12.dll`, please manually copy the CUDA `.dll` files into the same output directory as `Darknet.exe`.  For example:
```bat
copy "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin\*.dll" c:\src\darknet\build\src-cli\Release\
```
That is an example!  Check to make sure what version you are running, and run the command that is appropriate for what you have installed.  Once you copy the necessary DLLs, remember to re-start the `msbuild.exe` command that failed.
