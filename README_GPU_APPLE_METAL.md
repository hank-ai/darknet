# Notice

> [!CAUTION]
> **This is not the primary README file.** Start with [README.md](README.md#building). This page only covers Apple GPU support on macOS.

# Apple GPUs

Darknet uses the GPU in Apple silicon Macs through Apple's Metal, Metal Performance Shaders (MPS), and MPSGraph frameworks. MPS accelerates inference, and MPSGraph handles training. Both are included when building with Apple GPU support, and are available without an additional install when you install the Xcode development tools.

Apple GPU training currently supports fewer network configurations and training options than CUDA or ROCm. See the [MPSGraph and CUDA training feature overview](#mpsgraph-and-cuda-training-feature-overview) below for a comparison. In short, the currently supported configurations are:

- YOLOv4-tiny (`yolov4-tiny.cfg`)
- YOLOv4-tiny-3l (`yolov4-tiny-3l.cfg`)
- YOLOv7-tiny (`yolov7-tiny.cfg`)

These configs, in particular the YOLOv4 variants, are currently the most recommended configs to use and thus were targeted first. See: [https://www.ccoderun.ca/programming/darknet_faq/#configuration_template](https://www.ccoderun.ca/programming/darknet_faq/#configuration_template)

You do not need to have an expensive high tier Mac to already start training your own custom networks. The mentioned recommended configs just need a couple of GBs of memory to train, and such a base model Macbook or Mac Mini is already more than capable of training your own network in a couple of minutes!

Inference has been tested on a MacBook Neo (the Macbook with an iPhone chip in it) and confirmed working. We would like to know if training works on these Macs too, if you have a MacBook Neo and are interested in this, please join us in the Darknet/YOLO Discord.

We would like to get GPU acceleration for inference to Intel Macs. If you have an Intel Mac and are interested in this, please join us on the Darknet/YOLO Discord.

> **Tip**
>
> [DarkMark now fully supports Mac](https://codeberg.org/CCodeRun/DarkMark), and it is highly recommended to install DarkMark alongside Darknet to manage your datasets!

## Inference

Apple GPU inference has been tested with a large set of models, such as YoloV4 and YoloV4-tiny, Yolov7 and Yolov7-tiny. Darknet uses the GPU for supported operations and runs the remaining operations on the CPU. The amount of GPU acceleration therefore depends on the network configuration. The aforementioned models have been tested to have no CPU fallbacks remaining.

## Training

Apple GPU training currently officially supports YOLOv4-tiny, YOLOv4-tiny-3l, and YOLOv7-tiny using 32-bit floating-point precision (FP32). The supported settings and current limitations are described in the tables below.

Training requires fixed image dimensions, so set `random=0` in the configuration's YOLO sections. If the configuration uses an unsupported layer or training option, Darknet stops at startup and explains which setting needs attention.

## Requirements

- macOS 26 or newer for an Apple GPU build. CPU-only builds configured with `DARKNET_TRY_MPS=OFF` retain the older macOS deployment compatibility.
- A Mac with Apple silicon (M1 or newer, A18 or newer) is required for Apple GPU inference and training. Intel Macs currently require a CPU-only build.
- The macOS 26 SDK or newer. CMake rejects an older selected SDK.
- Xcode Command Line Tools installed:

```sh
xcode-select --install
```

- CMake 3.24 or newer.
- A C++ compiler provided by Xcode or the Xcode Command Line Tools.
- OpenCV installed as described in the [main macOS build instructions](README.md#macos-cmake-method).

If training reports that Apple Metal kernels cannot be built or loaded, check that the selected developer tools provide the macOS 26 SDK or newer with `xcrun --sdk macosx --show-sdk-version`. Update or select the appropriate Xcode installation, then rebuild Darknet.

## Build And Install

Install the dependencies and follow any instructions for removing an older installation in the [main macOS build instructions](README.md#macos-cmake-method). Then, from the Darknet source directory:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DDARKNET_TRY_MPS=ON
cmake --build build --parallel
sudo cmake --install build --prefix /usr/local
```

### Intel Macs (CPU-only)

Intel Macs do not currently support Darknet MPS acceleration. Build without Apple GPU support:

```sh
brew install openblas
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release -DDARKNET_TRY_MPS=OFF -DDARKNET_TRY_OPENBLAS=ON
cmake --build build-cpu --parallel
sudo cmake --install build-cpu --prefix /usr/local
```

## Check The Build

Once installed, open a new terminal and test the CLI:

```sh
darknet version
```

When Apple GPU support is available, Darknet prints the Apple device name near startup. For example:

```sh
Darknet V6 "Winston" v6.x [your build tag] [build date]
Apple MPS device: Apple M4 Pro, 17.8 GiB
```

When Apple training is enabled for a supported config, training also prints:

```sh
Apple MPS device: Apple M4 Pro, 17.8 GiB
Prepare additional network for mAP calculation...
Allocating workspace:  4.9 MiB
Allocating workspace:  4.9 MiB
Apple training backend: MPSGraph enabled
```

The device name and memory sizes depend on your Mac and network. If the Apple device and MPSGraph training messages are missing, check that the CMake output found Metal, MPS, and MPSGraph, and make sure `DARKNET_TRY_MPS=ON` was used.

## Train a network

Use your dataset's `.data` and `.cfg` files:

```sh
darknet detector train /path/to/dataset.data /path/to/network.cfg -map -dont_show
```

This starts training from scratch and calculates mAP during training. To start with existing weights, add their path after the `.cfg` file. Use `random=0` in the YOLO sections; changing image dimensions during training is not currently supported on Apple GPUs.

### MPSGraph And CUDA Training Feature Overview

The Apple column describes the current MPSGraph training path. CUDA/cuDNN has
broader coverage across Darknet's training options.

| Feature | Apple MPSGraph | CUDA/cuDNN | Short description |
| --- | --- | --- | --- |
| Precision | ~ | ✅ | Apple training is FP32 only; CUDA also supports FP16 and mixed-precision paths. |
| Input sizing and multi-scale training | ~ | ✅ | Apple graphs require fixed dimensions and `random=0`; CUDA supports dynamic multi-scale training. |
| YOLO configuration scope | ~ | ✅ | Apple supports two or three detection heads, as used by YOLOv4-tiny, YOLOv4-tiny-3l, and YOLOv7-tiny. |
| SGD optimizer | ✅ | ✅ | Momentum and weight decay are supported by both backends. |
| Adam optimizer | ❌ | ✅ | Apple MPSGraph training currently uses SGD only. |
| Learning-rate policies | ✅ | ✅ | Standard network learning-rate scheduling is used by both backends. |
| Batch and subdivisions | ✅ | ✅ | Fixed-shape batches and subdivision accumulation are supported on Apple. |
| Batch normalization | ✅ | ✅ | Convolutional batch normalization and running statistics are supported. |
| Convolution geometry | ~ | ✅ | Apple supports 1x1/3x3 convolutions with stride 1/2, but not dilation or special variants. |
| Grouped convolution and route | ~ | ✅ | Apple supports grouped convolution plus route concatenation/group slicing with restrictions. |
| Pooling and upsample | ~ | ✅ | Apple supports 2x2 stride-2 pooling, YOLOv4 SPP pooling, and nearest-neighbor 2x upsample. |
| Activation functions | ~ | ✅ | Apple supports linear, leaky, and logistic activations; CUDA supports a wider set. |
| YOLO loss options | ~ | ✅ | Apple requires CIoU and has restrictions on thresholds, focal loss, label smoothing, and class remapping. |
| Advanced training switches | ~ | ✅ | See the detailed switch table below; several optional training features are not supported on Apple. |
| Sequence/tracking training | ❌ | ✅ | Apple requires `time_steps=1` and does not support `track=1`. |
| Multi-GPU training | ❌ | ✅ | Apple MPSGraph training supports exactly one GPU/device; CUDA supports multi-GPU training. |

#### Advanced Training Switches

Darknet checks these settings before Apple GPU training starts. If a setting is unsupported, training stops and explains which setting needs attention.

| Config switch | Apple MPSGraph | CUDA/cuDNN | Short description |
| --- | --- | --- | --- |
| `adversarial`, `adversarial_lr` | ❌ | ✅ | Adversarial training is not supported on Apple GPUs. |
| `dynamic_minibatch` | ❌ | ✅ | Apple requires the batch size to stay fixed during training. |
| `ema_alpha` | ❌ | ✅ | Exponential moving-average weight updates are not supported. |
| `badlabels_rejection_percentage`, `num_sigmas_reject_badlabels`, `equidistant_point` | ❌ | ✅ | Bad-label rejection and equidistant-point filtering are not supported. |
| `weights_reject_freq` | ❌ | ✅ | Periodic weight rejection is not supported. |
| `try_fix_nan` | ❌ | ✅ | Apple training does not provide the NaN-fixing path. |
| `track` | ❌ | ✅ | Sequence tracking training is not supported. |
| `time_steps` other than `1` | ~ | ✅ | Apple supports only `time_steps=1`. |
| `sequential_subdivisions` different from `subdivisions` | ~ | ✅ | Apple requires sequential and configured subdivisions to match. |
| Layer-level `onlyforward`, `stopbackward`, or `clip` | ❌ | ✅ | These layer controls are not supported in the Apple training graph. |

Legend: ✅ supported, ~ partially supported or restricted, ❌ unsupported.

## Using Darknet as a library

Keep each network on the same thread for setup, changes, and predictions. Separate networks may run on separate threads.
