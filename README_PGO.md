# Notice

> [!CAUTION]
> **This is not the primary "readme" file!**  Please start with [README.md](README.md#Building).  This file contains an optional subset of the instructions which may provide an optimized Darknet/YOLO.

# GNU GCC PGO (Profile-Guided Optimization)

> [!IMPORTANT]
> This is an advanced topic, and is completely optional.  Most users should skip this readme.

## TLDR

```sh
cd ~/src/darknet/build
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 -DDARKNET_PROFILE_GEN=ON -DDARKNET_PROFILE_USE=OFF ..
make -j4 package
sudo dpkg -i darknet...etc...
#
# [run Darknet/YOLO to generate profile output files]
#
cd ~/src/darknet/build
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 -DDARKNET_PROFILE_GEN=OFF -DDARKNET_PROFILE_USE=ON ..
make -j4 package
sudo dpkg -i darknet...etc...
```

## Using GNU GCC PGO (Profile-Guided Optimization)

When using GNU GCC -- the usual C/C++ compiler on most Linux distributions -- PGO (Profile-Guided Optimization) can be used to help Darknet/YOLO run faster.  PGO with Darknet/YOLO [makes a significant difference](#results) when using fast CPUs without a GPU.  It has marginal impact when using CUDA, or with tiny devices like Raspberry Pi.

PGO is a 2-step process.  First you turn it on to generate some compiler-specific profile files.  Then you tell the compiler to use those files to rebuild Darknet/YOLO as an optimized application.

Initially, when running `cmake` you must enable `-DDARKNET_PROFILE_GEN` and disable `-DDARKNET_PROFILE_USE`:

```sh
cd ~/src/darknet/build
cmake -DDARKNET_PROFILE_GEN=ON -DDARKNET_PROFILE_USE=OFF ..
make -j4 package
sudo dpkg -i darknet...etc...
```

Once Darknet has been built and installed, run a normal training session, or run inference on some videos and images so the necessary profile data is generated.

For example, download [the usual LEGO Gears project](https://www.ccoderun.ca/programming/yolo_faq/#datasets), and try running this command:

```sh
cd ~/nn/LegoGears
darknet_05_process_videos_multithreaded LegoGears DSCN*.MOV
```

This should create some `.gcda` and `.gcno` files in the `~/src/darknet/build/src-lib/CMakeFiles/darknetobjlib.dir/` subdirectory.

You must then rebuild Darknet to _use_ the profile data generated with the previous command:

```sh
cd ~/src/darknet/build
cmake -DDARKNET_PROFILE_GEN=OFF -DDARKNET_PROFILE_USE=ON ..
make -j4 package
sudo dpkg -i darknet...etc...
```

Now run the same Darknet command to see the change in FPS from the optimized Darknet/YOLO codebase.

## Problem Solving

If you have multiple compilers installed, you may need to specify which one CMake should be using.  If you get errors like this:

	darknet.cpp: warning: ‘darknet/build/src-lib/CMakeFiles/darknetobjlib.dir/darknet.cpp.gcda’ is version ‘B23*’, expected version ‘B33*’

See which compiler you have installed.  For example:

```sh
> ls -lh /usr/bin/g++*
lrwxrwxrwx 1 root root  6 Jan 31  2024 /usr/bin/g++ -> g++-13*
lrwxrwxrwx 1 root root 23 Apr  3  2024 /usr/bin/g++-12 -> x86_64-linux-gnu-g++-12*
lrwxrwxrwx 1 root root 23 Sep  4  2024 /usr/bin/g++-13 -> x86_64-linux-gnu-g++-13*
```

Note when CMake first runs, it displays which compiler will be used, and which one nvcc uses:

	> cmake -DCMAKE_BUILD_TYPE=Release -DDARKNET_TRY_CUDA=ON -DDARKNET_PROFILE_USE=ON ..
	-- Darknet v5.0-64-gb5c1c24e-dirty
	-- Darknet branch name: v5
	-- The C compiler identification is GNU 13.3.0		# <-- NOTE v13.3.0
	-- The CXX compiler identification is GNU 13.3.0
	-- Looking for a CUDA compiler - /usr/bin/nvcc
	-- CUDA detected. Darknet will use NVIDIA GPUs.  CUDA compiler is /usr/bin/nvcc.
	-- The CUDA compiler identification is NVIDIA 12.0.140		# <-- NOTE v12.0.140
	-- Detecting CUDA compiler ABI info

Please note how NVCC is using the C++ 12.0.140 compiler, while the rest of Darknet is built with version 13.3.0.

If you attempt to generate and use profile information with objects compiled with different compiler versions, you'll see errors such as these:

	libgcov profiling error:src-lib/CMakeFiles/darknetobjlib.dir/yolo_layer.cpp.gcda:Version mismatch - expected 12.3 (release) (B23*) got 13.3 (release) (B33*)
	libgcov profiling error:src-lib/CMakeFiles/darknetobjlib.dir/weights.cpp.gcda:Version mismatch - expected 12.3 (release) (B23*) got 13.3 (release) (B33*)
	libgcov profiling error:src-lib/CMakeFiles/darknetobjlib.dir/utils.cpp.gcda:Version mismatch - expected 12.3 (release) (B23*) got 13.3 (release) (B33*)

To prevent this, you have to force CMake to use the same compiler as NVCC.  So knowing the `nvcc` in the example above is using v12.0.140, you can force CMake to use the same compiler this way:

```sh
cd ~/src/darknet/build
rm CMakeCache.txt
cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12 -DDARKNET_PROFILE_GEN=ON -DDARKNET_PROFILE_USE=OFF ..
make -j4 package
sudo dpkg -i darknet...etc...
```

Once you rebuild and reinstall, you'll want to delete any profile files created by PGO with a different compiler.  You can do that with:

```sh
cd ~/src/darknet/build
find . -name '*.gcno' -print -delete
find . -name '*.gcda' -print -delete
```
