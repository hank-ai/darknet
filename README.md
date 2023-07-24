# Darknet Object Detection Framework and YOLO

![darknet and hank.ai logos](artwork/darknet_and_hank_ai_logos.png)
* Read how [Hank.ai is helping the Darknet/YOLO community](https://hank.ai/darknet-welcomes-hank-ai-as-official-sponsor-and-commercial-entity/)
* Read the Darknet/YOLO FAQ:  https://www.ccoderun.ca/programming/darknet_faq/
* Join the Darknet/YOLO discord server:  https://discord.gg/zSq8rtW

# Papers

* Paper **YOLOv7**: https://arxiv.org/abs/2207.02696
* Paper **Scaled-YOLOv4**: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_Scaled-YOLOv4_Scaling_Cross_Stage_Partial_Network_CVPR_2021_paper.html
* Paper **YOLOv4**: https://arxiv.org/abs/2004.10934

# General Information

YOLOv7 is more accurate and faster than YOLOv5 by **120% FPS**, than YOLOX by **180% FPS**, than Dual-Swin-T by **1200% FPS**, than ConvNext by **550% FPS**, than SWIN-L by **500% FPS**, and PPYOLOE-X by **150% FPS**.

YOLOv7 surpasses all known object detectors in both speed and accuracy in the range from 5 FPS to 160 FPS and has the highest accuracy 56.8% AP among all known real-time object detectors with 30 FPS or higher on GPU V100, batch=1.

![comparison](https://user-images.githubusercontent.com/4096485/179425274-f55a36d4-8450-4471-816b-8c105841effd.jpg)

# Building

The various build methods available in the past have been merged together into a single unified solution.  Darknet now requires OpenCV, and uses CMake to generate the necessary project files.  ==Beware if you are following old tutorials with more complicated build steps, or build steps that don't match what is in this readme.==

## Linux CMake Method

These instructions assume a system running Ubuntu 22.04.

    sudo apt-get install build-essential git libopencv-dev
    mkdir ~/src
    cd ~/src
    git clone https://github.com/hank-ai/darknet
    cd darknet
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j4
    sudo cp libdarknet.so /usr/local/lib/
    sudo cp include/darknet.h /usr/local/include/
    sudo ldconfig

## Windows CMake Method

These instructions assume a brand new installation of Windows 11 22H2.

Open a normal `cmd.exe` command prompt window and run the following commands:

    winget install Git.Git
    winget install Kitware.CMake
    winget install Microsoft.VisualStudio.2022.Community

At this point we need to modify the Visual Studio installation to include support for C++ applications:

- click on the "Windows Start" menu and run "Visual Studio Installer"
- click on `Modify`
- select `Desktop Development With C++`
- click on `Modify` in the bottom-right corner, and then click on `Yes`

Once everything is downloaded and installed, click on the "Windows Start" menu again and select `Developer Command Prompt for VS 2022`.

> Advanced users:  Instead of running the `Developer Command Prompt`, you can run a normal command prompt or ssh into the device and manually run `"\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"`.

Run the following commands to install Microsoft VCPKG, which will then be used to build OpenCV:

    cd c:\
    mkdir c:\src
    cd c:\src
    git clone https://github.com/microsoft/vcpkg
    cd vcpkg
    bootstrap-vcpkg.bat
    vcpkg.exe integrate install
    vcpkg.exe integrate powershell
    vcpkg.exe install opencv[contrib,core,dnn,ffmpeg,jpeg,png,quirc,tiff,webp]:x64-windows

Be patient at this step.  This will take a long time to run.  It needs to download and build many things.

Assuming the previous step was successful, now we need to clone Darknet and build it.  During this step we also need to tell CMake where OpenCV was built by vcpkg in the previous steps:

    cd c:\src
    git clone https://github.com/hank-ai/darknet.git
    cd darknet
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=C:/src/vcpkg/scripts/buildsystems/vcpkg.cmake ..
    msbuild.exe /property:Platform=x64;Configuration=Release /target:Build -maxCpuCount -verbosity:normal -detailedSummary darknet.sln

> Advanced users:  Note that the output of the cmake command in the previous step is a "normal" Visual Studio solution file, darknet.sln.  If you are a software developer who regularly uses the Visual Studio GUI instead of `msbuild.exe` to build projects, you can ignore the command-line and load the Darknet project in Visual Studio or VS Code.

Once building has finished, you should now have a darknet.exe file you can run from the command-line.


When using this solution, see the flags in the first few lines of `Makefile`.  It is important to set these flags correctly.  If you want to use your CUDA-capable GPU, then you must also set the `ARCH=` flag, and have CUDA and CUDNN correctly installed.

## CMake

This solution works for all platforms.  You need to have the usual build tools installed, including `cmake`, `git`, and both C and C++ compilers.

    mkdir ~/src
    cd ~/src
    git clone https://github.com/hank-ai/darknet
    cd darknet
    mkdir build_release
    cd build_release
    cmake ..
    cmake --build . --target install --parallel 8

## Powershell

Typical solution for Windows.  You will need to have the usual build tools installed.  This is described here in the FAQ:  https://www.ccoderun.ca/programming/darknet_faq/#how_to_build_on_windows

    Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope Process
    md c:\src
    cd c:\src
    git clone https://github.com/hank-ai/darknet
    cd darknet
    ./build.ps1

There are many options available when running `build.ps1`.  For example:

    ./build.ps1 -UseVCPKG -EnableOPENCV -EnableCUDA -EnableCUDNN

## Vcpkg

This solution can be used from both Linux and Windows, but is much more common on Windows.  You can use [vcpkg](https://github.com/microsoft/vcpkg) to install old versions of Darknet.  The newer versions of Darknet are not available via vcpkg.

# Using Darknet

## CLI

This is not the full list of all commands supported by Darknet.  See [the previous readme](README_previous.md) for additional details and examples.

* Run a single image:  `darknet detector test animals.data animals.cfg animals_best.weights dog.jpg`
* Output coordinates:  `darknet detector test animals.data animals.cfg animals_best.weights -ext_output dog.jpg`
* Working with videos:  `darknet detector demo animals.data animals.cfg animals_best.weights -ext_output test.mp4`
* Reading from a webcam:  `darknet detector demo animals.data animals.cfg animals_best.weights -c 0`
* Smart webcam:  `darknet detector demo animals.data animals.cfg animals_best.weights http://192.168.0.80:8080/video?dummy=param.mjpg`
* Save results to a video:  `darknet detector demo animals.data animals.cfg animals_best.weights test.mp4 -out_filename res.avi`
* JSON and MJPEG server:  `darknet detector demo animals.data animals.cfg animals_best.weights test50.mp4 -json_port 8070 -mjpeg_port 8090 -ext_output`
* Running on a specific GPU:  `darknet detector demo animals.data animals.cfg animals_best.weights -i 1 test.mp4`
* To check accuracy mAP@IoU=50:  `darknet detector map animals.data animals.cfg animals_best.weights`
* To check accuracy mAP@IoU=75:  `darknet detector map animals.data animals.cfg animals_best.weights -iou_thresh 0.75`
* To train:  `darknet detector -map -dont_show train animals.data animals.cfg` (also see [the training section](#training) below

## Training

The simplest way to annotate and train is with the use of [DarkMark](https://github.com/stephanecharette/DarkMark) to create all of the necessary files.

If you'd rather manually setup the various files to train a custom network:

* Create a new folder where the files will be stored.  For this example, a neural network will be created to detect animals, so the following directory is created:  `~/nn/animals/`.
* Copy one of the Darknet configuration files you'd like to use as a template.  For example, see `cfg/yolov4-tiny.cfg`.  Place this in the folder you created.  For this example, we now have `~/nn/animals/animals.cfg`.
* Create a `animals.names` text file in the same folder where you placed the configuration file.  For this example, we now have `~/nn/animals/animals.cfg`.
* Edit the `animals.names` file with your text editor.  List the classes you want to use.  You need to have exactly 1 entry per line, with no blank lines and no comments.  For this example, the `.names` file will contain:
~~~
    dog
    cat
    bird
    horse
~~~

* Create a `animals.data` text file in the same folder.  For this example, the `.data` file will contain:
~~~
    classes = 4
    train = /home/username/nn/animals/animals_train.txt
    valid = /home/username/nn/animals/animals_valid.txt
    names = /home/username/nn/animals/animals.names
    backup = /home/username/nn/animals
~~~

* Create a folder where you'll store your images and annotations.  For example, this could be `~/nn/animals/dataset`.  Each image will need a coresponding `.txt` file which describes the annotations for that image.  The format of the `.txt` annotation files is very specific.  You cannot create these files by hand since each annotation needs to contain the exact coordinates for the annotation.  See [DarkMark](https://github.com/stephanecharette/DarkMark) or other similar software to annotate your images.
* Create the "train" and "valid" text files named in the `.data` file.  These two text files need to individually list all of the images which Darknet must use to train and for validation when calculating the mAP%.  Exactly one image per line.  The path and filenames may be relative or absolute.
* Modify your `.cfg` file with a text editor.
  * Make sure that `batch=64`.
  * Note the subdivisions.  Depending on the network dimensions and the amount of memory available on your GPU, you may need to increase the subdivisions.  The best value to use is `1` so start with that.
  * Note `max_batches=...`.  A good value to use when starting out is 2000 x the number of classes.  For this example, we have 4 animals, so 4 * 2000 = 8000.  For this example, we'll use `max_batches=8000`.
  * Note `steps=...`.  This should be set to 80% and 90% of `max_batches`.  For this example we'd use `steps=6400,7200`.
  * Note `width=...` and `height=...`.  These are the network dimensions.  The FAQ explains [how to calculate the best size to use](https://www.ccoderun.ca/programming/darknet_faq/#optimal_network_size).
  * Search for all instances of the line `classes=...` and modify it with the number of classes in your `.names` file.  For this example, we'd use `classes=4`.
  * Search for all instances of the line `filters=...` in the `[convolutional]` section **prior** to each `[yolo]` section.  The value to use is (number_of_classes + 5) * 3.  Meaning for this example, (4 + 5) * 3 = 27.  So we'd use `filters=27` on the appropriate lines.
* Start training!  Run the following commands:
~~~
    cd ~/nn/animals/
    ~/src/darknet/darknet detector -map -dont_show train animals.data animals.cfg
~~~

Be patient.  The best weights will be stored in `animals_best.weights`.  And the progress of training can be observed by viewing the `chart.png` file.

# Other Tools and Links

* To manage your Darknet/YOLO projects, annotate images, verify your annotations, and generate the necessary files to train with Darknet, [see DarkMark](https://github.com/stephanecharette/DarkMark).
* For a robust alternative CLI to Darknet, to use image tiling, for object tracking in your videos, or for a commercial C++ API to Darknet, [see DarkHelp](https://github.com/stephanecharette/DarkHelp).
* See if [the Darknet/YOLO FAQ](https://www.ccoderun.ca/programming/darknet_faq/) can help answer your questions.
* See the many tutorial and example videos on [Stéphane's YouTube channel](https://www.youtube.com/c/StephaneCharette/videos)
* If you have a support question or want to chat with other Darknet/YOLO users, [join the Darknet/YOLO discord server](https://discord.gg/zSq8rtW).

# Roadmap

Last updated 2023-07-23:

## Short-term goals

* convert Darknet to use the C++ compiler (g++ on Linux, VisualStudio on Windows)
* remove old solutions and Makefile
* make OpenCV non-optional
* remove STB
* re-write CMakeLists.txt to use the new CUDA detection and build out-of-source
* have better version number output

## Mid-term goals

* better use of `cv::Mat` instead of the custom `image` structure in C
* on-going code cleanup

## Long-term goals

* fix CUDA/CUDNN issues with all GPUs
* look into adding support for non-NVIDIA GPUs
