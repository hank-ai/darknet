#!/bin/bash -e

mkdir -p build
cd build

# Pick just 1 of the following -- either Release or Debug
set BUILD_TYPE=Release
#set BUILD_TYPE=Debug

cmake -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_CXX_COMPILER=g++-13 -DCMAKE_C_COMPILER=gcc-13 ..
make -j $(nproc)
make package

echo Done!
echo Make sure you install the .deb file:
ls -lh *.deb

echo Installing the .deb file...
sudo dpkg -i *.deb

cd ..
