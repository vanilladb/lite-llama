#!/usr/bin/sh
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=~/Developer/libtorch ..
cmake --build . --config Release

./llama