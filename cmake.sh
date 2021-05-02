#! /usr/bin/env bash

FF_USE_NCCL=ON

export CC="$(which gcc)" CXX="$(which g++)" NVCC="$(which nvcc)"

DIRNAME="$(basename "$PWD")"
if [[ $DIRNAME == "build" ]]; then
  BUILD_TYPE="Release"
elif [[ $DIRNAME == "debug" ]]; then
  BUILD_TYPE="Debug"
elif [[ $DIRNAME == "rwdebug" ]]; then
  BUILD_TYPE="RelWithDebInfo"
else
  >&2 echo "ERROR: Invalid build directory: $DIRNAME"
  exit 1
fi

echo "Using build type: $BUILD_TYPE"

# rm CMakeCache.txt
cmake -DFF_BUILD_TESTS=OFF \
      -DFF_USE_GASNET=OFF \
      -DFF_USE_NCCL=ON \
      -DFF_CUDA_ARCH=60 \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DFF_BUILD_MLP_UNIFY=ON \
      -DFF_BUILD_RESNET=ON \
      -DFF_BUILD_RESNEXT=ON \
      -DFF_USE_PYTHON=ON \
      ..
