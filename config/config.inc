#!/bin/bash

function run_cmake() {
# set CC and CXX
if [ -n "$CC" ]; then
  SET_CC="-DCMAKE_C_COMPILER=${CC}"
fi

if [ -n "$CXX" ]; then
  SET_CXX="-DCMAKE_CXX_COMPILER=${CXX}"
fi

# set FLAGS
if [ -n "$CC_FLAGS" ]; then
  SET_CC_FLAGS="CC_FLAGS=${CC_FLAGS}"
fi

if [ -n "$CC_FLAGS" ]; then
  SET_NVCC_FLAGS="NVCC_FLAGS=${NVCC_FLAGS}"
fi

if [ -n "$LD_FLAGS" ]; then
  SET_LD_FLAGS="LD_FLAGS=${LD_FLAGS}"
fi

# set build type
if [ -n "$BUILD_TYPE" ]; then
  SET_BUILD="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
fi

# set CUDA Arch
if [ -n "CUDA_ARCH" ]; then
  SET_CUDA_ARCH="-DCUDA_ARCH=${CUDA_ARCH}"
fi

# enable Python
if [ -n "$USE_PYTHON" ]; then
  SET_PYTHON="-DENABLE_PYTHON=ON"
fi

# set cudnn dir
if [ -n "$CUDNN_DIR" ]; then
  SET_CUDNN="-DCUDNN_ROOT=${CUDNN_DIR}"
fi

# enable GASNET
if [ -n "$USE_GASNET" ]; then
  SET_GASNET="-DENABLE_GASNET=ON"
fi

# set Legion fir
if [ -n "$LEGION_DIR" ]; then
  SET_LEGION="-DLEGION_ROOT=${LEGION_DIR}"
fi

# set Protobuf fir
if [ -n "$PROTOBUF_DIR" ]; then
  SET_PROTOBUF="-DPROTOBUF_ROOT=${PROTOBUF_DIR}"
fi

SRC_LOCATION=${SRC_LOCATION:=`dirname $0`/../}
CMAKE_COMMAND="${SET_CC_FLAGS} ${SET_NVCC_FLAGS} ${SET_LD_FLAGS} cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ${SET_CC} ${SET_CXX} ${SET_BUILD} ${SET_CUDA_ARCH} ${SET_PYTHON} ${SET_CUDNN} ${SET_GASNET} ${SET_LEGION} ${SET_PROTOBUF} $* ${SRC_LOCATION}"
echo $CMAKE_COMMAND
eval $CMAKE_COMMAND
}