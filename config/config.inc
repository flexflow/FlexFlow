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

#set installation dir
if [ -n "$INSTALL_DIR" ]; then
  SET_INSTALL_DIR="-DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}"
fi

# set build type
if [ -n "$BUILD_TYPE" ]; then
  SET_BUILD="-DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
fi

# set CUDA Arch
if [ -n "$FF_CUDA_ARCH" ]; then
  SET_CUDA_ARCH="-DFF_CUDA_ARCH=${FF_CUDA_ARCH}"
fi

# set CUDA dir
if [ -n "$CUDA_DIR" ]; then
  SET_CUDA="-DCUDA_PATH=${CUDA_DIR}"
fi

# set cudnn dir
if [ -n "$CUDNN_DIR" ]; then
  SET_CUDNN="-DCUDNN_PATH=${CUDNN_DIR}"
fi

# enable Python
if [ "$FF_USE_PYTHON" = "ON" ]; then
  SET_PYTHON="-DFF_USE_PYTHON=ON"
elif [ "$FF_USE_PYTHON" = "OFF" ]; then
  SET_PYTHON="-DFF_USE_PYTHON=OFF"
else
  SET_PYTHON="-DFF_USE_PYTHON=ON"
fi

# enable NCCL
if [ "$FF_USE_NCCL" = "ON" ]; then
  SET_NCCL="-DFF_USE_NCCL=ON"
elif [ "$FF_USE_NCCL" = "OFF" ]; then
  SET_NCCL="-DFF_USE_NCCL=OFF"
else
  SET_NCCL="-DFF_USE_NCCL=ON"
fi

# enable GASNET
if [ "$FF_USE_GASNET" = "ON" ]; then
  SET_GASNET="-DFF_USE_GASNET=ON"
elif [ "$FF_USE_GASNET" = "OFF" ]; then
  SET_GASNET="-DFF_USE_GASNET=OFF"
else
  SET_GASNET="-DFF_USE_GASNET=ON"
fi

# build C++ examples
if [ "$FF_BUILD_ALL_EXAMPLES" = "ON" ]; then
  SET_EXAMPLES="-DFF_BUILD_ALL_EXAMPLES=ON"
elif [ "$FF_BUILD_ALL_EXAMPLES" = "OFF" ]; then
  SET_EXAMPLES="-DFF_BUILD_ALL_EXAMPLES=OFF"
else
  SET_EXAMPLES="-DFF_BUILD_ALL_EXAMPLES=ON"
fi

# enable avx2
if [ "$FF_USE_AVX2" = "ON" ]; then
  SET_AVX2="-DFF_USE_AVX2=ON"
elif [ "$FF_USE_AVX2" = "OFF" ]; then
  SET_AVX2="-DFF_USE_AVX2=OFF"
else
  SET_AVX2="-DFF_USE_AVX2=OFF"
fi

#set max dims
if [ -n "$FF_MAX_DIM" ]; then
  SET_MAX_DIM="-DFF_MAX_DIM=${FF_MAX_DIM}"
fi

SRC_LOCATION=${SRC_LOCATION:=`dirname $0`/../}
CMAKE_COMMAND="${SET_CC_FLAGS} ${SET_NVCC_FLAGS} ${SET_LD_FLAGS} cmake -DCUDA_USE_STATIC_CUDA_RUNTIME=OFF ${SET_CC} ${SET_CXX} ${SET_INSTALL_DIR} ${SET_BUILD} ${SET_CUDA_ARCH} ${SET_CUDA} ${SET_CUDNN} ${SET_PYTHON} ${SET_PYBIND11} ${SET_NCCL} ${SET_GASNET} ${SET_EXAMPLES} ${SET_AVX2} ${SET_MAX_DIM} $* ${SRC_LOCATION}"
echo $CMAKE_COMMAND
eval $CMAKE_COMMAND
}
