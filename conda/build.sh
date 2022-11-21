#! /usr/bin/env bash

# build flexflow
CXXFLAGS=$(echo $CXXFLAGS | sed 's/-O2//g')
CXXFLAGS=$(echo $CXXFLAGS | sed 's/-std=c++17//g')
CXXFLAGS=$(echo $CXXFLAGS | sed 's/-DNDEBUG//g')
CXXFLAGS=$(echo $CXXFLAGS | sed 's/-D_FORTIFY_SOURCE=2//g')
export CXXFLAGS
CPPFLAGS=$(echo $CPPFLAGS | sed 's/-O2//g')
CPPFLAGS=$(echo $CPPFLAGS | sed 's/std=c++17//g')
CPPFLAGS=$(echo $CPPFLAGS | sed 's/-DNDEBUG//g')
CPPFLAGS=$(echo $CPPFLAGS | sed 's/-D_FORTIFY_SOURCE=2//g')
export CPPFLAGS

#export CUDNN_HOME=/projects/opt/centos7/cuda/10.1
#export CUDA_HOME=/projects/opt/centos7/cuda/10.1
export PROTOBUF_DIR=$BUILD_PREFIX
export FF_HOME=$SRC_DIR
export LG_RT_DIR=$SRC_DIR/legion/runtime
#export FF_ENABLE_DEBUG=1
#export DEBUG=0

cd python
make
