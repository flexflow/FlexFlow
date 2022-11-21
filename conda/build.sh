#! /usr/bin/env bash
set -euo pipefail

# Cd into FF_HOME
cd "${BASH_SOURCE[0]%/*}/../"

# build flexflow
# "search and replace" bash syntax used below to make shellcheck happy.
# see here: https://wiki-dev.bash-hackers.org/syntax/pe
CXXFLAGS="${CXXFLAGS//-O2/}"
CXXFLAGS="${CXXFLAGS//-std=c++17/}"
CXXFLAGS="${CXXFLAGS//-DNDEBUG/}"
CXXFLAGS="${CXXFLAGS//-D_FORTIFY_SOURCE=2/}"
export CXXFLAGS
CPPFLAGS="${CPPFLAGS//-O2/}"
CPPFLAGS="${CPPFLAGS//-std=c++17/}"
CPPFLAGS="${CPPFLAGS//-DNDEBUG/}"
CPPFLAGS="${CPPFLAGS//-D_FORTIFY_SOURCE=2/}"
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
