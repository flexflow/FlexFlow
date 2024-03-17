#! /usr/bin/env bash

set -euo pipefail
set -x

DIR="$(realpath -- "$(dirname "${BASH_SOURCE[0]}")")"
REPO="$(realpath -- "$DIR/../../../")"

export FF_GPU_BACKEND="cuda"
export FF_CUDA_ARCH=70
cd "$REPO"
mkdir build
cd build
#if [[ "${FF_GPU_BACKEND}" == "cuda" ]]; then
#  export FF_BUILD_ALL_EXAMPLES=ON
#  export FF_BUILD_UNIT_TESTS=ON
#fi
../config/config.linux \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        $FF_CMAKE_FLAGS

# vim: set tabstop=2 shiftwidth=2 expandtab:
