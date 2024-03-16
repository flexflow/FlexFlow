#! /usr/bin/env bash

set -euo pipefail
set -x

DIR="$(realpath -- "$(dirname "${BASH_SOURCE[0]}")")"
REPO="$(realpath -- "$DIR/../../../")"

export FF_GPU_BACKEND="cuda"
export FF_CUDA_ARCH=70
n_build_cores=$(($(nproc) - 1))
if (( $n_build_cores < 1 )) ; then n_build_cores=1 ; fi
cd "$REPO"
mkdir build
cd build
#if [[ "${FF_GPU_BACKEND}" == "cuda" ]]; then
#  export FF_BUILD_ALL_EXAMPLES=ON
#  export FF_BUILD_UNIT_TESTS=ON
#fi
../config/config.linux \
        -DCMAKE_CXX_COMPILER="clang++" \
        -DCMAKE_C_COMPILER="clang" \
	-DCMAKE_C_COMPILER_LAUNCHER=ccache \
	-DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
	-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        -DFF_USE_EXTERNAL_LEGION=ON \
        -DFF_USE_EXTERNAL_JSON=ON \
        -DFF_USE_EXTERNAL_FMT=ON \
        -DFF_USE_EXTERNAL_SPDLOG=ON
