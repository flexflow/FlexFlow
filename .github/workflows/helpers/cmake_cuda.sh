#! /usr/bin/env bash

set -euo pipefail
set -x

DIR="$(realpath -- "$(dirname "${BASH_SOURCE[0]}")")"
REPO="$(realpath -- "$DIR/../../../")"

export FF_GPU_BACKEND="cuda"
export FF_CUDA_ARCH=70

if [[ -d "$REPO/build-ci" ]]; then 
  rm -rf "$REPO/build-ci"
fi
mkdir "$REPO/build-ci"
cd "$REPO/build-ci"
#if [[ "${FF_GPU_BACKEND}" == "cuda" ]]; then
#  export FF_BUILD_ALL_EXAMPLES=ON
#  export FF_BUILD_UNIT_TESTS=ON
#fi
# append the cmake flag FF_USE_CODE_COVERAGE
IFS=" " read -r -a FLAGS <<< "$CMAKE_FLAGS"
echo "FLAGS: ${FLAGS[@]}"
FLAGS+=("-DFF_USE_CODE_COVERAGE=ON")
../config/config.linux \
        -DCMAKE_C_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
        -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache \
        "${FLAGS[@]}"

# vim: set tabstop=2 shiftwidth=2 expandtab:
