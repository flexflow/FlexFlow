#!/usr/bin/env bash

set -euo pipefail

# These modules are specific to Sapling, if deploying to another machine
# customize as necessary.
if [[ ! -v CI ]]; then
    module load cuda cmake
fi

err_out() {
    >&2 echo "$@"
    exit 1
}

while [[ $# -gt 0 ]]; do
  case $1 in
    --jobs|-j)
      THREADS="$2"
      shift
      shift
      ;;
    -*)
      err_out "Unknown option: $1"
      ;;
    *)
      err_out "Unknown argument: $1"
      ;;
  esac
done

: "${CC:=gcc-10}"
: "${CXX:=g++-10}"
: "${CUDAARCHS:=60}"
: "${THREADS:=$(nproc)}"
: "${CMAKE_PREFIX_PATH:=}"

export CC
export CXX
export CUDAARCHS
export THREADS

mkdir -p deploy/deps
pushd deploy/deps

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        err_out "Could not find command: $cmd"
    fi
}

require_cmd "$CC"
require_cmd "$CXX"
require_cmd cmake
require_cmd curl
require_cmd git
require_cmd make
require_cmd nvcc
require_cmd python3
require_cmd tar

function build_cmake_library {
    dep_name="$1"
    dep_url="$2"
    dep_args=("${@:3}")
    if [[ ! -e ${dep_name} ]]; then
        if [[ ${dep_url} == *.git ]]; then
            git clone --depth 1 --single-branch "${dep_url}" "${dep_name}"
        else
            mkdir "${dep_name}"
            curl -LsSf -o "${dep_name}.tar.gz" "${dep_url}"
            tar xfz "${dep_name}.tar.gz" -C "${dep_name}" --strip-components=1
        fi
    fi
    if [[ ! -e "${dep_name}"_install/include ]]; then
        mkdir -p "${dep_name}"_build "${dep_name}"_install
        pushd "${dep_name}"_build
        cmake ../"${dep_name}" -DCMAKE_INSTALL_PREFIX="$PWD"/../"${dep_name}"_install "${dep_args[@]}"
        make install -j"$THREADS"
        popd
        rm -rf "${dep_name}"_build
    fi
    export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:$PWD/${dep_name}_install"
}

if [[ ! -e gasnet ]]; then
    git clone --depth 1 --single-branch https://github.com/StanfordLegion/gasnet.git
fi
if [[ ! -e gasnet/release ]]; then
    make -C gasnet CONDUIT=ibv
fi
export GASNet_ROOT="$PWD"/gasnet/release

set -x

build_cmake_library zstd https://github.com/facebook/zstd.git -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library fmt https://github.com/fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library cpptrace https://github.com/jeremy-rifkin/cpptrace/archive/refs/tags/v1.0.4.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DCPPTRACE_USE_EXTERNAL_ZSTD=ON

build_cmake_library libassert https://github.com/jeremy-rifkin/libassert/archive/refs/tags/v2.2.1.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLIBASSERT_USE_EXTERNAL_CPPTRACE=ON

build_cmake_library Realm https://github.com/StanfordLegion/realm.git -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_SHARED_LIBS=ON -DREALM_ENABLE_GASNETEX=ON -DREALM_ENABLE_CUDA=ON -DREALM_ENABLE_PREALM=ON -DREALM_ENABLE_CPPTRACE=ON -DREALM_ENABLE_HDF5=OFF -DREALM_MAX_DIM=5

build_cmake_library benchmark https://github.com/google/benchmark/archive/refs/tags/v1.9.5.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DBENCHMARK_DOWNLOAD_DEPENDENCIES=ON

build_cmake_library rapidcheck https://github.com/emil-e/rapidcheck.git -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library tl-expected https://github.com/TartanLlama/expected/archive/refs/tags/v1.3.1.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library doctest https://github.com/doctest/doctest/archive/refs/tags/v2.4.12.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library spdlog https://github.com/gabime/spdlog/archive/refs/tags/v1.17.0.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSPDLOG_FMT_EXTERNAL=ON

build_cmake_library nlohmann_json https://github.com/nlohmann/json/archive/refs/tags/v3.12.0.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

build_cmake_library NCCL https://github.com/NVIDIA/nccl/archive/refs/tags/v2.29.7-1.tar.gz -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON

mkdir -p proj/
if [[ ! -e proj/venv ]]; then
    python3 -m venv proj/venv
fi

# shellcheck disable=SC1091
source proj/venv/bin/activate

if ! command -v proj >/dev/null 2>&1
then
    pip install --require-virtualenv 'git+https://git.sr.ht/~lockshaw/proj'
fi

popd # deploy/deps

ff_cmake_flags=(
    -DCMAKE_BUILD_TYPE=RelWithDebInfo
    -DCMAKE_INSTALL_PREFIX="$PWD/../install"
)

proj dtgen

mkdir build install
pushd build
cmake .. "${ff_cmake_flags[@]}"
make -j"$THREADS"
popd # build
