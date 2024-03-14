#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"
cd ../deps/tokenizers-cpp/example
cmake -D CMAKE_CXX_FLAGS=-fPIC
make -j
