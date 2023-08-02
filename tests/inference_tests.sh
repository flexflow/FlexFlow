#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# replace this with python tests
./inference/cpp_inference_tests.sh
