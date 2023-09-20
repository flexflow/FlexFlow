#! /usr/bin/env bash
set -x
set -e

cleanup() {
    rm -rf ../inference/prompt ../inference/output
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Enable Python tests (on by default)
PYTHON_INFERENCE_TESTS=${PYTHON_INFERENCE_TESTS:-ON}
# Enable C++ tests, (off by default)
CPP_INFERENCE_TESTS=${CPP_INFERENCE_TESTS:-OFF}
# Enable model parallelism tests in C++, if desired
TENSOR_PARALLELISM_TESTS=${TENSOR_PARALLELISM_TESTS:-OFF}

# Clean up before test (just in case)
cleanup

# Make sure supported version of protobuf is installed
pip3 install protobuf==3.20.3

# Create test prompt file
mkdir -p ../inference/prompt
echo '["Give three tips for staying healthy."]' > ../inference/prompt/test.json

# Create output folder
mkdir -p ../inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

if [[ "$PYTHON_INFERENCE_TESTS" == "ON" ]]; then
    echo "Running Python inference tests..."
    ./inference/python_inference_tests.sh
fi
if [[ "$CPP_INFERENCE_TESTS" == "ON" ]]; then
    # Manually download the weights in both half and full precision
    python3 ../inference/utils/download_hf_model.py "decapoda-research/llama-7b-hf" "JackFram/llama-160m-base" "facebook/opt-6.7b" "facebook/opt-125m" "tiiuae/falcon-7b"
    echo "Running C++ inference tests..."
    ./inference/cpp_inference_tests.sh
fi

