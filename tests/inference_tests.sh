#! /usr/bin/env bash
set -x
set -e

cleanup() {
    rm -rf ../inference/prompt ../inference/weights ../inference/tokenizer
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Clean up before test (just in case)
cleanup

# Update the transformers library to support the LLAMA model
pip3 install --upgrade transformers

# Download the weights
python3 ../inference/utils/download_weights.py

# Create test prompt file
mkdir -p ../inference/prompt
echo '["Give three tips for staying healthy."]' > ../inference/prompt/test.json

# Run test
../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-weight ../inference/weights/llama_7B_weights/ -ssm-weight ../inference/weights/llama_190M_weights/ -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -ssm-config ../inference/models/configs/llama_190M.json -llm-config ../inference/models/configs/llama_7B.json


# Clean up after test
cleanup
