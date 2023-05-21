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
python3 ../inference/utils/download_llama_weights.py
python3 ../inference/utils/download_opt_weights.py

# Create test prompt file
mkdir -p ../inference/prompt
echo '["Give three tips for staying healthy."]' > ../inference/prompt/test.json

###############################################################################################
############################ Speculative inference tests ######################################
###############################################################################################

# LLAMA
../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -ssm-model llama -ssm-weight ../inference/weights/llama_190M_weights/ -ssm-config ../inference/models/configs/llama_190M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json

# OPT
#../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -ssm-model opt -ssm-weight ../inference/weights/opt_125M_weights/ -ssm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json

###############################################################################################
############################ Incremental decoding tests #######################################
###############################################################################################

# LLAMA
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json

# OPT
#../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json

# Clean up after test
cleanup
