#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Token to access private huggingface models (e.g. LLAMA-2)
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-none}
if [[ "$HUGGINGFACE_TOKEN" != "none" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi

# Create test prompt file
mkdir -p ../inference/prompt
echo '["Two things are infinite: "]' > ../inference/prompt/peft.json

# Create output folder
mkdir -p ../inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

# Download test model
python ../inference/utils/download_peft_model.py goliaro/llama-160m-lora-full --base_model_name JackFram/llama-160m 
# if first time, add: --refresh-cache

./inference/peft/peft -ll:gpu 1 -ll:cpu 4 -ll:fsize 8192 -ll:zsize 12000 -ll:util 4 -llm-model JackFram/llama-160m -prompt ../inference/prompt/peft.json -peft-model goliaro/llama-160m-lora-full --use-full-precision --inference-debugging --fusion -enable-peft
