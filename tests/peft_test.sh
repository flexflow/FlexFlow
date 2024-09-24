#! /usr/bin/env bash
# set -x
set -e

cleanup() {
    rm -rf ~/.cache/flexflow/debug
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/.."

# Token to access private huggingface models (e.g. LLAMA-2)
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-none}
if [[ "$HUGGINGFACE_TOKEN" != "none" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi

# Clean up before test (just in case)
cleanup

# Create test prompt file
mkdir -p ./inference/prompt
echo '["Two things are infinite: "]' > ./inference/prompt/peft.json
echo '["“Two things are infinite: the universe and human stupidity; and I'\''m not sure about the universe.”"]' > ./inference/prompt/peft_dataset.json


# Create output folder
mkdir -p ./inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

# Download test model
python ./inference/utils/download_peft_model.py goliaro/llama-160m-lora --base_model_name JackFram/llama-160m 

# Run PEFT in Huggingface to get ground truth tensors
python ./tests/peft/hf_finetune.py --peft-model-id goliaro/llama-160m-lora --save-peft-tensors --use-full-precision

# Python test
echo "Python test"
python ./inference/python/ff_peft.py
# Check alignment
python ./tests/peft/peft_alignment_test.py -tp 2

# C++ test
echo "C++ test"
./build/inference/peft/peft \
    -ll:gpu 2 -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree 2 \
    -ll:fsize 8192 -ll:zsize 12000 \
    -llm-model JackFram/llama-160m \
    -finetuning-dataset ./inference/prompt/peft_dataset.json \
    -peft-model goliaro/llama-160m-lora \
    -enable-peft \
    --use-full-precision \
    --inference-debugging
# Check alignment
python ./tests/peft/peft_alignment_test.py -tp 2

# Print succeess message
echo ""
echo "PEFT tests passed!"
echo ""

# Cleanup after the test
cleanup
