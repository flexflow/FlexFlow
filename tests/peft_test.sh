#! /usr/bin/env bash
# set -x
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
echo '["“Two things are infinite: the universe and human stupidity; and I'\''m not sure about the universe.”"]' > ../inference/prompt/peft_dataset.json


# Create output folder
mkdir -p ../inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

# Download test model
python ../inference/utils/download_peft_model.py goliaro/llama-160m-lora --base_model_name JackFram/llama-160m 
# if first time, add: --refresh-cache

# # CPP test
# ../build/inference/peft/peft \
#     -ll:gpu 4 -ll:cpu 4 -ll:util 4 \
#     -tensor-parallelism-degree 4 \
#     -ll:fsize 8192 -ll:zsize 12000 \
#     -llm-model JackFram/llama-160m \
#     -finetuning-dataset ../inference/prompt/peft_dataset.json \
#     -peft-model goliaro/llama-160m-lora \
#     --use-full-precision \
#     --fusion \
#     -enable-peft

# # Python test
# python ../inference/python/ff_peft.py

cd ../build
./inference/peft/peft \
    -ll:gpu 1 -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree 1 \
    -ll:fsize 8192 -ll:zsize 12000 \
    -llm-model JackFram/llama-160m \
    -finetuning-dataset ../inference/prompt/peft_dataset.json \
    -peft-model goliaro/llama-160m-lora \
    -enable-peft \
    --use-full-precision \
    --inference-debugging

cd ../tests/peft
python hf_finetune.py --peft-model-id goliaro/llama-160m-lora --save-peft-tensors --use-full-precision

python peft_alignment_test.py

# Print succeess message
echo ""
echo "PEFT tests passed!"
echo ""
