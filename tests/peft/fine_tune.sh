#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

python hf_finetune.py --model-name decapoda-research/llama-7b-hf --lora-target-modules down_proj --use-full-precision --publish-peft-with-id goliaro/llama-7b-lora-full
python hf_finetune.py --model-name decapoda-research/llama-7b-hf --lora-target-modules down_proj --publish-peft-with-id goliaro/llama-7b-lora-half
python hf_finetune.py --model-name JackFram/llama-160m-base --lora-target-modules down_proj --use-full-precision --publish-peft-with-id goliaro/llama-160m-lora-full
python hf_finetune.py --model-name JackFram/llama-160m-base --lora-target-modules down_proj --publish-peft-with-id goliaro/llama-160m-lora-half

python hf_finetune.py --model-name meta-llama/Llama-2-7b-hf --lora-target-modules down_proj --use-full-precision --publish-peft-with-id goliaro/llama-2-7b-lora-full
python hf_finetune.py --model-name meta-llama/Llama-2-7b-hf --lora-target-modules down_proj --publish-peft-with-id goliaro/llama-2-7b-lora-half

python hf_finetune.py --model-name facebook/opt-6.7b --lora-target-modules fc2 --use-full-precision --publish-peft-with-id goliaro/opt-6.7b-lora-full
python hf_finetune.py --model-name facebook/opt-6.7b --lora-target-modules fc2 --publish-peft-with-id goliaro/opt-6.7b-lora-half
python hf_finetune.py --model-name facebook/opt-125m --lora-target-modules fc2 --use-full-precision --publish-peft-with-id goliaro/opt-125m-lora-full
python hf_finetune.py --model-name facebook/opt-125m --lora-target-modules fc2 --publish-peft-with-id goliaro/opt-125m-lora-half
