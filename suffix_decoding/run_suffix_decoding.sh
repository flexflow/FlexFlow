#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

make -j
source set_python_envs.sh

# python ../inference/utils/download_hf_model.py meta-llama/Llama-2-7b-hf JackFram/llama-160m meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Meta-Llama-3-8B
# python ../inference/utils/download_hf_model.py meta-llama/Meta-Llama-3-8B-Instruct Felladrin/Llama-160M-Chat-v1

export LEGION_BACKTRACE=1

rm /home/yak/goliaro/FlexFlow/inference/output/cortex.out || true
rm /home/yak/goliaro/suffix-tree-decoding/trace/flexflow/cortex_ff_FEATURE_EXTRACTION3.json || true
./inference/suffix_decoding/specinfer \
    -ll:gpu 1 -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree 1 \
    -ll:fsize 70000 -ll:zsize 90000 -ll:csize 200000 \
    --fusion \
    --max-sequence-length 1200 \
    --max-requests-per-batch 16 \
    --max-tokens-per-batch 1200 \
    --expansion-degree 1 \
    -llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    -ssm-model Felladrin/Llama-160M-Chat-v1 \
    -trace /home/yak/goliaro/suffix-tree-decoding/trace/cortex.json \
    -trace-output-path /home/yak/goliaro/suffix-tree-decoding/trace/flexflow/cortex_ff_FEATURE_EXTRACTION3.json \
    -output-file /home/yak/goliaro/FlexFlow/inference/output/cortex.out \
    -target-partition FEATURE_EXTRACTION
