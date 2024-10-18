#! /usr/bin/env bash
set -e
set -x

# Cd into parent directory of folder holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

# Download models
python ../inference/utils/download_hf_model.py meta-llama/Meta-Llama-3-8B-Instruct Felladrin/Llama-160M-Chat-v1
# export RUST_BACKTRACE=1

# gdb -ex run -ex bt --args ./inference/suffix_decoding/suffix_decoding \
#     -ll:gpu 4 -ll:cpu 4 -ll:util 4 \
#     -tensor-parallelism-degree 4 \
#     -ll:fsize 20000 -ll:zsize 30000 \
#     -llm-model meta-llama/Meta-Llama-3-8B \
#     -ssm-model Felladrin/Llama-160M-Chat-v1 \
#     -partition-name "" \
#     -prompt ../../suffix-tree-decoding/trace/spider_v2.json \
#     -output-file ../inference/output/spider_v2.out

./inference/incr_decoding/incr_decoding \
    -ll:gpu 4 -ll:cpu 4 -ll:util 4 \
    -tensor-parallelism-degree 4 \
    -ll:fsize 20000 -ll:zsize 60000 \
    --max-sequence-length 1200 \
    --max-requests-per-batch 1 \
    --max-tokens-per-batch 256 \
    -llm-model meta-llama/Meta-Llama-3-8B-Instruct \
    -prompt /home/yak/goliaro/suffix-tree-decoding/trace/cortex_v2.json \
    -output-file /home/yak/goliaro/FlexFlow/suffix_decoding/test.out \
    -target-partition FEATURE_EXTRACTION \
    -output-trace-path /home/yak/goliaro/suffix-tree-decoding/trace/flexflow/cortex_ff_FEATURE_EXTRACTION.json
