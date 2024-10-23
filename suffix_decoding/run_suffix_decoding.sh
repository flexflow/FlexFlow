#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

# make -j
source set_python_envs.sh

model_names=(
    meta-llama/Meta-Llama-3-70B-Instruct
    meta-llama/Meta-Llama-3-70B-Instruct
    meta-llama/Meta-Llama-3-8B-Instruct
    meta-llama/Meta-Llama-3-70B-Instruct
    meta-llama/CodeLlama-34b-Instruct-hf
    mistralai/Mistral-Large-Instruct-2407
    mistralai/Mistral-Large-Instruct-2407
)
small_model_names=(
    Felladrin/Llama-160M-Chat-v1
    Felladrin/Llama-160M-Chat-v1
    Felladrin/Llama-160M-Chat-v1
    Felladrin/Llama-160M-Chat-v1
    Felladrin/Llama-160M-Chat-v1 #meta-llama/Llama-3.2-1B-Instruct
    mistralai/Mistral-7B-Instruct-v0.2
    mistralai/Mistral-7B-Instruct-v0.2
)
partitions=(
    "QUESTION_SUGGESTION"
    "CATEGORIZATION"
    "FEATURE_EXTRACTION"
    "SQL_FANOUT1"
    "SQL_FANOUT2"
    "SQL_FANOUT3"
    "SQL_COMBINE"
)

export LEGION_BACKTRACE=1

for i in "${!partitions[@]}"; do
    model_name=${model_names[$i]}
    small_model_name=${small_model_names[$i]}
    partition_name=${partitions[$i]}
    echo "Running partition ${partition_name} with model ${model_name} and small model ${small_model_name}"
    python ../inference/utils/download_hf_model.py --half-precision-only $model_name $small_model_name
    
    rm /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}.out || true

    ./inference/suffix_decoding/specinfer \
        -ll:gpu 8 -ll:cpu 4 -ll:util 4 \
        -tensor-parallelism-degree 8 \
        -ll:fsize 70000 -ll:zsize 300000 -ll:csize 200000 \
        --fusion \
        --max-sequence-length 7000 \
        --max-requests-per-batch 16 \
        --max-tokens-per-batch 1200 \
        --max-output-length 900 \
        --expansion-degree 1 \
        -llm-model $model_name \
        -ssm-model $small_model_name \
        -trace /home/yak/goliaro/suffix-tree-decoding/trace/cortex.json \
        -trace-output-path /home/yak/goliaro/suffix-tree-decoding/trace/flexflow/cortex_ff_${partition_name}.json \
        -output-file /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}.out \
        -target-partition ${partition_name}
done