#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/../build"

make -j
source set_python_envs.sh

model_names=(
    meta-llama/Meta-Llama-3-70B-Instruct
)
small_model_names=(
    Felladrin/Llama-160M-Chat-v1
    meta-llama/Meta-Llama-3-8B-Instruct
)
partitions=(
    "SQL_FANOUT1"
)
batch_sizes=(
    8
    16
    32
)
tokens_per_batchs=(
    750
    1024
)

# download all models and small models
# python ../inference/utils/download_hf_model.py --half-precision-only  ${model_names[@]} ${small_model_names[@]}

export LEGION_BACKTRACE=1

for i in "${!partitions[@]}"; do
    partition_name=${partitions[$i]}
    rm /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}.csv || true
    for j in "${!model_names[@]}"; do
        for k in "${!small_model_names[@]}"; do
            for l in "${!batch_sizes[@]}"; do
                for m in "${!tokens_per_batchs[@]}"; do
                    model_name=${model_names[$j]}
                    small_model_name=${small_model_names[$k]}
                    batch_size=${batch_sizes[$l]}
                    tokens_per_batch=${tokens_per_batchs[$m]}
                    
                    echo "Running partition ${partition_name} with model ${model_name}, small model ${small_model_name}, batch size ${batch_size}, and tokens per batch ${tokens_per_batch}"
                    # create model name version where "/" is replaced with "-"
                    model_name_=$(echo $model_name | tr / -)
                    small_model_name_=$(echo $small_model_name | tr / -)
                    rm /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}_${model_name_}_${small_model_name_}_${batch_size}_${tokens_per_batch}.out || true

                    ./inference/suffix_decoding/specinfer \
                        -ll:gpu 8 -ll:cpu 4 -ll:util 4 \
                        -tensor-parallelism-degree 8 \
                        -ll:fsize 70000 -ll:zsize 300000 -ll:csize 200000 \
                        --fusion \
                        --max-sequence-length 7000 \
                        --max-requests-per-batch $batch_size \
                        --max-tokens-per-batch $tokens_per_batch \
                        --max-output-length 900 \
                        --max-tree-depth 4 \
                        --expansion-degree 1 \
                        -llm-model $model_name \
                        -ssm-model $small_model_name \
                        -trace /home/yak/goliaro/suffix-tree-decoding/trace/cortex.json \
                        -trace-output-path /home/yak/goliaro/suffix-tree-decoding/trace/flexflow/cortex_ff_${partition_name}_${model_name_}_${small_model_name_}_${batch_size}_${tokens_per_batch}.json \
                        -output-file /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}_${model_name_}_${small_model_name_}_${batch_size}_${tokens_per_batch}.out \
                        -csv-output-path /home/yak/goliaro/FlexFlow/inference/output/cortex_${partition_name}.csv \
                        -target-partition ${partition_name}
                done
            done
        done
    done
done