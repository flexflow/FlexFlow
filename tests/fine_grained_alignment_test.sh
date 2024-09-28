#! /usr/bin/env bash
# set -x
set -e

MODEL_NAME=${MODEL_NAME:-"JackFram/llama-160m"}
MEMORY_PER_GPU=${MEMORY_PER_GPU:-14000}
ZCOPY_MEMORY=${ZCOPY_MEMORY:-40000}
CACHE_PATH=${FF_CACHE_PATH:-"~/.cache/flexflow"}

cleanup() {
    rm -rf ${CACHE_PATH}/debug ./fine_grained_alignment_config.json ./inference/output/fine_grained_alignment_test_ff.txt ./inference/output/fine_grained_alignment_test_hf.txt
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}/.."

# Initial cleanup
cleanup

# Create test prompt file
mkdir -p ./inference/prompt
echo '["Three tips for staying healthy are: "]' > ./inference/prompt/test.json

# Create output folder
mkdir -p ./inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

python ./tests/inference/huggingface_inference.py --model-name $MODEL_NAME --max-length 10 --prompt-file ../../inference/prompt/test.json --output-file ../../inference/output/fine_grained_alignment_test_hf.txt --use-full-precision --inference-debugging

json_config=$(cat <<-END
    {
        "num_gpus": 4,
        "memory_per_gpu": ${MEMORY_PER_GPU},
        "zero_copy_memory_per_node": ${ZCOPY_MEMORY},
        "num_cpus": 4,
        "legion_utility_processors": 4,
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": 2,
        "pipeline_parallelism_degree": 2,
        "inference_debugging": true,
        "fusion": true,
        "refresh_cache": false,
        "llm_model": "${MODEL_NAME}",
        "cache_path": "${CACHE_PATH}",
        "full_precision": true,
        "prompt": "./inference/prompt/test.json",
        "max_length": 10,
        "output_file": "./inference/output/fine_grained_alignment_test_ff.txt"
    }
END
)
echo $json_config > ./fine_grained_alignment_config.json

python ./inference/python/incr_decoding.py -config-file ./fine_grained_alignment_config.json

# # C++ test
# echo "C++ test"
# ./build/inference/incr_decoding/incr_decoding \
#     -ll:gpu 2 -ll:cpu 4 -ll:util 4 \
#     -tensor-parallelism-degree 2 \
#     -ll:fsize 8192 -ll:zsize 12000 \
#     -llm-model $MODEL_NAME \
#     -prompt ./inference/prompt/peft.json \
#     --use-full-precision \
#     --inference-debugging

# Check alignment
python ./tests/inference/inference_alignment_test.py -m $MODEL_NAME -tp 2 -n 2

# Print succeess message
echo ""
echo "Inference alignment tests passed!"
echo ""

# Cleanup after the test
cleanup
