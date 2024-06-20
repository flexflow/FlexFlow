#! /usr/bin/env bash
set -x
set -e

NB_PROMPTS=10

cleanup() {
    rm -rf ../inference/prompt ../inference/output
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

HUGGINGFACE_TOKEN="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"

# Token to access private huggingface models (e.g. LLAMA-2)
HUGGINGFACE_TOKEN=${HUGGINGFACE_TOKEN:-none}
if [[ "$HUGGINGFACE_TOKEN" != "none" ]]; then
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
fi

# Clean up before test (just in case)
cleanup

# Make sure supported version of protobuf is installed
pip3 install protobuf==3.20.3

# Create test prompt file
mkdir -p ../inference/prompt
#echo '["Three tips for staying healthy are: ", "The three best tennis players of all time are: "]' > ../inference/prompt/test.json
cp test_data_$NB_PROMPTS.json ../inference/prompt

# Create output folder
mkdir -p ../inference/output

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

# Manually download the weights in both half and full precision
echo "Downloading models..."
python3 ../inference/utils/download_hf_model.py "meta-llama/Llama-2-7b-hf" "JackFram/llama-160m" #--refresh-cache #"facebook/opt-6.7b" "facebook/opt-125m" "tiiuae/falcon-7b"

# Running C++ inference tests
echo "Running C++ inference tests..."

if [[ "$1" == "incr_decoding" ]]
then
# ========== INCR DECODING ==========
../build/inference/incr_decoding/incr_decoding --max-requests-per-batch $2 -ll:gpu 4 -ll:cpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model meta-llama/Llama-2-7b-hf -prompt ../inference/prompt/test_data_$NB_PROMPTS.json -output-file ../inference/output/incr_decoding.txt -pipeline-parallelism-degree 4

else
# =========== SPEC INFER ============
../build/inference/spec_infer/spec_infer --max-requests-per-batch $2 -ll:gpu 4 -ll:cpu 4 -ll:fsize 14000 -ll:zsize 30000 -ll:csize 30000 -lg:eager_alloc_percentage 20 --fusion -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-160m -prompt ../inference/prompt/test_data_$NB_PROMPTS.json -output-file ../inference/output/spec_infer.txt -pipeline-parallelism-degree 4
fi