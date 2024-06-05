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
echo '["San Francisco, officially the City and County of San Francisco, is a "]' > ../inference/prompt/test_upload.json

# Create output folder
mkdir -p ../inference/output
mkdir -p ../inference/configs

# Enable backtrace in case we run into a segfault or assertion failure
export LEGION_BACKTRACE=1

# Create config files
cat > ../inference/configs/llama_small.json <<EOF
{
    "num_gpus": 1,
    "memory_per_gpu": 8000,
    "zero_copy_memory_per_node": 20000,
    "num_cpus": 4,
    "legion_utility_processors": 4,
    "offload": false,
    "fusion": true,
    "llm_model": "JackFram/llama-160m",
    "full_precision": false,
    "prompt": "../../inference/prompt/test_upload.json",
    "output_file": "../../inference/output/original_llama_small.txt"
}
EOF
cat > ../inference/configs/llama_small_upload.json <<EOF
{
    "num_gpus": 1,
    "memory_per_gpu": 8000,
    "zero_copy_memory_per_node": 20000,
    "num_cpus": 4,
    "legion_utility_processors": 4,
    "offload": false,
    "fusion": true,
    "llm_model": "goliaro/test-llama",
    "full_precision": false,
    "prompt": "../../inference/prompt/test_upload.json",
    "output_file": "../../inference/output/upload_llama_small.txt"
}
EOF
python ../inference/python/incr_decoding.py -config-file ../inference/configs/llama_small.json
python -c "from huggingface_hub import HfApi; api = HfApi(); api.delete_repo('goliaro/test-llama')" || true
python ../inference/utils/upload_hf_model.py JackFram/llama-160m --new-model-id goliaro/test-llama
python ../inference/python/incr_decoding.py -config-file ../inference/configs/llama_small_upload.json
diff <(tail -n +3 "../../inference/output/original_llama_small.txt") <(tail -n +3 "../../inference/output/upload_llama_small.txt")
