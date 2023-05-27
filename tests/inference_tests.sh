#! /usr/bin/env bash
set -x
set -e

cleanup() {
    rm -rf ../inference/prompt ../inference/weights ../inference/tokenizer
}

copy_embedding_weights(){
    cp ../inference/weights/opt_6B_weights/embed_tokens_weight ../inference/weights/opt_6B_weights/embed_tokens_weight_lm_head
    cp ../inference/weights/opt_125M_weights/embed_tokens_weight ../inference/weights/opt_125M_weights/embed_tokens_weight_lm_head
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Clean up before test (just in case)
cleanup

# Update the transformers library to support the LLAMA model
pip3 install --upgrade transformers

# Download the weights
python3 ../inference/utils/download_llama_weights.py
python3 ../inference/utils/download_opt_weights.py

# because huggingface reuse a weight in embedding and final linear
copy_embedding_weights

# Create test prompt file
mkdir -p ../inference/prompt
echo '["Give three tips for staying healthy."]' > ../inference/prompt/test.json

# Create output folder
mkdir -p ../inference/output

###############################################################################################
############################ Speculative inference tests ######################################
###############################################################################################

# LLAMA
../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -ssm-model llama -ssm-weight ../inference/weights/llama_160M_weights/ -ssm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_llama.txt

# OPT
../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -ssm-model opt -ssm-weight ../inference/weights/opt_125M_weights/ -ssm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_opt.txt

###############################################################################################
############################ Incremental decoding tests #######################################
###############################################################################################

# LLAMA (small model)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_160M_weights/ -llm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_160M.txt

# LLAMA (big model)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_7B.txt

# OPT (small model)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_125M_weights/ -llm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_opt_125M.txt

# OPT (big model)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json ../inference/output/incr_decoding_opt_6B.txt

# Clean up after test
cleanup
