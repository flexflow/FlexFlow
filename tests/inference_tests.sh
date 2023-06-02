#! /usr/bin/env bash
set -x
set -e

cleanup() {
    rm -rf ../inference/prompt ../inference/weights ../inference/tokenizer
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Clean up before test (just in case)
cleanup

# Update the transformers library to support the LLAMA model
pip3 install --upgrade transformers

# Download the weights in both half and full precision
python3 ../inference/utils/download_llama_weights.py
python3 ../inference/utils/download_llama_weights.py --use-full-precision
python3 ../inference/utils/download_opt_weights.py
python3 ../inference/utils/download_opt_weights.py --use-full-precision

# Create test prompt file
mkdir -p ../inference/prompt
echo '["Give three tips for staying healthy."]' > ../inference/prompt/test.json

# Create output folder
mkdir -p ../inference/output

###############################################################################################
############################ Speculative inference tests ######################################
###############################################################################################

# LLAMA
../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -ssm-model llama -ssm-weight ../inference/weights/llama_160M_weights/ -ssm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_llama.txt
# LLAMA (half precision)
../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights_half/ -llm-config ../inference/models/configs/llama_7B.json -ssm-model llama -ssm-weight ../inference/weights/llama_160M_weights_half/ -ssm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_llama_half.txt

# OPT
../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -ssm-model opt -ssm-weight ../inference/weights/opt_125M_weights/ -ssm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_opt.txt
# OPT (half precision)
../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights_half/ -llm-config ../inference/models/configs/opt_6B.json -ssm-model opt -ssm-weight ../inference/weights/opt_125M_weights_half/ -ssm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_opt_half.txt

###############################################################################################
############################ Incremental decoding tests #######################################
###############################################################################################

# LLAMA (small model)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model llama -llm-weight ../inference/weights/llama_160M_weights/ -llm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_160M.txt
# LLAMA (small model, half precision)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_160M_weights_half/ -llm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_160M_half.txt

# LLAMA (big model)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_7B.txt
# LLAMA (big model, half precision)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights_half/ -llm-config ../inference/models/configs/llama_7B.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_7B_half.txt

# OPT (small model)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model opt -llm-weight ../inference/weights/opt_125M_weights/ -llm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_opt_125M.txt
# OPT (small model, half precision)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_125M_weights_half/ -llm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_opt_125M_half.txt

# OPT (big model)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_opt_6B.txt
# OPT (big model, half precision)
../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights_half/ -llm-config ../inference/models/configs/opt_6B.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_opt_6B_half.txt


###############################################################################################
################################## Alignment tests ############################################
###############################################################################################

# Full precision
diff ../inference/output/incr_decoding_llama_7B.txt ../inference/output/spec_inference_llama.txt
diff ../inference/output/incr_decoding_opt_6B.txt ../inference/output/spec_inference_opt.txt

# Half precision
diff ../inference/output/incr_decoding_llama_7B_half.txt ../inference/output/spec_inference_llama_half.txt
diff ../inference/output/incr_decoding_opt_6B_half.txt ../inference/output/spec_inference_opt_half.txt

######################### Alignment tests with HuggingFace ####################################
pip3 install protobuf==3.20.3

# LLAMA (small model, full precision)
python3 ./inference/huggingface_inference.py --model-name "JackFram/llama-160m" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_160M.txt"

# LLAMA (small model, half precision)
python3 ./inference/huggingface_inference.py --model-name "JackFram/llama-160m" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_160M_half.txt"

# LLAMA (big model, full precision)
python3 ./inference/huggingface_inference.py --model-name "decapoda-research/llama-7b-hf" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_7B.txt"

# LLAMA (small model, half precision)
python3 ./inference/huggingface_inference.py --model-name "decapoda-research/llama-7b-hf" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_7B_half.txt"

diff <(tail -n +2 "../inference/output/huggingface_llama_160M.txt") <(tail -n +3 "../inference/output/incr_decoding_llama_160M.txt")
diff <(tail -n +2 "../inference/output/huggingface_llama_160M_half.txt") <(tail -n +3 "../inference/output/incr_decoding_llama_160M_half.txt")
diff <(tail -n +2 "../inference/output/huggingface_llama_7B.txt") <(tail -n +3 "../inference/output/incr_decoding_llama_7B.txt")
diff <(tail -n +2 "../inference/output/huggingface_llama_7B_half.txt") <(tail -n +3 "../inference/output/incr_decoding_llama_7B_half.txt")

###############################################################################################
###################################### Cleanup ################################################
###############################################################################################

# Clean up after test
cleanup
