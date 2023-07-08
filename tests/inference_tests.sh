#! /usr/bin/env bash
set -x
set -e

cleanup() {
    rm -rf ../inference/prompt ../inference/weights ../inference/tokenizer ../inference/output
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Enable model parallelism tests, if desired
TENSOR_PARALLELISM_TESTS=${TENSOR_PARALLELISM_TESTS:-OFF}

# Clean up before test (just in case)
cleanup

# Update the transformers library to support the LLAMA model

pip3 install --upgrade transformers sentencepiece

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

# Tensor parallelism tests
if [ "$TENSOR_PARALLELISM_TESTS" = "ON" ]; then
    # LLAMA
    ../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -ssm-model llama -ssm-weight ../inference/weights/llama_160M_weights/ -ssm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_llama_tp.txt -tensor-parallelism-degree 2
    # LLAMA (half precision)
    ../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights_half/ -llm-config ../inference/models/configs/llama_7B.json -ssm-model llama -ssm-weight ../inference/weights/llama_160M_weights_half/ -ssm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_llama_half_tp.txt -tensor-parallelism-degree 2
    
    # OPT
    ../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -ssm-model opt -ssm-weight ../inference/weights/opt_125M_weights/ -ssm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_opt_tp.txt -tensor-parallelism-degree 2
    # OPT (half precision)
    ../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights_half/ -llm-config ../inference/models/configs/opt_6B.json -ssm-model opt -ssm-weight ../inference/weights/opt_125M_weights_half/ -ssm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/spec_inference_opt_half_tp.txt -tensor-parallelism-degree 2
fi

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

# Tensor parallelism tests
if [ "$TENSOR_PARALLELISM_TESTS" = "ON" ]; then
    # LLAMA (small model)
    ../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model llama -llm-weight ../inference/weights/llama_160M_weights/ -llm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_160M_tp.txt -tensor-parallelism-degree 2
    # LLAMA (small model, half precision)
    ../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_160M_weights_half/ -llm-config ../inference/models/configs/llama_160M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_160M_half_tp.txt -tensor-parallelism-degree 2

    # LLAMA (big model)
    ../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_7B_tp.txt -tensor-parallelism-degree 2
    # LLAMA (big model, half precision)
    ../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights_half/ -llm-config ../inference/models/configs/llama_7B.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_llama_7B_half_tp.txt -tensor-parallelism-degree 2

    # OPT (small model)
    ../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model opt -llm-weight ../inference/weights/opt_125M_weights/ -llm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_opt_125M_tp.txt -tensor-parallelism-degree 2
    # OPT (small model, half precision)
    ../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_125M_weights_half/ -llm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_opt_125M_half_tp.txt -tensor-parallelism-degree 2

    # OPT (big model)
    ../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --use-full-precision -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_opt_6B_tp.txt -tensor-parallelism-degree 2
    # OPT (big model, half precision)
    ../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights_half/ -llm-config ../inference/models/configs/opt_6B.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json -output-file ../inference/output/incr_decoding_opt_6B_half_tp.txt -tensor-parallelism-degree 2
fi

###############################################################################################
############################### Alignment and Speed tests #####################################
###############################################################################################

############ Alignment between speculative inference and incremental decoding #################
# Full precision
diff <(tail -n +2 "../inference/output/incr_decoding_llama_7B.txt") <(tail -n +2 "../inference/output/spec_inference_llama.txt")
diff <(tail -n +2 "../inference/output/incr_decoding_opt_6B.txt") <(tail -n +2 "../inference/output/spec_inference_opt.txt")
# Half precision
#diff <(tail -n +2 "../inference/output/incr_decoding_llama_7B_half.txt") <(tail -n +2 "../inference/output/spec_inference_llama_half.txt")
#diff <(tail -n +2 "../inference/output/incr_decoding_opt_6B_half.txt" ) <(tail -n +2 "../inference/output/spec_inference_opt_half.txt")

# Speed test: speculative inference should be at very least 1.5x faster than incremental decoding
function compare_speed_spec_infer_incr_decoding {
    local incrDec_file="$1"
    local specInf_file="$2"

    # Read the float numbers from the first line of the files
    incrDec=$(sed -n '1 s/end-to-end latency: \(.*\)/\1/p' "$incrDec_file")
    specInf=$(sed -n '1 s/end-to-end latency: \(.*\)/\1/p' "$specInf_file")

    if ! command -v bc &> /dev/null; then
        echo "bc is not installed. Installing..."
        sudo apt-get install -y bc
    fi
    
    # Perform the comparison
    threshold=$(bc <<< "$specInf * 1.5")
    if (( $(echo "$incrDec >= $threshold" | bc -l) )); then
        #echo "The latency in $specInf_file is at least 1.5x smaller than the latency from $incrDec_file."
        :
    else
        echo "Error: The latency in $specInf_file is not at least 1.5x smaller than the latency in $incrDec_file!"
        exit 1
    fi
}
# Full precision
compare_speed_spec_infer_incr_decoding "../inference/output/incr_decoding_llama_7B.txt" "../inference/output/spec_inference_llama.txt"
compare_speed_spec_infer_incr_decoding "../inference/output/incr_decoding_opt_6B.txt" "../inference/output/spec_inference_opt.txt"
# Half precision
#compare_speed_spec_infer_incr_decoding "../inference/output/incr_decoding_llama_7B_half.txt" "../inference/output/spec_inference_llama_half.txt"
#compare_speed_spec_infer_incr_decoding "../inference/output/incr_decoding_opt_6B_half.txt" "../inference/output/spec_inference_opt_half.txt"

############ Alignment between tensor model parallelism and pipeline parallelism only #################
if [ "$TENSOR_PARALLELISM_TESTS" = "ON" ]; then
    diff <(tail -n +2 "../inference/output/spec_inference_llama_tp.txt") <(tail -n +2 "../inference/output/spec_inference_llama.txt")
    diff <(tail -n +2 "../inference/output/spec_inference_opt_tp.txt") <(tail -n +2 "../inference/output/spec_inference_opt.txt")
    diff <(tail -n +2 "../inference/output/spec_inference_llama_half_tp.txt") <(tail -n +2 "../inference/output/spec_inference_llama_half.txt")
    diff <(tail -n +2 "../inference/output/spec_inference_opt_half_tp.txt") <(tail -n +2 "../inference/output/spec_inference_opt_half.txt")
    diff <(tail -n +2 "../inference/output/incr_decoding_llama_160M_tp.txt") <(tail -n +2 "../inference/output/incr_decoding_llama_160M.txt")
    # diff <(tail -n +2 "../inference/output/incr_decoding_llama_160M_half_tp.txt") <(tail -n +2 "../inference/output/incr_decoding_llama_160M_half.txt")
    diff <(tail -n +2 "../inference/output/incr_decoding_llama_7B_tp.txt") <(tail -n +2 "../inference/output/incr_decoding_llama_7B.txt")
    diff <(tail -n +2 "../inference/output/incr_decoding_llama_7B_half_tp.txt") <(tail -n +2 "../inference/output/incr_decoding_llama_7B_half.txt")
    diff <(tail -n +2 "../inference/output/incr_decoding_opt_125M_tp.txt") <(tail -n +2 "../inference/output/incr_decoding_opt_125M.txt")
    diff <(tail -n +2 "../inference/output/incr_decoding_opt_125M_half_tp.txt") <(tail -n +2 "../inference/output/incr_decoding_opt_125M_half.txt")
    diff <(tail -n +2 "../inference/output/incr_decoding_opt_6B_tp.txt") <(tail -n +2 "../inference/output/incr_decoding_opt_6B.txt")
    diff <(tail -n +2 "../inference/output/incr_decoding_opt_6B_half_tp.txt") <(tail -n +2 "../inference/output/incr_decoding_opt_6B_half.txt")
fi

######################### Alignment tests with HuggingFace ####################################
pip3 install protobuf==3.20.3

# LLAMA (small model, full precision)
python3 ./inference/huggingface_inference.py --model-name "JackFram/llama-160m" --tokenizer-model-name "JackFram/llama-160m" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_160M.txt" --gpu

# LLAMA (small model, half precision)
python3 ./inference/huggingface_inference.py --model-name "JackFram/llama-160m" --tokenizer-model-name "JackFram/llama-160m" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_160M_half.txt" --gpu

# LLAMA (big model, full precision)
python3 ./inference/huggingface_inference.py --model-name "decapoda-research/llama-7b-hf" --tokenizer-model-name "JackFram/llama-160m" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_7B.txt"

# LLAMA (big model, half precision)
python3 ./inference/huggingface_inference.py --model-name "decapoda-research/llama-7b-hf" --tokenizer-model-name "JackFram/llama-160m" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_7B_half.txt" --gpu

# OPT (small model, full precision)
python3 ./inference/huggingface_inference.py --model-name "facebook/opt-125m" --tokenizer-model-name "facebook/opt-125m" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_125M.txt" --gpu --max-length 128

# OPT (small model, half precision)
python3 ./inference/huggingface_inference.py --model-name "facebook/opt-125m" --tokenizer-model-name "facebook/opt-125m" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_125M_half.txt" --gpu --max-length 128

# OPT (big model, full precision)
#python3 ./inference/huggingface_inference.py --model-name "facebook/opt-6.7b" --tokenizer-model-name "facebook/opt-6.7b" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_6B.txt" --max-length 127

# OPT (big model, half precision)
#python3 ./inference/huggingface_inference.py --model-name "facebook/opt-6.7b" --tokenizer-model-name "facebook/opt-6.7b" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_6B_half.txt" --gpu --max-length 127

diff <(tail -n +2 "../inference/output/huggingface_llama_160M.txt") <(tail -n +4 "../inference/output/incr_decoding_llama_160M.txt")
diff <(tail -n +2 "../inference/output/huggingface_llama_160M_half.txt") <(tail -n +4 "../inference/output/incr_decoding_llama_160M_half.txt")
diff <(tail -n +2 "../inference/output/huggingface_llama_7B.txt") <(tail -n +4 "../inference/output/incr_decoding_llama_7B.txt")
diff <(tail -n +2 "../inference/output/huggingface_llama_7B_half.txt") <(tail -n +4 "../inference/output/incr_decoding_llama_7B_half.txt")

diff <(tail -n +2 "../inference/output/huggingface_opt_125M.txt") <(tail -n +4 "../inference/output/incr_decoding_opt_125M.txt")
diff <(tail -n +2 "../inference/output/huggingface_opt_125M_half.txt") <(tail -n +4 "../inference/output/incr_decoding_opt_125M_half.txt")
#diff <(tail -n +2 "../inference/output/huggingface_opt_6B.txt") <(tail -n +4 "../inference/output/incr_decoding_opt_6B.txt")
#diff <(tail -n +2 "../inference/output/huggingface_opt_6B_half.txt") <(tail -n +4 "../inference/output/incr_decoding_opt_6B_half.txt")

###############################################################################################
###################################### Cleanup ################################################
###############################################################################################

# Clean up after test
# cleanup
