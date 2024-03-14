#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

###############################################################################################
############################ Speculative inference tests ######################################
###############################################################################################

# LLAMA
../../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/spec_inference_llama.txt -pipeline-parallelism-degree 4
# LLAMA (half precision)
../../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/spec_inference_llama_half.txt -pipeline-parallelism-degree 4

# OPT
../../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model facebook/opt-6.7b -ssm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/spec_inference_opt.txt -pipeline-parallelism-degree 4
# OPT (half precision)
../../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model facebook/opt-6.7b -ssm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/spec_inference_opt_half.txt -pipeline-parallelism-degree 4

# Tensor parallelism tests
if [ "$TENSOR_PARALLELISM_TESTS" = "ON" ]; then
    # LLAMA
    ../../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/spec_inference_llama_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
    # LLAMA (half precision)
    ../../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model meta-llama/Llama-2-7b-hf -ssm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/spec_inference_llama_half_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
    
    # OPT
    ../../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model facebook/opt-6.7b -ssm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/spec_inference_opt_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
    # OPT (half precision)
    ../../build/inference/spec_infer/spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model facebook/opt-6.7b -ssm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/spec_inference_opt_half_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
fi

###############################################################################################
############################ Incremental decoding tests #######################################
###############################################################################################

# LLAMA (small model)
../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_160M.txt -pipeline-parallelism-degree 4

../../build/inference/incr_decoding/incr_decoding -ll:gpu 1 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_160M.txt -pipeline-parallelism-degree 1

# LLAMA (small model, half precision)
../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_160M_half.txt -pipeline-parallelism-degree 4

# LLAMA (big model)
../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model meta-llama/Llama-2-7b-hf -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_2_7B.txt -pipeline-parallelism-degree 4
# LLAMA (big model, half precision)
../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model meta-llama/Llama-2-7b-hf -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_2_7B_half.txt -pipeline-parallelism-degree 4

# OPT (small model)
../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_125M.txt -pipeline-parallelism-degree 4
# OPT (small model, half precision)
../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_125M_half.txt -pipeline-parallelism-degree 4

# OPT (big model)
../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model facebook/opt-6.7b -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_6B.txt -pipeline-parallelism-degree 4
# OPT (big model, half precision)
../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model facebook/opt-6.7b -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_6B_half.txt -pipeline-parallelism-degree 4

# Falcon (full precision)
../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 40000 --fusion --use-full-precision -llm-model tiiuae/falcon-7b -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_falcon_7B.txt -pipeline-parallelism-degree 4
# Falcon (half precision)
# ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model tiiuae/falcon-7b -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_falcon_7B.txt -pipeline-parallelism-degree 4

# # StarCoder (full precision)
# ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model bigcode/starcoderbase-7b -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_starcoder_7B.txt -pipeline-parallelism-degree 4
# # StarCoder (half precision)
# ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model bigcode/starcoderbase-7b -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_starcoder_7B_half.txt -pipeline-parallelism-degree 4

# Tensor parallelism tests
if [ "$TENSOR_PARALLELISM_TESTS" = "ON" ]; then
    # LLAMA (small model)
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_160M_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_160M_tp4.txt -pipeline-parallelism-degree 1 -tensor-parallelism-degree 4
    # LLAMA (small model, half precision)
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_160M_half_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model JackFram/llama-160m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_160M_half_tp4.txt -pipeline-parallelism-degree 1 -tensor-parallelism-degree 4

    # LLAMA (big model)
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model meta-llama/Llama-2-7b-hf -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_2_7B_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
    # LLAMA (big model, half precision)
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model meta-llama/Llama-2-7b-hf -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_llama_2_7B_half_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2

    # OPT (small model)
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_125M_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_125M_tp4.txt -pipeline-parallelism-degree 1 -tensor-parallelism-degree 4
    # OPT (small model, half precision)
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_125M_half_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model facebook/opt-125m -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_125M_half_tp.txt -pipeline-parallelism-degree 1 -tensor-parallelism-degree 4

    # OPT (big model)
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion --use-full-precision -llm-model facebook/opt-6.7b -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_6B_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
    # OPT (big model, half precision)
    ../../build/inference/incr_decoding/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 --fusion -llm-model facebook/opt-6.7b -prompt ../../inference/prompt/test.json -output-file ../../inference/output/incr_decoding_opt_6B_half_tp.txt -pipeline-parallelism-degree 2 -tensor-parallelism-degree 2
fi

###############################################################################################
############################### Alignment and Speed tests #####################################
###############################################################################################

##################################### Helper functions #######################################
function check_partial_token_match {
    local file1="$1"
    local file2="$2"
    local num_tokens_to_match=30

    # Read the second line of the first file
    third_line=$(sed -n '3p' "$file1")
    read -r line1 <<< "$third_line"
    tokens1=${line1#*: }
    IFS=',' read -ra arr1 <<< "$tokens1"

    # Read the second line of the second file
    third_line=$(sed -n '3p' "$file2")
    read -r line2 <<< "$third_line"
    tokens2=${line2#*: }
    IFS=',' read -ra arr2 <<< "$tokens2"

    # Compare the first few integers in the two lists
    for ((i = 0; i < num_tokens_to_match; i++)); do
        if [[ "${arr1[$i]}" != "${arr2[$i]}" ]]; then
            echo "The first $num_tokens_to_match tokens in files $file1 and $file2 are not identical."
            exit 1
        fi
    done
    #echo "The first $num_tokens_to_match integers are identical."
}

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

function compare_decoding_steps_spec_infer_incr_decoding {
    local incrDec_file="$1"
    local specInf_file="$2"

    # Read the number of decoding steps from the second line of the files
    second_line=$(sed -n '2p' "$incrDec_file")
    read -r line <<< "$second_line"
    incrDec=${line#*: }
    second_line=$(sed -n '2p' "$specInf_file")
    read -r line <<< "$second_line"
    specInf=${line#*: }

    if ! command -v bc &> /dev/null; then
        echo "bc is not installed. Installing..."
        sudo apt-get install -y bc
    fi
    
    # Perform the comparison
    threshold=$(bc <<< "$specInf * 1.5")
    if (( $(echo "$incrDec >= $threshold" | bc -l) )); then
        #echo "The decoding steps in $specInf_file are at least 1.5x less than those in $incrDec_file."
        :
    else
        echo "Error: The decoding steps in $specInf_file are not at least 1.5x less than those in $incrDec_file!"
        exit 1
    fi
}

############ Alignment between speculative inference and incremental decoding #################
# Full precision
diff <(tail -n +3 "../../inference/output/incr_decoding_llama_2_7B.txt") <(tail -n +3 "../../inference/output/spec_inference_llama.txt")
diff <(tail -n +3 "../../inference/output/incr_decoding_opt_6B.txt")   <(tail -n +3 "../../inference/output/spec_inference_opt.txt")
# Half precision
check_partial_token_match "../../inference/output/incr_decoding_llama_2_7B_half.txt" "../../inference/output/spec_inference_llama_half.txt"
check_partial_token_match "../../inference/output/incr_decoding_opt_6B_half.txt" "../../inference/output/spec_inference_opt_half.txt"

# Speed test: speculative inference should be at very least 1.5x faster than incremental decoding
# Full precision
#compare_speed_spec_infer_incr_decoding "../../inference/output/incr_decoding_llama_2_7B.txt" "../../inference/output/spec_inference_llama.txt"
#compare_speed_spec_infer_incr_decoding "../../inference/output/incr_decoding_opt_6B.txt" "../../inference/output/spec_inference_opt.txt"
compare_decoding_steps_spec_infer_incr_decoding "../../inference/output/incr_decoding_llama_2_7B.txt" "../../inference/output/spec_inference_llama.txt"
compare_decoding_steps_spec_infer_incr_decoding "../../inference/output/incr_decoding_opt_6B.txt" "../../inference/output/spec_inference_opt.txt"
# Half precision
#compare_speed_spec_infer_incr_decoding "../../inference/output/incr_decoding_llama_2_7B_half.txt" "../../inference/output/spec_inference_llama_half.txt"
#compare_speed_spec_infer_incr_decoding "../../inference/output/incr_decoding_opt_6B_half.txt" "../../inference/output/spec_inference_opt_half.txt"
compare_decoding_steps_spec_infer_incr_decoding "../../inference/output/incr_decoding_llama_2_7B_half.txt" "../../inference/output/spec_inference_llama_half.txt"
compare_decoding_steps_spec_infer_incr_decoding "../../inference/output/incr_decoding_opt_6B_half.txt" "../../inference/output/spec_inference_opt_half.txt"

############ Alignment between tensor model parallelism and pipeline parallelism only #################
if [ "$TENSOR_PARALLELISM_TESTS" = "ON" ]; then
    diff <(tail -n +3 "../../inference/output/spec_inference_llama_tp.txt") <(tail -n +3 "../../inference/output/spec_inference_llama.txt")
    diff <(tail -n +3 "../../inference/output/spec_inference_opt_tp.txt")  <(tail -n +3 "../../inference/output/spec_inference_opt.txt")
    check_partial_token_match "../../inference/output/spec_inference_llama_half_tp.txt" "../../inference/output/spec_inference_llama_half.txt"
    check_partial_token_match "../../inference/output/spec_inference_opt_half_tp.txt" "../../inference/output/spec_inference_opt_half.txt"
    diff <(tail -n +3 "../../inference/output/incr_decoding_llama_160M_tp.txt") <(tail -n +3 "../../inference/output/incr_decoding_llama_160M.txt")
    check_partial_token_match "../../inference/output/incr_decoding_llama_160M_half_tp.txt" "../../inference/output/incr_decoding_llama_160M_half.txt"
    diff <(tail -n +3 "../../inference/output/incr_decoding_llama_2_7B_tp.txt") <(tail -n +3 "../../inference/output/incr_decoding_llama_2_7B.txt")
    check_partial_token_match "../../inference/output/incr_decoding_llama_2_7B_half_tp.txt" "../../inference/output/incr_decoding_llama_2_7B_half.txt"
    diff <(tail -n +3 "../../inference/output/incr_decoding_opt_125M_tp.txt") <(tail -n +3 "../../inference/output/incr_decoding_opt_125M.txt")
    check_partial_token_match "../../inference/output/incr_decoding_opt_125M_half_tp.txt" "../../inference/output/incr_decoding_opt_125M_half.txt"
    diff <(tail -n +3 "../../inference/output/incr_decoding_opt_6B_tp.txt") <(tail -n +3 "../../inference/output/incr_decoding_opt_6B.txt")
    check_partial_token_match "../../inference/output/incr_decoding_opt_6B_half_tp.txt" "../../inference/output/incr_decoding_opt_6B_half.txt"
fi

######################### Alignment tests with HuggingFace ####################################

# LLAMA (small model, full precision)
python3 ./huggingface_inference.py --model-name "JackFram/llama-160m" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_160M.txt" --gpu

# LLAMA (small model, half precision)
python3 ./huggingface_inference.py --model-name "JackFram/llama-160m" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_160M_half.txt" --gpu

# LLAMA (big model, full precision)
python3 ./huggingface_inference.py --model-name "meta-llama/Llama-2-7b-hf" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_2_7B.txt"

# LLAMA (big model, half precision)
python3 ./huggingface_inference.py --model-name "meta-llama/Llama-2-7b-hf" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_2_7B_half.txt" --gpu

# OPT (small model, full precision)
python3 ./huggingface_inference.py --model-name "facebook/opt-125m" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_125M.txt" --gpu --max-length 128

# OPT (small model, half precision)
python3 ./huggingface_inference.py --model-name "facebook/opt-125m" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_125M_half.txt" --gpu --max-length 128

# OPT (big model, full precision)
python3 ./huggingface_inference.py --model-name "facebook/opt-6.7b" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_6B.txt" --max-length 128

# OPT (big model, half precision)
# python3 ./huggingface_inference.py --model-name "facebook/opt-6.7b" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_6B_half.txt" --gpu --max-length 128

# Falcon (full precision)
python3 ./huggingface_inference.py --model-name "tiiuae/falcon-7b" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_falcon_7B.txt" --max-length 128


diff "../../inference/output/huggingface_llama_160M.txt" <(tail -n +4 "../../inference/output/incr_decoding_llama_160M.txt")
diff <( < ../../inference/output/huggingface_llama_160M_half.txt tr -s '[:space:]' '\n' | head -n 20) <(tail -n +4 "../../inference/output/incr_decoding_llama_160M_half.txt" | tr -s '[:space:]' '\n' | head -n 20)
diff "../../inference/output/huggingface_llama_2_7B.txt" <(tail -n +4 "../../inference/output/incr_decoding_llama_2_7B.txt")
diff <( < ../../inference/output/huggingface_llama_2_7B_half.txt tr -s '[:space:]' '\n' | head -n 20) <(tail -n +4 "../../inference/output/incr_decoding_llama_2_7B_half.txt" | tr -s '[:space:]' '\n' | head -n 20)

diff "../../inference/output/huggingface_opt_125M.txt" <(tail -n +4 "../../inference/output/incr_decoding_opt_125M.txt")
diff <( < ../../inference/output/huggingface_opt_125M_half.txt tr -s '[:space:]' '\n' | head -n 20) <(tail -n +4 "../../inference/output/incr_decoding_opt_125M_half.txt" | tr -s '[:space:]' '\n' | head -n 20)
diff "../../inference/output/huggingface_opt_6B.txt" <(tail -n +4 "../../inference/output/incr_decoding_opt_6B.txt")
# diff "../../inference/output/huggingface_opt_6B_half.txt" <(tail -n +4 "../../inference/output/incr_decoding_opt_6B_half.txt")
diff "../../inference/output/huggingface_falcon_7B.txt" <(tail -n +4 "../../inference/output/incr_decoding_falcon_7B.txt")

