#! /usr/bin/env bash
set -x
set -e

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Generate test configs
rm -rf python_test_configs/*.json
python python_test_configs/generate_configs.py

# Run all tests
# Loop through .json files in the ./python_test_configs dir 
for file in ./python_test_configs/*.json; do
    # Check filename prefix
    if [[ $file == *"incr_dec"* ]]; then
      script="../../inference/python/incr_decoding.py"
    elif [[ $file == *"spec_infer"* ]]; then  
      script="../../inference/python/spec_infer.py"
    fi
    # Run script
    python "$script" -config-file "$file" 
done


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
diff <(tail -n +3 "../../inference/output/incr_dec-python-llama-2-7b-hf-full_prec-1_tp_4_pp.txt") <(tail -n +3 "../../inference/output/spec_infer-python-llama-2-7b-hf-full_prec-1_tp_4_pp.txt")
diff <(tail -n +3 "../../inference/output/incr_dec-python-opt-6.7b-full_prec-1_tp_4_pp.txt")   <(tail -n +3 "../../inference/output/spec_infer-python-opt-6.7b-full_prec-1_tp_4_pp.txt")
# Half precision
check_partial_token_match "../../inference/output/incr_dec-python-llama-2-7b-hf-half_prec-1_tp_4_pp.txt" "../../inference/output/spec_infer-python-llama-2-7b-hf-half_prec-1_tp_4_pp.txt"
check_partial_token_match "../../inference/output/incr_dec-python-opt-6.7b-half_prec-1_tp_4_pp.txt" "../../inference/output/spec_infer-python-opt-6.7b-half_prec-1_tp_4_pp.txt"

# Speed test: speculative inference should be at very least 1.5x faster than incremental decoding
# Full precision
compare_decoding_steps_spec_infer_incr_decoding "../../inference/output/incr_dec-python-llama-2-7b-hf-full_prec-1_tp_4_pp.txt" "../../inference/output/spec_infer-python-llama-2-7b-hf-full_prec-1_tp_4_pp.txt"
compare_decoding_steps_spec_infer_incr_decoding "../../inference/output/incr_dec-python-opt-6.7b-full_prec-1_tp_4_pp.txt" "../../inference/output/spec_infer-python-opt-6.7b-full_prec-1_tp_4_pp.txt"
# Half precision
compare_decoding_steps_spec_infer_incr_decoding "../../inference/output/incr_dec-python-llama-2-7b-hf-half_prec-1_tp_4_pp.txt" "../../inference/output/spec_infer-python-llama-2-7b-hf-half_prec-1_tp_4_pp.txt"
compare_decoding_steps_spec_infer_incr_decoding "../../inference/output/incr_dec-python-opt-6.7b-half_prec-1_tp_4_pp.txt" "../../inference/output/spec_infer-python-opt-6.7b-half_prec-1_tp_4_pp.txt"

############ Alignment between tensor model parallelism and pipeline parallelism only #################
## Specinfer
# LLAMA
diff <(tail -n +3 "../../inference/output/spec_infer-python-llama-2-7b-hf-full_prec-2_tp_2_pp.txt") <(tail -n +3 "../../inference/output/spec_infer-python-llama-2-7b-hf-full_prec-1_tp_4_pp.txt")
check_partial_token_match "../../inference/output/spec_infer-python-llama-2-7b-hf-half_prec-2_tp_2_pp.txt" "../../inference/output/spec_infer-python-llama-2-7b-hf-half_prec-1_tp_4_pp.txt"
# OPT
diff <(tail -n +3 "../../inference/output/spec_infer-python-opt-6.7b-full_prec-2_tp_2_pp.txt")  <(tail -n +3 "../../inference/output/spec_infer-python-opt-6.7b-full_prec-1_tp_4_pp.txt")
check_partial_token_match "../../inference/output/spec_infer-python-opt-6.7b-half_prec-2_tp_2_pp.txt" "../../inference/output/spec_infer-python-opt-6.7b-half_prec-1_tp_4_pp.txt"

## Incremental decoding
# Small LLAMA
diff <(tail -n +3 "../../inference/output/incr_dec-python-llama-160m-full_prec-2_tp_2_pp.txt") <(tail -n +3 "../../inference/output/incr_dec-python-llama-160m-full_prec-1_tp_4_pp.txt")
check_partial_token_match "../../inference/output/incr_dec-python-llama-160m-half_prec-2_tp_2_pp.txt" "../../inference/output/incr_dec-python-llama-160m-half_prec-1_tp_4_pp.txt"
diff <(tail -n +3 "../../inference/output/incr_dec-python-llama-160m-full_prec-4_tp_1_pp.txt") <(tail -n +3 "../../inference/output/incr_dec-python-llama-160m-full_prec-1_tp_4_pp.txt")
check_partial_token_match "../../inference/output/incr_dec-python-llama-160m-half_prec-4_tp_1_pp.txt" "../../inference/output/incr_dec-python-llama-160m-half_prec-1_tp_4_pp.txt"
# Big LLAMA
diff <(tail -n +3 "../../inference/output/incr_dec-python-llama-2-7b-hf-full_prec-2_tp_2_pp.txt") <(tail -n +3 "../../inference/output/incr_dec-python-llama-2-7b-hf-full_prec-1_tp_4_pp.txt")
check_partial_token_match "../../inference/output/incr_dec-python-llama-2-7b-hf-half_prec-2_tp_2_pp.txt" "../../inference/output/incr_dec-python-llama-2-7b-hf-half_prec-1_tp_4_pp.txt"
#diff <(tail -n +3 "../../inference/output/incr_dec-python-llama-2-7b-hf-full_prec-4_tp_1_pp.txt") <(tail -n +3 "../../inference/output/incr_dec-python-llama-2-7b-hf-full_prec-1_tp_4_pp.txt")
#check_partial_token_match "../../inference/output/incr_dec-python-llama-2-7b-hf-half_prec-4_tp_1_pp.txt" "../../inference/output/incr_dec-python-llama-2-7b-hf-half_prec-1_tp_4_pp.txt"
# Small OPT
diff <(tail -n +3 "../../inference/output/incr_dec-python-opt-125m-full_prec-2_tp_2_pp.txt") <(tail -n +3 "../../inference/output/incr_dec-python-opt-125m-full_prec-1_tp_4_pp.txt")
check_partial_token_match "../../inference/output/incr_dec-python-opt-125m-half_prec-2_tp_2_pp.txt" "../../inference/output/incr_dec-python-opt-125m-half_prec-1_tp_4_pp.txt"
diff <(tail -n +3 "../../inference/output/incr_dec-python-opt-125m-full_prec-4_tp_1_pp.txt") <(tail -n +3 "../../inference/output/incr_dec-python-opt-125m-full_prec-1_tp_4_pp.txt")
check_partial_token_match "../../inference/output/incr_dec-python-opt-125m-half_prec-4_tp_1_pp.txt" "../../inference/output/incr_dec-python-opt-125m-half_prec-1_tp_4_pp.txt"
# Big OPT
diff <(tail -n +3 "../../inference/output/incr_dec-python-opt-6.7b-full_prec-2_tp_2_pp.txt") <(tail -n +3 "../../inference/output/incr_dec-python-opt-6.7b-full_prec-1_tp_4_pp.txt")
check_partial_token_match "../../inference/output/incr_dec-python-opt-6.7b-half_prec-2_tp_2_pp.txt" "../../inference/output/incr_dec-python-opt-6.7b-half_prec-1_tp_4_pp.txt"
#diff <(tail -n +3 "../../inference/output/incr_dec-python-opt-6.7b-full_prec-4_tp_1_pp.txt") <(tail -n +3 "../../inference/output/incr_dec-python-opt-6.7b-full_prec-1_tp_4_pp.txt")
#check_partial_token_match "../../inference/output/incr_dec-python-opt-6.7b-half_prec-4_tp_1_pp.txt" "../../inference/output/incr_dec-python-opt-6.7b-half_prec-1_tp_4_pp.txt"


######################### Alignment tests with HuggingFace ####################################

# LLAMA (small model, full precision)
python3 ./huggingface_inference.py --model-name "JackFram/llama-160m" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_160M.txt" --gpu

# LLAMA (small model, half precision)
python3 ./huggingface_inference.py --model-name "JackFram/llama-160m" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_160M_half.txt" --gpu

# LLAMA (big model, full precision)
python3 ./huggingface_inference.py --model-name "meta-llama/Llama-2-7b-hf" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_7B.txt"

# LLAMA (big model, half precision)
python3 ./huggingface_inference.py --model-name "meta-llama/Llama-2-7b-hf" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_llama_7B_half.txt" --gpu

# OPT (small model, full precision)
python3 ./huggingface_inference.py --model-name "facebook/opt-125m" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_125M.txt" --gpu --max-length 128

# OPT (small model, half precision)
python3 ./huggingface_inference.py --model-name "facebook/opt-125m" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_125M_half.txt" --gpu --max-length 128

# OPT (big model, full precision)
python3 ./huggingface_inference.py --model-name "facebook/opt-6.7b" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_6B.txt" --max-length 128

# OPT (big model, half precision)
#python3 ./huggingface_inference.py --model-name "facebook/opt-6.7b" --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_opt_6B_half.txt" --gpu --max-length 128

# Falcon (full precision)
python3 ./huggingface_inference.py --model-name "tiiuae/falcon-7b" --use-full-precision --prompt-file "../../inference/prompt/test.json" --output-file "../../inference/output/huggingface_falcon_7B.txt" --max-length 128

diff "../../inference/output/huggingface_llama_160M.txt" <(tail -n +4 "../../inference/output/incr_dec-python-llama-160m-full_prec-1_tp_4_pp.txt")
diff <( < ../../inference/output/huggingface_llama_160M_half.txt tr -s '[:space:]' '\n' | head -n 20) <(tail -n +4 "../../inference/output/incr_dec-python-llama-160m-half_prec-1_tp_4_pp.txt" | tr -s '[:space:]' '\n' | head -n 20)
diff "../../inference/output/huggingface_llama_7B.txt" <(tail -n +4 "../../inference/output/incr_dec-python-llama-2-7b-hf-full_prec-1_tp_4_pp.txt")
diff <( < ../../inference/output/huggingface_llama_7B_half.txt tr -s '[:space:]' '\n' | head -n 20) <(tail -n +4 "../../inference/output/incr_dec-python-llama-2-7b-hf-half_prec-1_tp_4_pp.txt" | tr -s '[:space:]' '\n' | head -n 20)

diff "../../inference/output/huggingface_opt_125M.txt" <(tail -n +4 "../../inference/output/incr_dec-python-opt-125m-full_prec-1_tp_4_pp.txt")
diff <( < ../../inference/output/huggingface_opt_125M_half.txt tr -s '[:space:]' '\n' | head -n 20) <(tail -n +4 "../../inference/output/incr_dec-python-opt-125m-half_prec-1_tp_4_pp.txt" | tr -s '[:space:]' '\n' | head -n 20)
diff "../../inference/output/huggingface_opt_6B.txt" <(tail -n +4 "../../inference/output/incr_dec-python-opt-6.7b-full_prec-1_tp_4_pp.txt")
#diff "../../inference/output/huggingface_opt_6B_half.txt" <(tail -n +4 "../../inference/output/incr_dec-python-opt-6.7b-half_prec-1_tp_4_pp.txt")
diff "../../inference/output/huggingface_falcon_7B.txt" <(tail -n +4 "../../inference/output/incr_dec-python-falcon-7b-full_prec-1_tp_4_pp.txt")
