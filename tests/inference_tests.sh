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

# Download the weights
python3 ../inference/utils/download_llama_weights.py
python3 ../inference/utils/download_opt_weights.py

# Create test prompt file
mkdir -p ../inference/prompt
echo '["Give three tips for staying healthy."]' > ../inference/prompt/test.json

# Find location of the spec_infer and incr_decoding programs on the system. When building in place, they will be located at
# $FF_HOME/build/inference/{spec_infer,incr_decoding}/. When installing FlexFlow with pip (as it happens in CI
# located in the FlexFlow package folder within the Python library folder. 
PATH_PREFIX=""
OLD_PATH=""
OLD_LD_LIBRARY_PATH=""
if [[ -f "../build/inference/spec_infer/spec_infer" ]]; then
    echo "Running inference tests from folder: ../build"
    PATH_PREFIX="../build"
else
    python_packages=$(python -c "from distutils import sysconfig; print(sysconfig.get_python_lib(plat_specific=False,standard_lib=False))")
	OLD_PATH="$PATH"
	OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
	export PATH="${python_packages}/flexflow/bin:${PATH}"
	export LD_LIBRARY_PATH="${python_packages}/flexflow/lib:${LD_LIBRARY_PATH}"
	IFS=:
	found=false
	for path in $PATH; do
		if [[ -f "$path/spec_infer" ]]; then
			echo "Running inference tests from folder: $path"
			found=true
            PATH_PREFIX="$path"
        fi
    done
    if [ ! $found ]; then echo "Inference test binaries not found"; exit 1; fi 
fi

###############################################################################################
############################ Speculative inference tests ######################################
###############################################################################################

FULL_PREFIX=""
if [[ "${PATH_PREFIX}" == "../build" ]]; then
    FULL_PREFIX="${PATH_PREFIX}/inference/spec_infer/"
fi

# LLAMA
"$FULL_PREFIX"spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -ssm-model llama -ssm-weight ../inference/weights/llama_190M_weights/ -ssm-config ../inference/models/configs/llama_190M.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json

# OPT
"$FULL_PREFIX"spec_infer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -ssm-model opt -ssm-weight ../inference/weights/opt_125M_weights/ -ssm-config ../inference/models/configs/opt_125M.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json

###############################################################################################
############################ Incremental decoding tests #######################################
###############################################################################################

FULL_PREFIX=""
if [[ "${PATH_PREFIX}" == "../build" ]]; then
    FULL_PREFIX="${PATH_PREFIX}/inference/spec_infer/"
fi

# LLAMA
"$FULL_PREFIX"/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model llama -llm-weight ../inference/weights/llama_7B_weights/ -llm-config ../inference/models/configs/llama_7B.json -tokenizer ../inference/tokenizer/tokenizer.model -prompt ../inference/prompt/test.json

# OPT
"$FULL_PREFIX"/incr_decoding -ll:gpu 4 -ll:fsize 14000 -ll:zsize 30000 -llm-model opt -llm-weight ../inference/weights/opt_6B_weights/ -llm-config ../inference/models/configs/opt_6B.json -tokenizer ../inference/tokenizer/ -prompt ../inference/prompt/test.json

# TODO: check alignment of results

if [[ "${PATH_PREFIX}" != "../build" ]]; then
    export PATH="$OLD_PATH"
	export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
fi

# Clean up after test
cleanup
