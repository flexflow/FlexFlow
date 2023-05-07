#! /usr/bin/env bash
set -e
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit 1; fi
GPUS=$1
BATCHSIZE=$((GPUS * 64))
FSIZE=13800
ZSIZE=12192

GPU_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ $(( GPUS )) -gt $(( GPU_AVAILABLE )) ]; then echo "The test requires $GPUS GPUs, but only $GPU_AVAILABLE are available. Try reducing the number of nodes, or the number of gpus/node." ; exit; fi

# Check if the LLAMA/LLAMA inference example exists in the build folder. If so, run the tests out of the build folder
# Otherwise, look for the example binaries in the folders in the PATH, plus in the subdirectory of the flexflow
# Python package (if it exists)
if [[ -f "$FF_HOME/build/examples/cpp/inference/LLAMA/LLAMA" ]]; then
	echo "Running C++ tests from folder: $FF_HOME/build/examples/cpp"
	# Inference examples
	if [ $(( GPU_AVAILABLE )) -lt $(( 4 )) ]; then echo "Skipping LLAMA test because it requires 4 GPUs, but only $GPU_AVAILABLE are available. " ; exit 1; fi
	"$FF_HOME"/build/examples/cpp/inference/LLAMA/LLAMA -ll:gpu "$GPUS" -ll:util 8 -ll:fsize "$FSIZE" -ll:zsize 30000 --only-data-parallel
	#"$FF_HOME"/build/examples/cpp/inference/mixture_of_experts/inference_moe -ll:gpu "$GPUS" -ll:util 8 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --only-data-parallel
	#"$FF_HOME"/build/examples/cpp/inference/transformers/inference_transformers -ll:gpu "$GPUS" -ll:util 8 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --only-data-parallel
else
	python_packages=$(python -c "from distutils import sysconfig; print(sysconfig.get_python_lib(plat_specific=False,standard_lib=False))")
	OLD_PATH="$PATH"
	OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
	export PATH="${python_packages}/flexflow/bin:${PATH}"
	export LD_LIBRARY_PATH="${python_packages}/flexflow/lib:${LD_LIBRARY_PATH}"
	IFS=:
	found=false
	for path in $PATH; do
		if [[ -f "$path/LLAMA" ]]; then
			echo "Running Inference Tests from folder: $path"
			found=true
			# Inference examples
			if [ $(( GPU_AVAILABLE )) -lt $(( 4 )) ]; then echo "Skipping LLAMA test because it requires 4 GPUs, but only $GPU_AVAILABLE are available. " ; exit 1; fi
			LLAMA -ll:gpu "$GPUS" -ll:util 8 -ll:fsize "$FSIZE" -ll:zsize 30000 --only-data-parallel
			#inference_moe -ll:gpu "$GPUS" -ll:util 8 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --only-data-parallel
			#inference_transformers -ll:gpu "$GPUS" -ll:util 8 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --only-data-parallel
		fi
	done
	export PATH="$OLD_PATH"
	export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
	if [ ! $found ]; then echo "C++ test binaries not found"; exit 1; fi
fi
