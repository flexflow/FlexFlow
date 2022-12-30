#! /usr/bin/env bash
set -x
set -e

GPUS=1
BATCHSIZE=$((GPUS * 64))
FSIZE=14048
ZSIZE=12192

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi
interpreter=${1:-python}
if [[ "$interpreter" == "python" ]]; then
	EXE="python"
	export FF_USE_NATIVE_PYTHON=1
	export PYTHONPATH="${FF_HOME}/build/python:${PYTHONPATH}"
	echo "Running a single-GPU Python test to check the Python interface (native python interpreter)"
	$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp.py -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
elif [[ "$interpreter" == "flexflow_python" ]]; then
	EXE="$FF_HOME"/python/flexflow_python
	echo "Running a single-GPU Python test to check the Python interface (flexflow_python interpreter)"
	$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
else
	echo "Invalid Python interpreter"
	exit 1
fi
