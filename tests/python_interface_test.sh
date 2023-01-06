#! /usr/bin/env bash
set -x
set -e

check_python_interface() {
	# Usage: check_python_interface {python, flexflow_python}
	GPUS=1
	BATCHSIZE=$((GPUS * 64))
	FSIZE=14048
	ZSIZE=12192
	interpreter=${1:-python}
	if [[ "$interpreter" == "python" ]]; then
		export FF_USE_NATIVE_PYTHON=1
		EXE="python"
		echo "Running a single-GPU Python test to check the Python interface (native python interpreter)"
		$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp.py -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
		unset FF_USE_NATIVE_PYTHON
	elif [[ "$interpreter" == "flexflow_python" ]]; then
		EXE="$FF_HOME"/python/flexflow_python
		echo "Running a single-GPU Python test to check the Python interface (flexflow_python interpreter)"
		$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	else
		echo "Invalid Python interpreter"
		exit 1
	fi
}


if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi

installation_status=${1:-"before-installation"}
echo "Running Python interface tests (installation status: ${installation_status})"
if [[ "$installation_status" == "before-installation" ]]; then
	# Import flexflow.core module in Python
	export PYTHONPATH="${FF_HOME}/python"
	python -c "import flexflow.core; exit()"
	unset PYTHONPATH
	# Run a single-gpu test using the flexflow_python interpreter
	check_python_interface flexflow_python
	# Run a single-gpu test using the native python interpreter
	export PYTHONPATH="${FF_HOME}/python:${FF_HOME}/build/python"
	check_python_interface python
	unset PYTHONPATH
elif [[ "$installation_status" == "after-installation" ]]; then
	# Import flexflow.core module in Python
	python -c "import flexflow.core; exit()"
	# Run a single-gpu test using the flexflow_python interpreter
	check_python_interface flexflow_python
	# Run a single-gpu test using the native python interpreter
	check_python_interface python
else
	echo "Invalid installation status!"
	echo "Usage: $0 {before-installation, after-installation}"
	exit 1
fi
