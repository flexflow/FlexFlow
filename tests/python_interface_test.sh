#! /usr/bin/env bash
set -x
set -e

check_python_interface() {
	# Usage: check_python_interface {python, flexflow_python} {before-installation, after-installation}
	GPUS=1
	BATCHSIZE=$((GPUS * 64))
	FSIZE=14048
	ZSIZE=12192
	interpreter=${1:-python}
	installation_status=${2:-"before-installation"}
	if [[ "$interpreter" == "python" ]]; then
		EXE="python"
		echo "Running a single-GPU Python test to check the Python interface (native python interpreter)"
		$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp.py -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	elif [[ "$interpreter" == "flexflow_python" ]]; then
		if [[ "$installation_status" == "before-installation" ]]; then
			EXE="$BUILD_FOLDER"/flexflow_python
		else
			EXE="flexflow_python"
		fi
		echo "Running a single-GPU Python test to check the Python interface (flexflow_python interpreter)"
		$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
	else
		echo "Invalid Python interpreter"
		exit 1
	fi
}


FF_HOME="$(realpath "${BASH_SOURCE[0]%/*}/..")"
export FF_HOME
# Edit the folder below if you did not build FlexFlow in $FF_HOME/build
BUILD_FOLDER="${FF_HOME}/build"
export BUILD_FOLDER

installation_status=${1:-"before-installation"}
echo "Running Python interface tests (installation status: ${installation_status})"
if [[ "$installation_status" == "before-installation" ]]; then
	# Import flexflow.core module in Python
	export PYTHONPATH="${FF_HOME}/python:${BUILD_FOLDER}/deps/legion/bindings/python:${PYTHONPATH}"
	export LD_LIBRARY_PATH="${BUILD_FOLDER}:${LD_LIBRARY_PATH}"
	python -c "import flexflow.core; exit()"
	unset PYTHONPATH
	unset LD_LIBRARY_PATH
	# Run a single-gpu test using the flexflow_python interpreter
	check_python_interface flexflow_python
	# Run a single-gpu test using the native python interpreter
	export LD_LIBRARY_PATH="${BUILD_FOLDER}:${BUILD_FOLDER}/deps/legion/lib:${LD_LIBRARY_PATH}"
	export PYTHONPATH="${FF_HOME}/python:${BUILD_FOLDER}/deps/legion/bindings/python:${PYTHONPATH}"
	check_python_interface python
	unset PYTHONPATH
	unset LD_LIBRARY_PATH
elif [[ "$installation_status" == "after-installation" ]]; then
	# Import flexflow.core module in Python
	python -c "import flexflow.core; exit()"
	# Run a single-gpu test using the flexflow_python interpreter
	check_python_interface flexflow_python after-installation
	# Run a single-gpu test using the native python interpreter
	check_python_interface python
else
	echo "Invalid installation status!"
	echo "Usage: $0 {before-installation, after-installation}"
	exit 1
fi
