#! /usr/bin/env bash
set -x
set -e

# Parameters
GPUS=$1
NUM_NODES=2

BATCHSIZE=$((GPUS * 64))
FSIZE=14048
ZSIZE=12192

PYTHONPATH="${FF_HOME}/python"
FF_USE_NATIVE_PYTHON=1
EXE="python"

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi

if [[ $NUM_NODES > 1 ]]; then
    FSIZE=$(( FSIZE / GPUS ))
    ZSIZE=$(( FSIZE / GPUS ))
fi

function run_test() {
    FF_HOME="$FF_HOME" EXE="$EXE" PYTHONPATH="$PYTHONPATH" FF_USE_NATIVE_PYTHON="$FF_USE_NATIVE_PYTHON" GPUS_PER_NODE="$GPUS" FSIZE="$FSIZE" ZSIZE="$ZSIZE" BATCHSIZE="$BATCHSIZE" ONLY_DATA_PARALLEL="--only-data-parallel" ADDITIONAL_FLAGS="${2:-}" ./mpi_runner.sh $1
}

#Sequential model tests
run_test "$FF_HOME"/examples/python/keras/seq_mnist_mlp.py
run_test "$FF_HOME"/examples/python/keras/seq_mnist_cnn.py
run_test "$FF_HOME"/examples/python/keras/seq_reuters_mlp.py
run_test "$FF_HOME"/examples/python/keras/seq_cifar10_cnn.py
run_test "$FF_HOME"/examples/python/keras/seq_mnist_mlp_net2net.py
run_test "$FF_HOME"/examples/python/keras/seq_mnist_cnn_net2net.py
run_test "$FF_HOME"/examples/python/keras/seq_mnist_cnn_nested.py

#Keras other
run_test "$FF_HOME"/examples/python/keras/callback.py
run_test "$FF_HOME"/examples/python/keras/unary.py
run_test "$FF_HOME"/examples/python/keras/reshape.py

#Functional API
run_test "$FF_HOME"/examples/python/keras/func_mnist_mlp.py
run_test "$FF_HOME"/examples/python/keras/func_mnist_mlp_concat.py
run_test "$FF_HOME"/examples/python/keras/func_mnist_mlp_concat2.py
run_test "$FF_HOME"/examples/python/keras/func_mnist_cnn.py
run_test "$FF_HOME"/examples/python/keras/func_mnist_cnn_concat.py
run_test "$FF_HOME"/examples/python/keras/func_cifar10_cnn.py
run_test "$FF_HOME"/examples/python/keras/func_cifar10_cnn_nested.py
run_test "$FF_HOME"/examples/python/keras/func_cifar10_alexnet.py
run_test "$FF_HOME"/examples/python/keras/func_mnist_mlp_net2net.py
run_test "$FF_HOME"/examples/python/keras/func_cifar10_cnn_net2net.py

#Python
run_test "$FF_HOME"/examples/python/native/print_layers.py '--epochs 5'
run_test "$FF_HOME"/examples/python/native/split.py
run_test "$FF_HOME"/examples/python/native/alexnet.py '--epochs 40'
run_test "$FF_HOME"/examples/python/native/mnist_mlp.py '--epochs 5'
run_test "$FF_HOME"/examples/python/native/mnist_cnn.py '--epochs 5'
run_test "$FF_HOME"/examples/python/native/cifar10_cnn.py '--epochs 40'
run_test "$FF_HOME"/examples/python/native/cifar10_cnn_attach.py '--epochs 5'
run_test "$FF_HOME"/examples/python/native/mnist_mlp_attach.py '--epochs 5'

#Possible crash
run_test "$FF_HOME"/examples/python/keras/func_cifar10_cnn_concat.py
run_test "$FF_HOME"/examples/python/keras/func_cifar10_cnn_concat_model.py
run_test "$FF_HOME"/examples/python/keras/func_cifar10_cnn_concat_seq_model.py
run_test "$FF_HOME"/examples/python/native/cifar10_cnn_concat.py '--epochs 40'
