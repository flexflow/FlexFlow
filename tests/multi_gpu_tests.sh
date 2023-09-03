#! /usr/bin/env bash
set -x
set -e

# Default to single-node, single GPU
GPUS=${1:-1} # number of GPUS per node
NUM_NODES=${2:-1} # number of nodes
BATCHSIZE=$(( NUM_NODES * GPUS * 64))
FSIZE=13800
ZSIZE=12192
ONLY_DATA_PARALLEL=true

FF_HOME="$(realpath "${BASH_SOURCE[0]%/*}/..")"
export FF_HOME
# Edit the folder below if you did not build FlexFlow in $FF_HOME/build
BUILD_FOLDER="${FF_HOME}/build"
export BUILD_FOLDER

if [[ $NUM_NODES -gt 1 ]]; then
    export GPUS
    export NUM_NODES
    EXE="$FF_HOME"/tests/multinode_helpers/mpi_wrapper1.sh
else
    if [[ -f "$BUILD_FOLDER/flexflow_python" ]]; then
        EXE="$BUILD_FOLDER"/flexflow_python
    else
        EXE="flexflow_python"
    fi
fi

# Check that number of GPUs requested is available
echo "Running GPU tests with $NUM_NODES node(s) and $GPUS gpu(s)/node"
GPU_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_REQUESTED=$(( GPUS * NUM_NODES))
if [ $GPU_REQUESTED -gt $(( GPU_AVAILABLE )) ]; then echo "The test requires $GPU_REQUESTED GPUs, but only $GPU_AVAILABLE are available. Try reducing the number of nodes, or the number of gpus/node." ; exit; fi

# Generate configs JSON files
test_params=$(jq -n --arg num_gpus "$GPUS" --arg memory_per_gpu "$FSIZE" --arg zero_copy_memory_per_node "$ZSIZE" --arg batch_size "$BATCHSIZE" --arg only_data_parallel "$ONLY_DATA_PARALLEL" '{"num_gpus":$num_gpus,"memory_per_gpu":$memory_per_gpu,"zero_copy_memory_per_node":$zero_copy_memory_per_node,"batch_size":$batch_size,"only_data_parallel":$only_data_parallel}')
test_params_5_epochs=$(echo "$test_params" | jq '. + {"epochs": 5}')
test_params_40_epochs=$(echo "$test_params" | jq '. + {"epochs": 40}')
test_params_5_epochs_no_batch_size=$(echo "$test_params_5_epochs" | jq 'del(.batch_size)')
mkdir -p /tmp/flexflow/multi_gpu_tests
echo "$test_params" > /tmp/flexflow/multi_gpu_tests/test_params.json
echo "$test_params_5_epochs" > /tmp/flexflow/multi_gpu_tests/test_params_5_epochs.json
echo "$test_params_40_epochs" > /tmp/flexflow/multi_gpu_tests/test_params_40_epochs.json
echo "$test_params_5_epochs_no_batch_size" > /tmp/flexflow/multi_gpu_tests/test_params_5_epochs_no_batch_size.json

#Sequential model tests
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_cnn.py /tmp/flexflow/multi_gpu_tests/test_params.json
#$EXE "$FF_HOME"/examples/python/keras/seq_reuters_mlp.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/seq_cifar10_cnn.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp_net2net.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_cnn_net2net.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_cnn_nested.py /tmp/flexflow/multi_gpu_tests/test_params.json

#Keras other
$EXE "$FF_HOME"/examples/python/keras/callback.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/unary.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/reshape.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/elementwise_mul_broadcast.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/reduce_sum.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/identity_loss.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/elementwise_max_min.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/rsqrt.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/gather.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/regularizer.py /tmp/flexflow/multi_gpu_tests/test_params.json

#Functional API
$EXE "$FF_HOME"/examples/python/keras/func_mnist_mlp.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_mnist_mlp_concat.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_mnist_mlp_concat2.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_mnist_cnn.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_mnist_cnn_concat.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_nested.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_alexnet.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_mnist_mlp_net2net.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_net2net.py /tmp/flexflow/multi_gpu_tests/test_params.json

#Python
$EXE "$FF_HOME"/examples/python/native/print_layers.py /tmp/flexflow/multi_gpu_tests/test_params_5_epochs.json
$EXE "$FF_HOME"/examples/python/native/split.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/native/alexnet.py /tmp/flexflow/multi_gpu_tests/test_params_40_epochs.json
$EXE "$FF_HOME"/examples/python/native/mnist_mlp.py /tmp/flexflow/multi_gpu_tests/test_params_5_epochs.json
$EXE "$FF_HOME"/examples/python/native/mnist_cnn.py /tmp/flexflow/multi_gpu_tests/test_params_5_epochs.json
$EXE "$FF_HOME"/examples/python/native/cifar10_cnn.py /tmp/flexflow/multi_gpu_tests/test_params_40_epochs.json
$EXE "$FF_HOME"/examples/python/native/cifar10_cnn_attach.py /tmp/flexflow/multi_gpu_tests/test_params_5_epochs_no_batch_size.json
$EXE "$FF_HOME"/examples/python/native/mnist_mlp_attach.py /tmp/flexflow/multi_gpu_tests/test_params_5_epochs_no_batch_size.json

#Possible crash
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_concat.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_concat_model.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_concat_seq_model.py /tmp/flexflow/multi_gpu_tests/test_params.json
$EXE "$FF_HOME"/examples/python/native/cifar10_cnn_concat.py /tmp/flexflow/multi_gpu_tests/test_params_40_epochs.json
