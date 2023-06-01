#! /usr/bin/env bash
set -x
set -e

# Default to single-node, single GPU
GPUS=${1:-1} # number of GPUS per node
NUM_NODES=${2:-1} # number of nodes
BATCHSIZE=$(( NUM_NODES * GPUS * 64))
FSIZE=13800
ZSIZE=12192

FF_HOME="$(realpath "${BASH_SOURCE[0]%/*}/..")"
export FF_HOME

if [[ $NUM_NODES -gt 1 ]]; then
    export GPUS
    export NUM_NODES
    EXE="$FF_HOME"/tests/multinode_helpers/mpi_wrapper1.sh
else
    EXE="$FF_HOME"/python/legion_python
fi

echo "Running GPU tests with $NUM_NODES node(s) and $GPUS gpu(s)/node"
GPU_AVAILABLE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
GPU_REQUESTED=$(( GPUS * NUM_NODES))
if [ $GPU_REQUESTED -gt $(( GPU_AVAILABLE )) ]; then echo "The test requires $GPU_REQUESTED GPUs, but only $GPU_AVAILABLE are available. Try reducing the number of nodes, or the number of gpus/node." ; exit; fi

#Sequential model tests
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_cnn.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/seq_reuters_mlp.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/seq_cifar10_cnn.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_mlp_net2net.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_cnn_net2net.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/seq_mnist_cnn_nested.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel

#Keras other
$EXE "$FF_HOME"/examples/python/keras/callback.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/unary.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/reshape.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/elementwise_mul_broadcast.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/reduce_sum.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/identity_loss.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel 
$EXE "$FF_HOME"/examples/python/keras/elementwise_max_min.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel 
$EXE "$FF_HOME"/examples/python/keras/rsqrt.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/gather.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/regularizer.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel

#Functional API
$EXE "$FF_HOME"/examples/python/keras/func_mnist_mlp.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_mnist_mlp_concat.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_mnist_mlp_concat2.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_mnist_cnn.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_mnist_cnn_concat.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_nested.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_alexnet.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_mnist_mlp_net2net.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_net2net.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel

#Python
$EXE "$FF_HOME"/examples/python/native/print_layers.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --epochs 5 -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/native/split.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/native/alexnet.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --epochs 40 --only-data-parallel
$EXE "$FF_HOME"/examples/python/native/mnist_mlp.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --epochs 5 -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/native/mnist_cnn.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --epochs 5 -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/native/cifar10_cnn.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --epochs 40 --only-data-parallel
$EXE "$FF_HOME"/examples/python/native/cifar10_cnn_attach.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --epochs 5 --only-data-parallel
$EXE "$FF_HOME"/examples/python/native/mnist_mlp_attach.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --epochs 5 --only-data-parallel

#Possible crash
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_concat.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_concat_model.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/keras/func_cifar10_cnn_concat_seq_model.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
$EXE "$FF_HOME"/examples/python/native/cifar10_cnn_concat.py -ll:py 1 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" --epochs 40 --only-data-parallel
