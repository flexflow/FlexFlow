#! /usr/bin/env bash
set -x
set -e

GPUS=$1
BATCHSIZE=$((GPUS * 64))
FSIZE=14048
ZSIZE=12192

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi
EXE="python"

#Sequantial model tests
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

