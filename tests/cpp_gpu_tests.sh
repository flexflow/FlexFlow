#! /usr/bin/env bash
set -x
set -e

GPUS=$1
BATCHSIZE=$((GPUS * 64))
FSIZE=14048
ZSIZE=12192

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi
BUILD_FOLDER=${BUILD_FOLDER:-"build"}

"$FF_HOME/$BUILD_FOLDER"/examples/cpp/AlexNet/alexnet -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
# TODO: fix DLRM test
#"$FF_HOME/$BUILD_FOLDER"/examples/cpp/DLRM/dlrm -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
"$FF_HOME/$BUILD_FOLDER"/examples/cpp/InceptionV3/inception -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
"$FF_HOME/$BUILD_FOLDER"/examples/cpp/MLP_Unify/mlp_unify -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
"$FF_HOME/$BUILD_FOLDER"/examples/cpp/ResNet/resnet -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
"$FF_HOME/$BUILD_FOLDER"/examples/cpp/Transformer/transformer -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b $((GPUS * 8)) --only-data-parallel
"$FF_HOME/$BUILD_FOLDER"/examples/cpp/XDL/xdl -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
"$FF_HOME/$BUILD_FOLDER"/examples/cpp/candle_uno/candle_uno -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
# TODO: fix MoE dataloader segfault
#"$FF_HOME/$BUILD_FOLDER"/examples/cpp/mixture_of_experts/moe -ll:gpu 1 -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b 64 --only-data-parallel
"$FF_HOME/$BUILD_FOLDER"/examples/cpp/resnext50/resnext50 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
"$FF_HOME/$BUILD_FOLDER"/examples/cpp/split_test/split_test -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
"$FF_HOME/$BUILD_FOLDER"/examples/cpp/split_test_2/split_test_2 -ll:gpu "$GPUS" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b ${BATCHSIZE} --only-data-parallel
