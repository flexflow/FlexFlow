#! /usr/bin/env bash

set -x
set -e

usage() {
  >&2 echo "Usage: $0 <num-gpus>" 
  exit 1
}

if [[ $# -ne 1 ]]; then 
  usage
fi

GPUS=$1
BATCHSIZE=$((GPUS * 64))

if [[ -z "$FF_HOME" ]]; then 
  >&2 echo "FF_HOME variable is not defined, aborting tests"
  exit 1 
fi
EXE="gdb -ex r --args $FF_HOME/python/flexflow_python"

#Accuracy tests
#$EXE $FF_HOME/examples/python/native/mnist_mlp.py -a -ll:py 1 -ll:gpu $GPUS -ll:fsize 12000 -ll:zsize 12192 --epochs 5 -b ${BATCHSIZE}
$EXE $FF_HOME/examples/python/native/mnist_cnn.py -a -ll:py 1 -ll:gpu $GPUS -ll:fsize 12000 -ll:zsize 12192 --epochs 5 -b ${BATCHSIZE}
#$EXE $FF_HOME/examples/python/native/cifar10_cnn.py -a -ll:py 1 -ll:gpu $GPUS -ll:fsize 12000 -ll:zsize 12192 --epochs 40
#$EXE $FF_HOME/examples/python/native/alexnet.py -a -ll:py 1 -ll:gpu $GPUS -ll:fsize 12000 -ll:zsize 12192 --epochs 40

