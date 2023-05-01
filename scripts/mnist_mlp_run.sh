#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate flexflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
~/FlexFlow/python/flexflow_python ~/FlexFlow/examples/python/native/mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 8000 -ll:zsize 8000
