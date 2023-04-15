#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate flexflow
~/FlexFlow/python/flexflow_python ~/FlexFlow/examples/python/native/mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 8000 -ll:zsize 8000