#! /usr/bin/env bash

eval "$(conda shell.bash hook)";
rm align/conv2d/out/*.pt;
conda activate flexflow;
./python/flexflow_python align/conv2d/align_conv2d_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16;
conda activate pytorch;
python align/conv2d/align_conv2d_torch.py;

