#! /usr/bin/env bash

eval "$(conda shell.bash hook)";
rm align/add/out/*.pt;
conda activate flexflow;
./python/flexflow_python align/add/align_add_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16;
conda activate pytorch;
python align/add/align_add_torch.py;

