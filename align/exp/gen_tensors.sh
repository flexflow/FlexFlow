#! /usr/bin/env bash

eval "$(conda shell.bash hook)";
rm align/exp/out/*.pt;
conda activate flexflow;
./python/flexflow_python align/exp/align_exp_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16;
conda activate pytorch;
python align/exp/align_exp_torch.py;
