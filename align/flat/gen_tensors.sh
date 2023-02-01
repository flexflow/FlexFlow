#! /usr/bin/env bash

eval "$(conda shell.bash hook)";
rm align/flat/out/*.pt;
conda activate flexflow;
./python/flexflow_python align/flat/align_flat_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16;
conda activate pytorch;
python align/flat/align_flat_torch.py;
