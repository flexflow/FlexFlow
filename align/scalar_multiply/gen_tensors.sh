#! /usr/bin/env bash

eval "$(conda shell.bash hook)";
rm align/scalar_multiply/out/*.pt;
conda activate flexflow;
./python/flexflow_python align/scalar_multiply/align_scalar_multiply_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16;
conda activate pytorch;
python align/scalar_multiply/align_scalar_multiply_torch.py;
