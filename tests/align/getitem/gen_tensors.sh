#! /usr/bin/env bash

eval "$(conda shell.bash hook)";
rm align/getitem/out/*.pt;
conda activate flexflow;
./python/flexflow_python align/getitem/align_getitem_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16;
conda activate pytorch;
python align/getitem/align_getitem_torch.py;
