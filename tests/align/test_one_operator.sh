#! /usr/bin/env bash
eval "$(conda shell.bash hook)";
operator=$1
rm tests/align/out/$operator/*.pt;
conda activate flexflow;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o $operator;
conda activate pytorch;
python tests/align/align_create_tensor_torch.py -o $operator;

conda activate flexflow;
python -m pytest tests/align/align_test.py::test_$operator
