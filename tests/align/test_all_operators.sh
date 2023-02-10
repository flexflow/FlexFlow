#! /usr/bin/env bash
eval "$(conda shell.bash hook)";

rm -rf align/out

#create flexflow tensors
conda activate flexflow;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o add;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o concat;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o conv2d;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o cos;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o embedding;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o exp;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o flat;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o getitem;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o identity;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o multiply;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o pool2d;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o reducesum;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o relu;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o reshape;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o scalar_add;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o scalar_multiply;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o scalar_sub;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o scalar_truediv;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o sigmoid;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o sin;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o subtract;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o tanh;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o transpose;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o view_embedding;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o max;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o min;
./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o linear;

#create torch tensorss
conda activate pytorch;
python tests/align/align_create_tensor_torch.py -o add;
python tests/align/align_create_tensor_torch.py -o concat;
python tests/align/align_create_tensor_torch.py -o conv2d;
python tests/align/align_create_tensor_torch.py -o cos;
python tests/align/align_create_tensor_torch.py -o embedding;
python tests/align/align_create_tensor_torch.py -o exp;
python tests/align/align_create_tensor_torch.py -o flat;
python tests/align/align_create_tensor_torch.py -o getitem;
python tests/align/align_create_tensor_torch.py -o identity;
python tests/align/align_create_tensor_torch.py -o multiply;
python tests/align/align_create_tensor_torch.py -o pool2d;
python tests/align/align_create_tensor_torch.py -o reducesum;
python tests/align/align_create_tensor_torch.py -o relu;
python tests/align/align_create_tensor_torch.py -o reshape;
python tests/align/align_create_tensor_torch.py -o scalar_add;
python tests/align/align_create_tensor_torch.py -o scalar_multiply;
python tests/align/align_create_tensor_torch.py -o scalar_sub;
python tests/align/align_create_tensor_torch.py -o scalar_truediv;
python tests/align/align_create_tensor_torch.py -o sigmoid;
python tests/align/align_create_tensor_torch.py -o sin;
python tests/align/align_create_tensor_torch.py -o subtract;
python tests/align/align_create_tensor_torch.py -o tanh;
python tests/align/align_create_tensor_torch.py -o transpose;
python tests/align/align_create_tensor_torch.py -o view_embedding;
python tests/align/align_create_tensor_torch.py -o max;
python tests/align/align_create_tensor_torch.py -o min;
python tests/align/align_create_tensor_torch.py -o linear;

conda activate flexflow;
python -m pytest tests/align/align_test.py