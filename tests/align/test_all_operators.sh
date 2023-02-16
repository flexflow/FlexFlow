#! /usr/bin/env bash
eval "$(conda shell.bash hook)"

rm -rf align/out

function generate_ff_tensor(){
    ./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o "$1"
}

function generate_torch_tensor(){
    python tests/align/align_create_tensor_torch.py -o "$1"
}

#create flexflow tensors
conda activate flexflow
conda info --envs
generate_ff_tensor add
generate_ff_tensor concat
generate_ff_tensor conv2d
generate_ff_tensor cos
generate_ff_tensor embedding
generate_ff_tensor exp
generate_ff_tensor flat
generate_ff_tensor getitem
generate_ff_tensor identity
generate_ff_tensor multiply
generate_ff_tensor pool2d
generate_ff_tensor reducesum
generate_ff_tensor relu
generate_ff_tensor reshape
generate_ff_tensor scalar_add
generate_ff_tensor scalar_multiply
generate_ff_tensor scalar_sub
generate_ff_tensor scalar_truediv
generate_ff_tensor sigmoid
generate_ff_tensor sin
generate_ff_tensor subtract
generate_ff_tensor tanh
generate_ff_tensor transpose
generate_ff_tensor view_embedding
generate_ff_tensor max
generate_ff_tensor min
generate_ff_tensor linear


#create torch tensorss
conda activate pytorch
generate_torch_tensor add
generate_torch_tensor concat
generate_torch_tensor conv2d
generate_torch_tensor cos
generate_torch_tensor embedding
generate_torch_tensor exp
generate_torch_tensor flat
generate_torch_tensor getitem
generate_torch_tensor identity
generate_torch_tensor multiply
generate_torch_tensor pool2d
generate_torch_tensor reducesum
generate_torch_tensor relu
generate_torch_tensor reshape
generate_torch_tensor scalar_add
generate_torch_tensor scalar_multiply
generate_torch_tensor scalar_sub
generate_torch_tensor scalar_truediv
generate_torch_tensor sigmoid
generate_torch_tensor sin
generate_torch_tensor subtract
generate_torch_tensor tanh
generate_torch_tensor transpose
generate_torch_tensor view_embedding
generate_torch_tensor max
generate_torch_tensor min
generate_torch_tensor linear

conda activate flexflow
python -m pytest tests/align/align_test.py