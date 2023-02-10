#! /usr/bin/env bash
eval "$(conda shell.bash hook)";

rm -rf align/out

function generate_ff_tensor(){
    ./python/flexflow_python tests/align/align_create_tensor_ff.py -ll:py 1 -ll:gpu "$1" -ll:fsize "$2" -ll:zsize "$3" -b "$4" -o "$5";
}

function generate_torch_tensor(){
    generate_torch_tensor "$1";
}

#create flexflow tensors
conda activate flexflow;
generate_ff_tensor 1 5000 4096 16 add
generate_ff_tensor 1 5000 4096 16 concat
generate_ff_tensor 1 5000 4096 16 conv2d
generate_ff_tensor 1 5000 4096 16 cos
generate_ff_tensor 1 5000 4096 16 embedding
generate_ff_tensor 1 5000 4096 16 exp
generate_ff_tensor 1 5000 4096 16 flat
generate_ff_tensor 1 5000 4096 16 getitem
generate_ff_tensor 1 5000 4096 16 identity
generate_ff_tensor 1 5000 4096 16 multiply
generate_ff_tensor 1 5000 4096 16 pool2d
generate_ff_tensor 1 5000 4096 16 reducesum
generate_ff_tensor 1 5000 4096 16 relu
generate_ff_tensor 1 5000 4096 16 reshape
generate_ff_tensor 1 5000 4096 16 scalar_add
generate_ff_tensor 1 5000 4096 16 scalar_multiply
generate_ff_tensor 1 5000 4096 16 scalar_sub
generate_ff_tensor 1 5000 4096 16 scalar_truediv
generate_ff_tensor 1 5000 4096 16 sigmoid
generate_ff_tensor 1 5000 4096 16 sin
generate_ff_tensor 1 5000 4096 16 subtract
generate_ff_tensor 1 5000 4096 16 tanh
generate_ff_tensor 1 5000 4096 16 transpose
generate_ff_tensor 1 5000 4096 16 view_embedding
generate_ff_tensor 1 5000 4096 16 max
generate_ff_tensor 1 5000 4096 16 min
generate_ff_tensor 1 5000 4096 16 linear


#create torch tensorss
conda activate pytorch;
generate_torch_tensor add;
generate_torch_tensor concat;
generate_torch_tensor conv2d;
generate_torch_tensor cos;
generate_torch_tensor embedding;
generate_torch_tensor exp;
generate_torch_tensor flat;
generate_torch_tensor getitem;
generate_torch_tensor identity;
generate_torch_tensor multiply;
generate_torch_tensor pool2d;
generate_torch_tensor reducesum;
generate_torch_tensor relu;
generate_torch_tensor reshape;
generate_torch_tensor scalar_add;
generate_torch_tensor scalar_multiply;
generate_torch_tensor scalar_sub;
generate_torch_tensor scalar_truediv;
generate_torch_tensor sigmoid;
generate_torch_tensor sin;
generate_torch_tensor subtract;
generate_torch_tensor tanh;
generate_torch_tensor transpose;
generate_torch_tensor view_embedding;
generate_torch_tensor max;
generate_torch_tensor min;
generate_torch_tensor linear;

conda activate flexflow;
python -m pytest tests/align/align_test.py