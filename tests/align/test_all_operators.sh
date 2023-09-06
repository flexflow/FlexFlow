#! /usr/bin/env bash
eval "$(conda shell.bash hook)"

rm -rf align/out

function generate_ff_tensor(){
    ./build/flexflow_python tests/align/align_create_tensor_ff.py -ll:gpu 1 -ll:fsize 5000 -ll:zsize 4096 -b 16 -o "$1"
}

function generate_torch_tensor(){
    python tests/align/align_create_tensor_torch.py -o "$1"
}

ops=(add concat conv2d cos embedding exp flat getitem identity multiply pool2d reducesum relu reshape scalar_add scalar_multiply scalar_sub scalar_truediv sigmoid sin subtract tanh transpose view_embedding max min linear gather)

#create flexflow tensors
conda activate flexflow
conda info --envs
for(( i=0;i<${#ops[@]};i++)) 
do
    generate_ff_tensor "${ops[i]}";
done;

#create torch tensorss
conda activate pytorch
for(( i=0;i<${#ops[@]};i++)) 
do
    generate_torch_tensor "${ops[i]}";
done;

conda activate flexflow
python -m pytest tests/align/align_test.py
