#!/bin/bash

export GASNET=${PWD}/GASNet-2019.9.0
export LEGION=${PWD}/legion
export PROTOBUF=${PWD}/protobuf

module unload cuda cudnn NCCL

#cuda v10
#module load cuda/10.0
#module load cudnn/v7.6-cuda.10.0
#module load NCCL/2.4.8-1-cuda.10.0
#export CUDA=/public/apps/cuda/10.1
#export CUDNN=/public/apps/cudnn/v7.6/cuda
#export NCCL=/public/apps/NCCL/2.4.8-1

#cuda v9.2
module load cuda/9.2
module load cudnn/v7.3-cuda.9.2
module load NCCL/2.2.13-1-cuda.9.2
export CUDA=/public/apps/cuda/9.2
export CUDNN=/public/apps/cudnn/v7.3/cuda
export NCCL=/public/apps/NCCL/2.2.13-1

module load cmake/3.15.3/gcc.7.3.0
module load anaconda3/2019.07

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PROTOBUF/src/.libs
export PATH=$PATH:$PROTOBUF
