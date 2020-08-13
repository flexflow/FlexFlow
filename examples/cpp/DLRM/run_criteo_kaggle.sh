#!/bin/bash

per_gpu_batch_size=256
numgpu="$1"
batchsize=$((numgpu * per_gpu_batch_size))
dataset="$2"

./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 12000 -ll:zsize 20000 --arch-sparse-feature-size 16 --arch-embedding-size 1396-550-1761917-507795-290-21-11948-608-3-58176-5237-1497287-3127-26-12153-1068715-10-4836-2085-4-1312273-17-15-110946-91-72655 --arch-mlp-bot 13-512-256-64-16 --arch-mlp-top 224-512-256-1 --dataset ${dataset} --epochs 100 --batch-size ${batchsize}
