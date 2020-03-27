#!/bin/bash

per_gpu_batch_size=256
numgpu="$1"
batchsize=$((numgpu * per_gpu_batch_size))

./dlrm -ll:gpu ${numgpu} -ll:cpu 4 -ll:fsize 12000 -ll:zsize 20000 -ll:util ${numgpu} --arch-sparse-feature-size 64 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 64-512-512-64 --arch-mlp-top 576-1024-1024-1024-1 --epochs 20 --batch-size ${batchsize} -dm:memoize --strategy ../../src/runtime/dlrm_strategy_8nEmb_${numgpu}cpu_${numgpu}gpu.pb
