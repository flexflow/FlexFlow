#!/bin/bash

numgpu="$1"
numnodes="$2"
per_gpu_batch_size=512
totalgpus=$((numgpu * numnodes))
batchsize=$((totalgpus * per_gpu_batch_size))

# 8 Embedding Tables
GASNET_BACKTRACE=1 jsrun -n ${numnodes} -a 1 -r 1 -c 24 -g 6 --bind rs ./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 14000 -ll:zsize 20000 -ll:util 12 -ll:dma 4 --arch-sparse-feature-size 64 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 64-512-512-64 --arch-mlp-top 576-1024-1024-1024-1 --epochs 5 --batch-size ${batchsize} -dm:memoize --nodes ${numnodes} -lg:prof ${numnodes} -lg:prof_logfile $MEMBERWORK/csc335/dlrm_profiles/prof_%.gz

# 24 Embedding Tables
#GASNET_BACKTRACE=1 jsrun -n ${numnodes} -a 1 -r 1 -c 24 -g 6 --bind rs ./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 14000 -ll:zsize 20000 -ll:util 12 -ll:dma 4 --arch-sparse-feature-size 64 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 64-512-512-64 --arch-mlp-top 576-1024-1024-1024-1 --epochs 5 --batch-size ${batchsize} -dm:memoize --nodes ${numnodes} -lg:prof ${numnodes} -lg:prof_logfile $MEMBERWORK/csc335/dlrm_profiles/prof_%.gz

#--strategy ../../src/runtime/dlrm_strategy_${totalgpus}gpus.pb

