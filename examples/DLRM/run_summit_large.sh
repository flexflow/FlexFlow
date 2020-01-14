#!/bin/bash

numgpu="$1"
numnodes="$2"
per_gpu_batch_size=256
totalgpus=$((numgpu * numnodes))
batchsize=$((totalgpus * per_gpu_batch_size))

# 24 Embedding Tables
GASNET_BACKTRACE=1 jsrun -n ${numnodes} -a 1 -r 1 -c 24 -g 6 --bind rs ./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 14000 -ll:zsize 20000 -ll:util 6 -ll:dma 4 --embedding-bag-size 100 --arch-sparse-feature-size 64 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 2048-4096-4096-4096-4096-4096 --arch-mlp-top 10240-4096-4096-4096-4096-1 --epochs 5 --batch-size ${batchsize} -dm:memorize --nodes ${numnodes} -lg:prof ${numnodes} -lg:prof_logfile $MEMBERWORK/csc335/dlrm_profiles/prof_%.gz --strategy ../../src/runtime/dlrm_strategy_${totalgpus}gpus.pb 

# 6 Embedding Tables
#GASNET_BACKTRACE=1 jsrun -n ${numnodes} -a 1 -r 1 -c 24 -g 6 --bind rs ./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 14000 -ll:zsize 20000 -ll:util 6 -ll:dma 4 --embedding-bag-size 100 --arch-sparse-feature-size 64 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 2048-4098-4098-4098-4098-4098 --arch-mlp-top 4480-4098-4098-4098-4098-4098-4098-4098-4098-4098-4098 --epochs 5 --batch-size ${batchsize} -dm:memorize --nodes ${numnodes} -lg:prof ${numnodes} -lg:prof_logfile $MEMBERWORK/csc335/dlrm_profiles/prof_%.gz --strategy ../../src/runtime/dlrm_strategy_${totalgpus}gpus.pb 

# 1 Embedding Table
#GASNET_BACKTRACE=1 jsrun -n ${numnodes} -a 1 -r 1 -c 24 -g 6 --bind rs ./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 14000 -ll:zsize 20000 -ll:util 6 -ll:dma 4 --embedding-bag-size 100 --arch-sparse-feature-size 64 --arch-embedding-size 10000000 --arch-mlp-bot 2048-4096-4096-4096-4096-4096 --arch-mlp-top 4160-4096-4096-4096-4096-4096-4096-4096-4096-4096-1 --epochs 5 --batch-size ${batchsize} -dm:memorize --nodes ${numnodes}
