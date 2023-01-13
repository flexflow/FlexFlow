#!/bin/bash

numgpu="$1"
numnodes="$2"
per_gpu_batch_size=1024
totalgpus=$((numgpu * numnodes))
batchsize=$((totalgpus * per_gpu_batch_size))

# 2 Embedding Tables
#LEGION_FREEZE_ON_ERROR=1 jsrun -n ${numnodes} -a 1 -r 1 -c 24 -g 6 --bind rs ./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 15000 -ll:zsize 20000 -ll:util 6 -ll:bgwork 12 --arch-sparse-feature-size 256 --arch-embedding-size 1000000-1000000 --arch-mlp-bot 2048-2048-2048-2048-2048-2048-2048-2048-2048 --arch-mlp-top 4096-4096-4096-4096-4096 --epochs 100 --batch-size ${batchsize} --nodes ${numnodes} --control-replication #--import $MEMBERWORK/csc335/dlrm_profiles/dlrm.strategy #-lg:prof ${numnodes} -lg:prof_logfile $MEMBERWORK/csc335/dlrm_profiles/prof_%.gz

# 8 Embedding Tables
#LEGION_FREEZE_ON_ERROR=1 jsrun -n ${numnodes} -a 1 -r 1 -c 24 -g 6 --bind rs ./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 15000 -ll:zsize 20000 -ll:util 6 -ll:bgwork 12 --arch-sparse-feature-size 256 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 2048-2048-2048-2048-2048-2048-2048-2048-2048 --arch-mlp-top 4096-4096-4096-4096-4096 --epochs 100 --batch-size ${batchsize} --nodes ${numnodes} --control-replication --budget 1000 --simulator-workspace-size 5093657088 #--import $MEMBERWORK/csc335/dlrm_profiles/dlrm.strategy #-lg:prof ${numnodes} -lg:prof_logfile $MEMBERWORK/csc335/dlrm_profiles/prof_%.gz
LEGION_FREEZE_ON_ERROR=1 jsrun -n "${numnodes}" -a 1 -r 1 -c 24 -g 6 --bind rs ./dlrm -ll:gpu "${numgpu}" -ll:cpu 1 -ll:fsize 15000 -ll:zsize 20000 -ll:util 6 -ll:bgwork 12 --arch-sparse-feature-size 256 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 1024-1024-1024-1024-1024-1024-1024-1024-1024 --arch-mlp-top 1024-1024-1024-1024-1024 --epochs 100 --batch-size "${batchsize}" --nodes "${numnodes}" --control-replication --budget 1000 --simulator-workspace-size 5093657088 #--import $MEMBERWORK/csc335/dlrm_profiles/dlrm.strategy #-lg:prof ${numnodes} -lg:prof_logfile $MEMBERWORK/csc335/dlrm_profiles/prof_%.gz

# 24 Embedding Tables
#GASNET_BACKTRACE=1 jsrun -n ${numnodes} -a 1 -r 1 -c 24 -g 6 --bind rs ./dlrm -ll:gpu ${numgpu} -ll:cpu 1 -ll:fsize 14000 -ll:zsize 20000 -ll:util 12 -ll:dma 4 --arch-sparse-feature-size 64 --arch-embedding-size 1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000 --arch-mlp-bot 64-512-512-64 --arch-mlp-top 576-1024-1024-1024-1 --epochs 5 --batch-size ${batchsize} -dm:memoize --nodes ${numnodes} -lg:prof ${numnodes} -lg:prof_logfile $MEMBERWORK/csc335/dlrm_profiles/prof_%.gz

#--strategy ../../src/runtime/dlrm_strategy_${totalgpus}gpus.pb

