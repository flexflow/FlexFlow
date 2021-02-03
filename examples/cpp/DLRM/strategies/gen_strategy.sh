#!/bin/bash

if [ $# -ne 3 ]
then
	echo "Need 3 arguments (per-node-gpu per-node_emb num_nodes)"
	exit
fi

pngpus=$1
pnembs=$2
nnodes=$3

ngpus=$((pngpus * nnodes))
nembs=$((pnembs * nnodes))

#python3 dlrm_strategy.py -f dlrm_strategy.cc -g ${ngpus} -e ${nembs}

echo "Compile..."
g++ dlrm_strategy.cc strategy.pb.cc -o generator -std=c++11 -L${PROTOBUF}/src/.libs -lprotobuf -L/usr/local/lib -I/usr/local/include -I${PROTOBUF}/src -pthread -O2

echo "Generate..."
./generator --gpu ${pngpus} --emb ${pnembs} --node ${nnodes}

echo "Done. dlrm_strategy_emb_${pnembs}_gpu_${pngpus}_node_${nnodes}.pb"

