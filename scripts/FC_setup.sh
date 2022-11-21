#! /usr/bin/env bash

# Cd into FF_HOME
cd "${BASH_SOURCE[0]%/*}/../"

git submodule update --init --recursive
./scripts/FC_env_setup.sh

cd $PROTOBUF || exit
git submodule update --init --recursive
##git checkout 6d4e7fd #still cannot get the strategy compile to use the local runtime. So need to checkout v 3.10.0
./autogen.sh
./configure
make -j
cd .. || exit

cd $GASNET || exit
./FC.build_script.sh
cd .. || exit

cd src/runtime || exit
../../protobuf/src/protoc --cpp_out=. strategy.proto
./gen_strategy.sh 8 8 1 # for 8 gpu per node,  and 8 embeddings per node, and 1 node
cd ../.. || exit

cd $LEGION || exit
git checkout control_replication
cd ../ || exit


make app=examples/DLRM/dlrm -j
cd examples/DLRM || exit
./run_random.sh 1 