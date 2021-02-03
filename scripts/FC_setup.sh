git submodule update --init --recursive
source FC_env_setup.sh

cd $PROTOBUF
git submodule update --init --recursive
##git checkout 6d4e7fd #still cannot get the strategy compile to use the local runtime. So need to checkout v 3.10.0
./autogen.sh
./configure
make -j
cd ..

cd $GASNET
./FC.build_script.sh
cd ..

cd src/runtime
../../protobuf/src/protoc --cpp_out=. strategy.proto
./gen_strategy.sh 8 8 1 # for 8 gpu per node,  and 8 embeddings per node, and 1 node
cd ../..

cd $LEGION
git checkout control_replication
cd ../


make app=examples/DLRM/dlrm -j
cd examples/DLRM
./run_random.sh 1 