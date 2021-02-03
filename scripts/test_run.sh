# git checkout dcr # We are using the dcr branch by default
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
./gen_strategy.sh 2 1 1 # for 2 gpu per node, testing purpose
cd ../..

cd $LEGION
git checkout control_replication
cd ../


make app=src/ops/tests/concat_test -j -f Makefile
cd src/ops/tests 
./test_run_FF_target.sh concat_test 2 && cp output.txt output_2gpus.txt
./test_run_FF_target.sh concat_test 1 && cp output.txt output_1gpus.txt

