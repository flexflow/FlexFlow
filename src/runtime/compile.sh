#!/bin/bash

g++ dlrm_strategy.cc strategy.pb.cc -o generator -std=c++11 -lprotobuf -L/usr/local/lib -I/usr/local/include -I${PROTOBUF}/src -pthread -O2
g++ dlrm_strategy_hetero.cc strategy.pb.cc -o generator_hetero -std=c++11 -lprotobuf -L/usr/local/lib -I/usr/local/include -I${PROTOBUF}/src -pthread -O2
