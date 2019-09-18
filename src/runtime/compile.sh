#!/bin/bash

g++ dlrm_strategy.cc strategy.pb.cc -o generator -std=c++11 -lprotobuf -L/usr/local/lib -I/usr/local/include -pthread -O2
