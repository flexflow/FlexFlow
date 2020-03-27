#!/bin/bash
cd src/runtime
protoc --cpp_out=. strategy.proto
cd ../..
