#! /usr/bin/env bash

echo "Running XDL with a parallelization strategy discovered by Unity"
"$FF_HOME"/build/examples/cpp/XDL/xdl -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 --budget 20

echo "Running XDL with data parallelism"
"$FF_HOME"/build/examples/cpp/XDL/xdl -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 --budget 20 --only-data-parallel
