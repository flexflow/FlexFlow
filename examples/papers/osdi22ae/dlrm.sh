#! /usr/bin/env bash

echo "Running DLRM with a parallelization strategy discovered by Unity"
"$FF_HOME"/build/examples/cpp/DLRM/dlrm -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 --budget 20

echo "Running DLRM with data parallelism"
"$FF_HOME"/build/examples/cpp/DLRM/dlrm -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 --budget 20 --only-data-parallel
