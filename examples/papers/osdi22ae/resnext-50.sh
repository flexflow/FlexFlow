#! /usr/bin/env bash

echo "Running ResNeXt-50 with a parallelization strategy discovered by Unity"
"$FF_HOME"/build/examples/cpp/resnext50/resnext50 -ll:gpu 4 -ll:fsize 12000 -ll:zsize 14000 -b 16 --budget 20

echo "Running ResNeXt-50 with data parallelism"
"$FF_HOME"/build/examples/cpp/resnext50/resnext50 -ll:gpu 4 -ll:fsize 12000 -ll:zsize 14000 -b 16 --budget 20 --only-data-parallel
