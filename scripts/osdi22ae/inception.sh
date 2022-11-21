#! /usr/bin/env bash

echo "Running Inception-v3 with a parallelization strategy discovered by Unity"
"$FF_HOME"/build/examples/cpp/InceptionV3/inception -ll:gpu 4 -ll:fsize 11000 -ll:zsize 14000 -b 64 --budget 10

echo "Running Inception-v3 with data parallelism"
"$FF_HOME"/build/examples/cpp/InceptionV3/inception -ll:gpu 4 -ll:fsize 11000 -ll:zsize 14000 -b 64 --budget 10 --only-data-parallel
