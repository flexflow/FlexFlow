#! /usr/bin/env bash

echo "Running BERT with a parallelization strategy discovered by Unity"
"$FF_HOME"/build/examples/cpp/Transformer/transformer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 -b 8 --budget 30

echo "Running BERT Uno with data parallelism"
"$FF_HOME"/build/examples/cpp/Transformer/transformer -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 -b 8 --budget 30 --only-data-parallel
