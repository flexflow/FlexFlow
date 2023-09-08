#! /usr/bin/env bash

echo "Running MLP with a parallelization strategy discovered by Unity"
"$FF_HOME"/build/examples/cpp/MLP_Unify/mlp_unify -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 --budget 20

echo "Running MLP with data parallelism"
"$FF_HOME"/build/examples/cpp/MLP_Unify/mlp_unify -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 --budget 20 --only-data-parallel
