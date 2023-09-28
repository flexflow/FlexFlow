#! /usr/bin/env bash

echo "Running CANDLE Uno with a parallelization strategy discovered by Unity"
"$FF_HOME"/build/examples/cpp/candle_uno/candle_uno -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 --budget 20

echo "Running CANDLE Uno with data parallelism"
"$FF_HOME"/build/examples/cpp/candle_uno/candle_uno -ll:gpu 4 -ll:fsize 14000 -ll:zsize 14000 --budget 20 --only-data-parallel
