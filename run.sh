#! /usr/bin/env bash

set -euo pipefail

DIRNAME="$(basename "$PWD")"
if [[ $DIRNAME == "build" || $DIRNAME == "debug" ]]; then
  echo "Correct directory"
else
  >&2 echo "ERROR: Invalid build directory: $DIRNAME"
  exit 1
fi

export FF_HOME="$PWD/../"

export OUTPUT_DIR="$FF_HOME/data/outputs/"
export SEARCH_CURVE_DIR="$FF_HOME/data/search-curves/"
export COMPGRAPH_DIR="$FF_HOME/data/compgraphs/"
for d in "$OUTPUT_DIR" "$SEARCH_CURVE_DIR" "$COMPGRAPH_DIR"; do
  mkdir -p "$d";
done

export BUDGET=0
export BATCH_SIZE=32
export NUM_GPUS=8
export NUM_NODES=8
export BANDWIDTH=20
export BASENAME="inception_n${NUM_NODES}_g${NUM_GPUS}_b${BATCH_SIZE}_bw${BANDWIDTH}_bu${BUDGET}"
echo "$BASENAME"

./examples/cpp/InceptionV3/inception \
  -ll:gpu 1 \
  -ll:fsize 13000 \
  -ll:zsize 16384 \
  -ll:csize 40000 \
  --budget "$BUDGET" \
  -level xfers=warn,xfer_sim=warn,DP=debug \
  --batch-size "$BATCH_SIZE" \
  --compgraph "$COMPGRAPH_DIR/${BASENAME}_found.dot" \
  --search-curve "$SEARCH_CURVE_DIR/${BASENAME}_found.csv" \
  --simulator-workspace-size 2717908992 \
  --search-curve-interval 1 
  # | tee "$OUTPUT_DIR/${BASENAME}_found.txt"
  #-level DP=spew

# COMMAND="$PWD/../python/flexflow_python" 
# COMMAND="gdb --args ${COMMAND}"

# $COMMAND \
#   "$PWD/../examples/python/native/resnext50.py" \
#   -ll:py 1 \
#   -ll:gpu 4 \
#   -ll:csize 40000 \
#   -ll:zsize 16384 \
#   -ll:fsize 14000 \
#   --batch-size 16 \
#   --budget 1 \
#   --epochs 1 \
#   --overlap \
#   --nodes 1 \
#   -ll:util 4

