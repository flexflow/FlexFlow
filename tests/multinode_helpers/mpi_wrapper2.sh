#! /usr/bin/env bash
set -x
set -e

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi
if [ -z "$NUM_NODES" ]; then echo "NUM_NODES variable is not defined, aborting tests"; exit; fi
if [ -z "$GPUS" ]; then echo "GPUS variable is not defined, aborting tests"; exit; fi

CUDA_VISIBLE_DEVICES=$(seq -s, $((OMPI_COMM_WORLD_RANK * GPUS ))  $(( OMPI_COMM_WORLD_RANK * GPUS +1 )) )
export CUDA_VISIBLE_DEVICES

EXE="$FF_HOME"/python/flexflow_python

$EXE "$@"

