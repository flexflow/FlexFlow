#! /usr/bin/env bash
set -x
set -e

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting tests"; exit; fi
if [ -z "$NUM_NODES" ]; then echo "NUM_NODES variable is not defined, aborting tests"; exit; fi
if [ -z "$GPUS" ]; then echo "GPUS variable is not defined, aborting tests"; exit; fi

mpirun -np "$NUM_NODES" "$FF_HOME"/tests/multinode_helpers/mpi_wrapper2.sh "$@"
