#! /usr/bin/env bash
set -x
set -e

if [ -z "$NUM_NODES" ]; then echo "NUM_NODES variable is not defined, aborting tests"; exit; fi
if [ -z "$GPUS" ]; then echo "GPUS variable is not defined, aborting tests"; exit; fi

# We need to wrap the instruction below in its own script because the CUDA_VISIBLE_DEVICES environment
# variable will need to be set differently for each node, but the "mpirun" command should take a single
# executable as its first argument
CUDA_VISIBLE_DEVICES=$(seq -s, $((OMPI_COMM_WORLD_RANK * GPUS ))  $(( OMPI_COMM_WORLD_RANK * GPUS +1 )) )
export CUDA_VISIBLE_DEVICES

python "$@"
