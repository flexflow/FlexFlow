#! /usr/bin/env bash
set -x
set -e

GPUS_PER_NODE=${GPUS_PER_NODE:-2}
export CUDA_VISIBLE_DEVICES=$(seq -s, $((OMPI_COMM_WORLD_RANK * GPUS_PER_NODE ))  $(( OMPI_COMM_WORLD_RANK * GPUS_PER_NODE +1 )) )

TEST=$1

mpirun -np $GPUS_PER_NODE $EXE $TEST -ll:gpu "$GPUS_PER_NODE" -ll:fsize "$FSIZE" -ll:zsize "$ZSIZE" -b "${BATCHSIZE}" "$ONLY_DATA_PARALLEL" ${ADDITIONAL_FLAGS:-}

