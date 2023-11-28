#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate flexflow
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# Path to your FlexFlow build
FLEXFLOW_DIR=/home/ubuntu/FlexFlow/build

# Path to your UCX installation
UCX_DIR=/home/ubuntu/ucx-1.15.0/install

export REALM_UCP_BOOTSTRAP_PLUGIN=$FLEXFLOW_DIR/deps/legion/lib/realm_ucp_bootstrap_mpi.so
export LD_LIBRARY_PATH=$FLEXFLOW_DIR/deps/legion/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$FLEXFLOW_DIR:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$UCX_DIR/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/conda/envs/flexflow/lib:$LD_LIBRARY_PATH

mpiexec -x REALM_UCP_BOOTSTRAP_PLUGIN -x PATH -x LD_LIBRARY_PATH --hostfile ~/hostfile --mca btl_tcp_if_include ens5 -np 2 "$FLEXFLOW_DIR"/flexflow_python "$FLEXFLOW_DIR"/../examples/python/native/mnist_mlp.py -ll:py 1 -ll:gpu 1 -ll:fsize 8000 -ll:zsize 8000
