# Running FlexFlow On Multiple Nodes
First follow the first three steps in [INSTALL.md](INSTALL.md) to download source code and install dependencies on each node.

## 4. Configuring the FlexFlow build
Set `FF_USE_GASNET=ON` and `FF_GASNET_CONDUIT` as a specific conduit (e.g. `ibv`, `mpi`, `udp`, `ucx`) in `config/config.linux`.

## 5. Build FlexFlow
Follow the step 5 in [INSTALL.md](INSTALL.md) to build FlexFlow on each node.

## 6. Test FlexFlow
### Set the `FF_HOME` environment variable on each node before running FlexFlow. To make it permanent, you can add the following line in ~/.bashrc.
```
export FF_HOME=/path/to/FlexFlow
```
### Run FlexFlow Python examples
The Python examples are in the [examples/python](https://github.com/flexflow/FlexFlow/tree/master/examples/python). The native, Keras integration and PyTorch integration examples are listed in `native`, `keras` and `pytorch` respectively.

To run the Python examples, you have two options: you can use the `flexflow_python` interpreter, available in the `python` folder, or you can use the native Python interpreter. If you choose to use the native Python interpreter, you should either install FlexFlow, or, if you prefer to build without installing, export the following flags on each node:

* `export PYTHONPATH="${FF_HOME}/python:${FF_HOME}/build/python"`
* `export FF_USE_NATIVE_PYTHON=1`


A script to run a Python example on multiple nodes is available at `scripts/multinode_run.sh`. Run the script using [`mpirun`](https://www.open-mpi.org/doc/current/man1/mpirun.1.php) or [`srun`](https://slurm.schedmd.com/srun.html). For example:
```
mpirun --host <host1>:<slot1>,<host2>:<slot2> -np <num_proc> ./scripts/multinode_run.sh
```