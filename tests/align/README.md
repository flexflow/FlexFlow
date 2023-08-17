# FlexFlow-PyTorch Alignment

This is an ongoing effort to align FlexFlow with PyTorch as a means to verify
the correctness of FlexFlow. Support for additional operators will be coming
soon, and all alignment files here are subject to change.
## Install the Python dependencies
install `pytest` module in flexflow environment.

## Running the Alignment Tests
Note that FlexFlow requires a CPU installation of PyTorch, so we recommend a
separate `conda` environment for each (e.g. named `flexflow` and `pytorch`,
respectively).

Assuming those two `conda` environments, we may run
```
cd FlexFlow
conda activate flexflow
./tests/align/test_all_operators.sh
```

