# an example of running llama model with beam search

## how to run it?
1. build the flexflow with***
2. cd into the folder FF_HOME/build/examples/cpp/inference/LLAMA 
3. download the weight and token file from aws s3. eg. 
3. run *LLAMA* with -b 5 and --only-data-parallel
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

