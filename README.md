# FlexFlow

FlexFlow is a deep learning framework that accelerates distributed DNN training by automatically discovering fast parallelization strategies.

## Prerequisties
* [CUDNN](https://developer.nvidia.com/cudnn) is used to perform low-level operations.
Download CUDNN and install it locally or at the system level.
```
export CUDNN_HOME=/path/to/cudnn
```

* [Legion](http://legion.stanford.edu) is the underlying runtime FlexFlow built on.

* [Protocol Buffer](https://github.com/protocolbuffers/protobuf) is used for representing parallelization strategies in FlexFlow.

* (Optional) [NCCL](https://github.com/NVIDIA/nccl) is used for parameter synchronization. When NCCL is not available, FlexFlow uses the Legion DMA subsystem for transferring parameters.

* (Optional) [GASNet](http://gasnet.lbl.gov) is used for multi-node executions. (see [GASNet installation instructions](http://legion.stanford.edu/gasnet))

## Install FlexFlow
See [instructions](INSTALL.md) to install FlexFlow from source.

## Build a DNN model
See the [examples](examples) folders for existing FlexFlow applications. Use the following command line to build a DNN model (e.g., InceptionV3).
```
./ffcompile.sh examples/InceptionV3
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDNN_HOME/lib64:$FF_HOME/protobuf/src/.libs
```

## Train a DNN model
To train a DNN model, run the complied application with the path to the training dataset, the path to the parallelization strategy, and some additional configuration flags. For example:
```
./alexnet -e 10 -b 256 --lr 0.1 --wd 1e-4 -p 10 -d path_to_dataset -s path_to_strategy -ll:gpu 4 -ll:fsize 90000 -ll:zsize 5000 -ll:cpu 4
```
* `-e` or `--epochs`: number of total epochs to run (default: 90)
* `-b` or `--batch-size`: global batch size in each iteration (default: 64)
* `--lr` or `--learning-rate`: initial learning rate (default: 0.1)
* `--wd` or `--weight-decay`: weight decay (default: 1e-4)
* `-p` or `--print-freq`: print frequency (default 10)
* `-d` or `--dataset`: path to the training dataset. If not set, synthetic data is used to conduct training. 
* `-s` or `--strategy`: path to the strategy to parallelize training. If not set, data parallelism is used as the default strategy.
* `-ll:gpu`: number of GPU processors to use on each node
* `-ll:fsize`: size of device memory on each GPU (in MB)
* `-ll:zsize`: size of zero-copy memory (pinned DRAM with direct GPU access) on each node (in MB). This is used for prefecthing training images from disk.
* `-ll:cpu`: number of data loading workers (default: 4)

## Parallelization Strategy
FlexFlow supports training a DNN model using parallelization strategies in the SOAP search space (see [paper](https://cs.stanford.edu/~zhihao/papers/sysml19a.pdf)). A parallelization strategy should describe how to parallelize each operator in a DNN. An example parallelization strategy for AlexNet is as follows.

| **Operator** | **Type** | **Configuration** | **Devices** |
|--------------|----------|-------------------|-------------|
| conv1        | conv2d   | n=4 c=1 h=1 w=1   | 0 1 2 3     |
| pool1        | pool2d   | n=4 c=1 h=1 w=1   | 0 1 2 3     |
| conv2        | conv2d   | n=1 c=1 h=2 w=2   | 0 2 1 3     |
| pool2        | pool2d   | n=1 c=1 h=2 w=2   | 0 2 1 3     |
| flat1        | flat     | n=2 c=1           | 0 2         |
| linear1      | linear   | n=1 c=3           | 0 2 3       |
| linear2      | linear   | n=1 c=3           | 0 1 2       |
| linear3      | linear   | n=1 c=1           | 0           |
Some example parallelization strategies are available in the `strategies` subfolder.

Publication
-----------
* Zhihao Jia, Matei Zaharia, and Alex Aiken. [Beyond Data and Model Parallelism for Deep Neural Networks](https://cs.stanford.edu/~zhihao/papers/sysml19a.pdf). In Proceedings of the 2nd Conference on Machine Learning and Systems (MLSys), Palo Alto, CA, April 2019.

* Zhihao Jia, Sina Lin, Charles R. Qi, and Alex Aiken. [Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks](http://proceedings.mlr.press/v80/jia18a/jia18a.pdf). In Proceedings of the International Conference on Machine Learning (ICML), Stockholm, Sweden, July 2018.
