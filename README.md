FlexFlow
========
A distributed deep learning framework that supports flexible parallelization strategies.

Prerequisties
-------------
* [CUDNN](https://developer.nvidia.com/cudnn) is used to perform low-level operations.

* [Legion](http://legion.stanford.edu) is the underlying runtime FlexFlow built on.

* (Optinal) [GASNet](http://gasnet.lbl.gov) is used for multi-node executions. (see [installation instructions](http://legion.stanford.edu/gasnet))

After you have cloned FlexFlow, use the following command lines to clone Legion and GASNet.
```
git submodule init
git submodule update
```

Compilation
-----------
* Download FlexFlow source code:
```
# Using git to download FlexFlow
git clone --recursive https://gitlab.com/fflow/flexflow
```

* Build a DNN model (e.g., alexnet):
```
./ffcompile.sh alexnet
```
where `alexnet.cc` defines all operators in a DNN.

* To build a distributed version of FlexFlow, add a `-d` flag:
```
./ffcompile.sh -d alexnet
```

Parallelization Strategy
------------------------
Flexflow accepts any parallelization strategy in layer-wise parallelism (see ) to parallelize training. A parallelization strategy should describe how to parallelize each operator in a DNN. An example parallelization strategy for AlexNet is as follows.

| **Layer** | **Type** | **Configuration** | **Devices** |
|-----------|----------|-------------------|-------------|
| conv1     | conv2d   | n=4 c=1 h=1 w=1   | 0 1 2 3     |
| pool1     | pool2d   | n=4 c=1 h=1 w=1   | 0 1 2 3     |
| conv2     | conv2d   | n=1 c=1 h=2 w=2   | 0 2 1 3     |
| pool2     | pool2d   | n=1 c=1 h=2 w=2   | 0 2 1 3     |
| flat1     | flat     | n=2 c=1           | 0 2         |
| linear1   | linear   | n=1 c=3           | 0 2 3       |
| linear2   | linear   | n=1 c=3           | 0 1 2       |
| linear3   | linear   | n=1 c=1           | 0           |
Some example parallelization strategies are available in the `strategies` subfolder.

Training a DNN model
--------------------
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

Publication
-----------
Zhihao Jia, Sina Lin, Charles R. Qi, and Alex Aiken. [Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks](http://proceedings.mlr.press/v80/jia18a/jia18a.pdf). In Proceedings of the International Conference on Machine Learning (ICML), Stockholm, Sweden, July 2018.
