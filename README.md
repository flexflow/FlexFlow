# Unity

## Getting Started

The following AMI provides a full environment for running Unity. A TODO machine should be used for the evaluation.

If the evaluator would prefer that we spin up the AWS instance ourselves, we encourage the evaluator to contact us and inform of us when they plan to do their evaluation so we can ensure the machine is running at that point. 

As a starting point, the evaluator can run the following command to test out a small toy model:

```
$ cd ~/FlexFlow
$ ./build/examples/cpp/split_test/split_test -ll:gpu 4 -ll:fsize 13000 -ll:zsize 16384 -ll:csize 40000 --budget 10 --batch-size 32 
```

## Detailed Instructions

### Availability

The Unity source code can be found at . 
Unity is maintained as part of the open-source FlexFlow project for distributed DNN training.

### Artifact Claims

For the models evaluated in the paper (i.e. ResNeXt-50, BERT-Large, DLRM, CANDLE-Uno, Inception-v3, MLP, and XDL), the evaluator should be able to use Unity to optimize and run the models.
Since access to the machine with 192 GPUs cannot be shared, we expect the evaluator to run and optimize the models on a smaller machine setup (likely 4-8 GPUs).
We expect the reviewer to see increased throughput as compared to the unoptimized data-parallel model.

### Running other models

Model performance can be seen via the 

Running in data parallel:

#### ResNeXt-50

```
$
```

#### BERT-Large

```
$
```

#### DLRM

```
$ 
```

#### CANDLE-Uno

```
$
```

#### Inception-v3

```
$
```

#### MLP

```
$
```

#### XDL

```
$
```

### Rebuilding Unity

When using the provided AMI, the evaluator can rebuild the artifact as follows:
```
$ cd ~/FlexFlow/build
$ ../cmake.sh
```

### Evaluating Search

While the runtime cannot execute the model on sizes larger than the provided machine, the search algorithm can be run for larger problem sizes to check its scalability/running time.
To run search as if on a larger machine/cluster, the following command line flags can be added to the standard invocation:

```
$ <standard-invocation> --search-num-nodes <desired-num-nodes> --search-num-workers <desired-num-gpus-per-node>
```

### Controlling Hyperparameters

