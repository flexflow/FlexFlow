# Unity

## Getting Started

The following AMI provides a full environment for running Unity. A TODO machine should be used for the evaluation.

If the evaluator would prefer that we spin up the AWS instance ourselves, we encourage the evaluator to contact us and inform of us when they plan to do their evaluation so we can ensure the machine is running at that point. 

As a starting point, the evaluator can run the following command to test out a small toy model:

```
$ cd ~/FlexFlow
$ ./build/examples/cpp/split_test/split_test -ll:gpu 4 -ll:fsize 13000 -ll:zsize 16384 -ll:csize 40000 --budget 10 --batch-size 32 [TODO ensure this is correct]
```

## Detailed Instructions

### Availability

The Unity source code can be found at https://github.com/flexflow/FlexFlow/tree/osdi2022ae. 
Unity is maintained as part of the open-source FlexFlow project.

### Artifact Claims

For the models evaluated in the paper (i.e. ResNeXt-50, BERT-Large, DLRM, CANDLE-Uno, Inception-v3, MLP, and XDL), the evaluator should be able to use Unity to optimize and run the models.
Since access to the machine with 192 GPUs cannot be shared, we expect the evaluator to run and optimize the models on a smaller machine setup (likely 4-8 GPUs).
We expect the reviewer to see increased throughput as compared to the unoptimized data-parallel model across these models.

### Running other models

Model performance can be seen via the [TODO document how to determine the model throughput]. 
All of the models are run for [TODO] iterations on artificial data to generate this throughput number.

Running in data parallel: [TODO document how to run models in data parallel]

The total running time for all of the below commands should be under roughly 20 minutes on the AWS instance. The execution of a Unity application can be controlled by the following flags

* -e or --epochs: number of total epochs to run (default: 1)
* -b or --batch-size: global batch size in each iteration (default: 64)
* -p or --print-freq: print frequency (default: 10)
* -ll:gpu: number of GPU processors to use on each node (default: 0)
* -ll:fsize: size of device memory on each GPU (in MB)
* -ll:zsize: size of zero-copy memory (pinned DRAM with direct GPU access) on each node (in MB). 
* --budget: the search budget
* --only-data-parallel: this flag disables Unity's search and train models in data parallelism

#### ResNeXt-50

```
$ ./scripts/resnext-50.sh
```

#### BERT-Large

```
$ ./scripts/bert-large.sh
```

#### DLRM

```
$ ./scripts/dlrm.sh 
```

#### CANDLE-Uno

```
$ ./scripts/candle-uno.sh
```

#### Inception-v3

```
$ ./scripts/inception-v3.sh
```

#### MLP

```
$ ./scripts/mlp.sh
```

#### XDL

```
$ ./scripts/xdl.sh
```

### Rebuilding Unity

When using the provided AMI, the evaluator can rebuild the artifact as follows:
```
$ cd ~/FlexFlow/build
$ make -j $(nproc)
```
Building should take no more than roughly 20 minutes on the recommended AWS instance.

### Evaluating Search

While the runtime cannot execute the model on sizes larger than the provided machine, the search algorithm can be run for larger problem sizes to check its scalability/running time.
To run search as if on a larger machine/cluster, the following command line flags can be added to the standard invocation:

```
$ <standard-invocation> --search-num-nodes <desired-num-nodes> --search-num-workers <desired-num-gpus-per-node>
```

### Generating Substitutions

Unity is capable of automatically generating and verifying the graph substitutions used in optimization. 
The process for generating these is described below.
Substitution generation can be run as described in Section 4 as follows: 
```
$ 
```
Note that substitution generation can take up to 30 minutes to complete.

To verify the generated substitutions:
```
$ [TODO]
```
There shoudl be a total of [TODO] substitutions generated.
Verification requires roughly [TODO] minutes to complete.

To view the generated substitutions, the following command can be used to transform the outputted protobuf file into json:
```
$ ~/FlexFlow/build/src/tools/protobuf_to_json [TODO]
```

Individual substitutions can be viewed as dot graphs via the following command:
```
$ ~/FlexFlow/build/src/tools/substitution_to_dot [TODO]
```
The output of this command can be visualized by copying and pasting it into https://dreampuf.github.io/GraphvizOnline/.
