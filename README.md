# Unity

## Getting Started

The following AMI provides a full environment for running Unity. An AWS p3.8xlarge machine should be used for the evaluation. To find the AMI for Unity's artifact, search for `unity-osdi22ae` under the public images in the US East (Ohio, us-east-2) region. For any questions regarding this process, please contact us. 

If the evaluator would prefer that we spin up the AWS instance ourselves, we encourage the evaluator to contact us and inform of us when they plan to do their evaluation so we can ensure the machine is running at that point. 

As a starting point, the evaluator can run the following command to test out a small toy model:

```
$ export FF_HOME = /path/to/flexflow
$ cd $FF_HOME
$ ./build/examples/cpp/split_test/split_test -ll:gpu 4 -ll:fsize 13000 -ll:zsize 16384 -ll:csize 40000 --budget 10 --batch-size 32
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

In this artifact, we evaluate Unity's performance of a DNN model by measuring its training throughput (i.e., number of training samples processed per second). Each application in this artifact will print the achieved throughput number in the last line. All of the models are run for 100 iterations on artificial data to generate this throughput number.

To disable Unity's search procedure and run model in data parallel, you can simply add the commond line flag `--only-data-parallel`.

The total running time for all of the below commands should be under roughly 20 minutes on the AWS instance. 

#### ResNeXt-50

```
$ $FF_HOME/scripts/osdi22ae/resnext-50.sh
```

#### BERT-Large

```
$ $FF_HOME/scripts/osdi22ae/bert-large.sh
```

#### DLRM

```
$ $FF_HOME/scripts/osdi22ae/dlrm.sh 
```

#### CANDLE-Uno

```
$ ./scripts/osdi22ae/candle-uno.sh
```

#### Inception-v3

```
$ $FF_HOME/scripts/osdi22ae/inception-v3.sh
```

#### MLP

```
$ $FF_HOME/scripts/osdi22ae/mlp.sh
```

#### XDL

```
$ $FF_HOME/scripts/osdi22ae/xdl.sh
```

The scripts use the following Unity flags to control the execution of a Unity's application:

* -e or --epochs: number of total epochs to run (default: 1)
* -b or --batch-size: global batch size in each iteration (default: 64)
* -p or --print-freq: print frequency (default: 10)
* -ll:gpu: number of GPU processors to use on each node (default: 0)
* -ll:fsize: size of device memory on each GPU (in MB)
* -ll:zsize: size of zero-copy memory (pinned DRAM with direct GPU access) on each node (in MB). 
* --budget: the search budget
* --only-data-parallel: this flag disables Unity's search and train models in data parallelism

### Rebuilding Unity

When using the provided AMI, the evaluator can rebuild the artifact as follows:
```
$ cd $FF_HOME/build
$ make -j $(nproc)
```
Building should take no more than roughly 20 minutes on the recommended AWS instance.

### Evaluating Search

While the runtime cannot execute the model on sizes larger than the provided machine, the search algorithm can be run for larger problem sizes to check its scalability/running time.
To run search as if on a larger machine/cluster, the following command line flags can be added to the standard invocation:

```
$ <standard-invocation> --search-num-nodes <desired-num-nodes> --search-num-workers <desired-num-gpus-per-node>
```
