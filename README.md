# FlexFlow
![build](https://github.com/flexflow/flexflow/workflows/build/badge.svg?branch=master) ![docker](https://github.com/flexflow/flexflow/workflows/docker-build/badge.svg?branch=master) [![Documentation Status](https://readthedocs.org/projects/flexflow/badge/?version=latest)](https://flexflow.readthedocs.io/en/latest/?badge=latest)


FlexFlow is a deep learning framework that accelerates distributed DNN training by automatically searching for efficient parallelization strategies. FlexFlow provides a drop-in replacement for TensorFlow Keras and PyTorch. Running existing Keras and PyTorch programs in FlexFlow only requires [a few lines of changes to the program](https://flexflow.ai/keras).

## Install FlexFlow
To install FlexFlow from source code, please read the [instructions](INSTALL.md). If you would like to quickly try FlexFlow, we also provide a [Dockerfile](./docker) with all dependencies pre-installed. You can also use `conda` to install the FlexFlow Python package (coming soon).

## TensorFlow Keras Support
Users can use FlexFlow to accelerate the training procedure of existing TensorFlow Keras models by just changing the following import header lines.
```python
from flexflow.keras.models import Model, Sequential
from flexflow.keras.layers import Input, Dense, Conv2D, ...
from flexflow.keras.callbacks import Callback, ...
```

FlexFlow uses a Python function called `top_level_task()` as the entry point of a program and automatically parallelize DNN training across all GPUs on all compute nodes. For example, the following code snippet shows parallelizing AlexNet training on the CIFAR10 dataset in FlexFlow. 
```python
def top_level_task():
  model = Sequential()
  model.add(Conv2D(filters=64, input_shape=(3,229,229), kernel_size=(11,11), strides=(4,4), padding=(2,2), activation="relu"))
  model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding="valid"))
  model.add(Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), padding=(2,2), activation="relu"))
  ## More lines for model construction
  model.add(Activation("softmax"))
  ## Model compilation
  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  ## Model training
  (x_train, y_train) = cifar10.load_data()
  model.fit(x_train, y_train, epochs=30)

if __name__ == "__main__":
  top_level_task()
```

During model compilation (i.e., `model.compile` in Keras), FlexFlow can [autotune](https://www.usenix.org/conference/osdi22/presentation/unger) the parallelization performance by searching for efficient strategies on the given parallel machine. Next, `model.fit` performs DNN training on all available GPUs (potentially across multiple nodes) using the best discovered strategy. As a result, users don't need to manually design and optimize the device assignments.

**More FlexFlow Keras examples**: see the [keras examples folder](https://github.com/flexflow/FlexFlow/tree/master/examples/python/keras).

## PyTorch Support
Users can also use FlexFlow to optimize the parallelization performance of existing PyTorch models in two steps. First, a PyTorch model can be exported to the FlexFlow model format using `flexflow.torch.fx.torch_to_flexflow`.
```python
import torch
import flexflow.torch.fx as fx

model = MyPyTorchModule()
fx.torch_to_flexflow(model, "mymodel.ff")
```

Second, a FlexFlow program can directly import a previously saved PyTorch model and [autotune](https://www.usenix.org/conference/osdi22/presentation/unger) the parallelization performance for a given parallel machine.

```
from flexflow.pytorch.model import PyTorchModel

def top_level_task():
  torch_model = PyTorchModel("mymodel.ff")
  output_tensor = torch_model.apply(ffmodel, input_tensor)
  ## Model compilation
  ffmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  ## Model training
  (x_train, y_train) = cifar10.load_data()
  ffmodel.fit(x_train, y_train, epochs=30)
```

**More FlexFlow PyTorch examples**: see the [pytorch examples folder](https://github.com/flexflow/FlexFlow/tree/master/examples/python/pytorch).

## ONNX Support
Similar to the PyTorch front-end, FlexFlow also supports training existing ONNX models by loading the models using `flexflow.onnx.model.ONNXModel`.

**More FlexFlow ONNX examples**: see the [ONNX examples folder](https://github.com/flexflow/FlexFlow/tree/master/examples/python/keras).

## C++ Interface
For users that prefer to program in C/C++. FlexFlow supports a C++ program inference that is equivalent to its Python APIs.

**More FlexFlow C++ examples**: see the [C++ examples folder](https://github.com/flexflow/FlexFlow/tree/master/examples/c++).


## Command-Line Flags
In addition to setting runtime configurations in a FlexFlow Python/C++ program, the FlexFlow runtime also accepts command-line arguments for various runtime parameters: 

FlexFlow training flags:
* `-e` or `--epochs`: number of total epochs to run (default: 1)
* `-b` or `--batch-size`: global batch size in each iteration (default: 64)
* `-p` or `--print-freq`: print frequency (default: 10)
* `-d` or `--dataset`: path to the training dataset. If not set, synthetic data is used to conduct training.

Legion runtime flags:
* `-ll:gpu`: number of GPU processors to use on each node (default: 0)
* `-ll:fsize`: size of device memory on each GPU (in MB)
* `-ll:zsize`: size of zero-copy memory (pinned DRAM with direct GPU access) on each node (in MB). This is used for prefecthing training images from disk.
* `-ll:cpu`: number of data loading workers (default: 4)
* `-ll:util`: number of utility threads to create per process (default: 1)
* `-ll:bgwork`: number of background worker threads to create per process (default: 1)

Performance auto-tuning flags:
* `--search-budget` or `--budget`: the number of iterations for the MCMC search (default: 0)
* `--search-alpha` or `--alpha`: a hyper-parameter for the search procedure (default: 0.05)
* `--export-strategy` or `--export`: path to export the best discovered strategy (default: None)
* `--import-strategy` or `--import`: path to import a previous saved strategy (default: None)
* `--enable-parameter-parallel`: allow FlexFlow to explore parameter parallelism for performance auto-tuning. (By default FlexFlow only considers data and model parallelism.)
* `--enable-attribute-parallel`: allow FlexFlow to explore attribute parallelism for performance auto-tuning. (By default FlexFlow only considers data and model parallelism.)
For performance tuning related flags: see [performance autotuning](https://flexflow.ai/search).

## Contributing
Please let us know if you encounter any bugs or have any suggestions by [submitting an issue](https://github.com/flexflow/flexflow/issues).

We welcome all contributions to FlexFlow from bug fixes to new features and extensions.

Please subscribe to the FlexFlow users mailing list for 

## Citations
* Colin Unger, Zhihao Jia, Wei Wu, Sina Lin, Mandeep Baines, Carlos Efrain Quintero Narvaez, Vinay Ramakrishnaiah, Nirmal Prajapati, Pat McCormick, Jamaludin Mohd-Yusof, Xi Luo, Dheevatsa Mudigere, Jongsoo Park, Misha Smelyanskiy, and Alex Aiken. [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization](https://www.usenix.org/conference/osdi22/presentation/unger). In Proceedings of the Symposium on Operating Systems Design and Implementation (OSDI), July 2022. 

* Zhihao Jia, Matei Zaharia, and Alex Aiken. [Beyond Data and Model Parallelism for Deep Neural Networks](https://cs.stanford.edu/~zhihao/papers/sysml19a.pdf). In Proceedings of the 2nd Conference on Machine Learning and Systems (MLSys), April 2019.

* Zhihao Jia, Sina Lin, Charles R. Qi, and Alex Aiken. [Exploring Hidden Dimensions in Parallelizing Convolutional Neural Networks](http://proceedings.mlr.press/v80/jia18a/jia18a.pdf). In Proceedings of the International Conference on Machine Learning (ICML), July 2018.

## The Team
FlexFlow is developed and maintained by teams at CMU, Facebook, Los Alamos National Lab, MIT, and Stanford (alphabetically).

## License
FlexFlow uses Apache License 2.0.
