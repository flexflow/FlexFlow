# FlexFlow Train: Distributed DNN Training with Flexible Parallelization Strategies.
FlexFlow Train is a deep learning framework that accelerates distributed DNN training by automatically searching for efficient parallelization strategies. FlexFlow Train provides a drop-in replacement for PyTorch and TensorFlow Keras. Running existing PyTorch and Keras programs in FlexFlow oTrain nly requires [a few lines of changes to the program](https://flexflow.ai/keras).


## PyTorch Support
Users can also use FlexFlow Train to optimize the parallelization performance of existing PyTorch models in two steps. First, a PyTorch model can be exported to the FlexFlow model format using `flexflow.torch.fx.torch_to_flexflow`.
```python
import torch
import flexflow.torch.fx as fx

model = MyPyTorchModule()
fx.torch_to_flexflow(model, "mymodel.ff")
```

Second, a FlexFlow Train program can directly import a previously saved PyTorch model and [autotune](https://www.usenix.org/conference/osdi22/presentation/unger) the parallelization performance for a given parallel machine.

```python
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

## TensorFlow Keras and ONNX Support
FlexFlow Train prioritizes PyTorch compatibility, but also includes frontends for [Tensorflow Keras](./docs/source/keras.rst) and [ONNX](./docs/source/onnx.rst) models.

## C++ Interface
For users that prefer to program in C/C++. FlexFlow Train supports a C++ program inference that is equivalent to its Python APIs.

**More FlexFlow C++ examples**: see the [C++ examples folder](https://github.com/flexflow/FlexFlow/tree/master/examples/cpp).


## Command-Line Flags
In addition to setting runtime configurations in a FlexFlow Train Python/C++ program, the FlexFlow Train runtime also accepts command-line arguments for various runtime parameters: 

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
* `--enable-parameter-parallel`: allow FlexFlow Train to explore parameter parallelism for performance auto-tuning. (By default FlexFlow Train only considers data and model parallelism.)
* `--enable-attribute-parallel`: allow FlexFlow Train to explore attribute parallelism for performance auto-tuning. (By default FlexFlow Train only considers data and model parallelism.)
For performance tuning related flags: see [performance autotuning](https://flexflow.ai/search).
