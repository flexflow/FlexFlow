v20.20 (Dec 20, 20)

### CHANGELOG v20.20

* Build
    * FlexFlow now supports both Makefile and CMake build. More details are available in [this instruction](https://github.com/flexflow/FlexFlow/blob/master/INSTALL.md).
* Frontend Supports
    * **PyTorch**. FlexFlow now supports training existing PyTorch models with minimal changes to the source code. To run PyTorch models in FlexFlow, users can first export a model to the ONNX format using `torch.onnx` and then load an ONNX model in FlexFlow for distributed training. More examples: https://github.com/flexflow/FlexFlow/tree/master/examples/python/pytorch
    * **ONNX**. FlexFlow supports training existing ONNX models through `flexflow.onnx.model`. More examples: https://github.com/flexflow/FlexFlow/tree/master/examples/python/onnx
    * **TensorFlow Keras**. Similar to the PyTorch support. `flexflow.keras` enables distributed training of existing TensorFlow Keras models. See [this bootcamp talk](https://www.youtube.com/watch?v=PvFHu__eP9Q) for more details.
* Parallelization Optimizer
    * Integrated the parallelization optimizer into the FlexFlow runtime. Users can now use the `--search-budget` and `--search-alpha` to control the FlexFlow parallelization optimizer for searching for optimized strategies. See [this post](https://flexflow.ai/search/) for the usage of the optimizer.
* Examples
   * More PyTorch, ONNX, TensorFlow Keras examples have been added to the `/examples/python` folder.
   * Updated the cpp examples to use the new runtime interface.
* Mapper
    * Implemented a new mapper with improved runtime performance.
* Legion
    * Updated the Legion version with improved runtime performance
