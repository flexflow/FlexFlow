:tocdepth: 1
*************
ONNX Support
*************

Similar to the PyTorch front-end, FlexFlow also supports training existing ONNX models. Since both ONNX and FlexFlow use Protocol Buffer, make sure they are linked with the Protocol Buffer of the same version. 

1. Export a ONNX Model to a external file
===============================================

A PyTorch model can be exported to the FlexFlow model format and saved into an external file::

    import onnx
    import torch
    import torch.nn as nn
    from torch.onnx import TrainingMode
    
    # create a PyTorch Model
    class MyPyTorchModule(nn.Module):
    ...

    # export the PyTorch model to a ONNX model
    model = MyPyTorchModule()
    torch.onnx.export(model, (input), "filename", export_params=False, training=TrainingMode.TRAINING)

2. Import a FlexFlow model from a external file
===============================================

A FlexFlow program can directly import a previously saved ONNX model and autotune the parallelization performance for a given parallel machine::

    from flexflow.torch.model import PyTorchModel

    #create input tensors
    dims_input = [ffconfig.get_batch_size(), 3, 32, 32]
    input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

    # create a flexflow model from the file
    onnx_model = ONNXModel("cifar10_cnn.onnx")
    output_tensor = onnx_model.apply(ffmodel, {"input.1": input_tensor})

    # use the Python API to train the model
    ffoptimizer = SGDOptimizer(ffmodel, 0.01)
    ffmodel.set_sgd_optimizer(ffoptimizer)
    ffmodel.compile(loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY, metrics=[MetricsType.METRICS_ACCURACY, MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY])
    ...
    ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

More FlexFlow ONNX examples are available on `GitHub <https://github.com/flexflow/FlexFlow/tree/master/examples/python/onnx>`_.
