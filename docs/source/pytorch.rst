:tocdepth: 1
******************
PyTorch Interface
******************

Users can use FlexFlow to optimize the parallelization performance of existing PyTorch models in two steps.
The PyTorch support requires the `PyTorch FX module <https://github.com/pytorch/pytorch/pull/42741>`_, so make sure your PyTorch is up to date. 

1. Export a PyTorch Model to an external file
===============================================

A PyTorch model can be exported to the FlexFlow model format and saved into an external file::

    import torch
    import flexflow.torch.fx as fx

    # create a PyTorch Model
    class MyPyTorchModule(nn.Module):
    ...
    
    # export the PyTorch Model to an external file
    model = MyPyTorchModule()
    fx.torch_to_flexflow(model, "filename")

2. Import a FlexFlow model from a external file
===============================================

A FlexFlow program can directly import a previously saved PyTorch model and autotune the parallelization performance for a given parallel machine::

    from flexflow.torch.model import PyTorchModel

    # create input tensors
    dims_input = [ffconfig.get_batch_size(), 3, 32, 32]
    input_tensor = ffmodel.create_tensor(dims_input, DataType.DT_FLOAT)

    # create a flexflow model from the file
    torch_model = PyTorchModel("filename")
    output_tensor = torch_model.apply(ffmodel, [input_tensor])

    # use the Python API to train the model
    ffoptimizer = SGDOptimizer(ffmodel, 0.01)
    ffmodel.set_sgd_optimizer(ffoptimizer)
    ffmodel.compile(
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[
            MetricsType.METRICS_ACCURACY,
            MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY,
        ],
    )
    ...
    ffmodel.fit(x=dataloader_input, y=dataloader_label, epochs=epochs)

More FlexFlow PyTorch examples are available on `GitHub <https://github.com/flexflow/FlexFlow/tree/master/examples/python/pytorch>`_.


  