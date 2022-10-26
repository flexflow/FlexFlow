# HuggingFace mT5 in FlexFlow
## Prerequisites and Setup

We mention a few prerequisites and tips for setting up.
- We assume access to at least one GPU and an installation of Anaconda.
- We assume PyTorch version 1.9.
- Using PyTorch and FlexFlow concurrently requires a CPU version of PyTorch.
    - To install the CPU version of `torch` (and `torchvision`), run:
    ```
    conda install pytorch==1.9.0 torchvision==0.10.0 cpuonly -c pytorch
    ```
    - To install the CPU version of `torch` from source, clone the [repository](https://github.com/pytorch/pytorch/tree/release/1.9), run `export USE_CUDA=0 USE_CUDNN=0 USE_MKLDNN=1`, run `git submodule sync; git submodule update --init --recursive`, and run `python setup.py develop` (or `python setup.py install`).
- We need an installation of the HuggingFace `transformers` repository.
    - To install `transformers`, run:
    ```
    conda install -c conda-forge transformers
    ```
    
    - To install `transformers` from source, clone the [repository](https://github.com/huggingface/transformers/tree/v4.10.2-release), and run `python setup.py develop` (or `python setup.py install`).
- To run PyTorch-FlexFlow examples, make sure to run `export FF_USE_CFFI=1` to use `cffi` instead of `pybind11`.
- Additional notes:
    - You may need to update `huggingface_hub` with:
    ```
    conda update huggingface_hub
    ```
    - If you encounter `ImportError: Found an incompatible version of torch.`, try updating to a later version of `transformers`.



## mT5 in PyTorch
We present an example of training mT5 for the Sinhalese-English translation
task from
[here](https://towardsdatascience.com/how-to-train-an-mt5-model-for-translation-with-simple-transformers-30ba5fa66c5f),
reusing some code from
[here](https://shivanandroy.com/fine-tune-t5-transformer-with-pytorch/). In
this section, we walk through the training script using PyTorch, and in the
next section, we walk through the training script using FlexFlow. The
corresponding code may be found in `mt5_torch.py` and `mt5_ff.py`,
respectively.

To download and uncompress the dataset, run:
```
cd examples/python/pytorch/mt5
wget https://object.pouta.csc.fi/Tatoeba-Challenge/eng-sin.tar
tar -xvf eng-sin.tar
gzip -d data/eng-sin/*.gz
```

This will create a directory `data/` containing a single subdirectory
`data/eng-sin/` containing `test.id`, `test.src`, `test.trg`, `train.id`,
`train.src`, and `train.trg`.

We extract, prepare, and save the data to `.tsv` by using
`DataPreparer.data_to_tsv()` -- this creates two new files, `data/train.tsv` and
`data/eval.tsv`, and only needs to be done once. Then, we can train using those
`.tsv` files. A base implementation for this may be found in `mt5_torch.py`,
which saves the `.tsv` files, trains for some number of epochs, and outputs a
`.csv` containing the predicted and actual text on the evaluation data.
```
python examples/python/pytorch/mt5/mt5_torch.py
```
_Note:_ Running `mt5_torch.py` requires a GPU-version of PyTorch.


## mT5 in FlexFlow

Now, we examine how to write a similar training script using FlexFlow. To
begin, FlexFlow dataloaders expect the data to be passed in as `numpy` arrays
and to be already preprocessed so that batches may be directly given to the
model. In `mt5_ff.py`, `data_to_numpy()` converts the `.tsv` files to `.npy`,
and `preprocess_train()` performs the necessary preprocessing.

_Note:_ `data_to_numpy()` takes a while to run.

Next, following the conventional FlexFlow terminology, we define a _top-level
task_ to train the mT5 model. The key steps are as follows (including some
notable code snippets):
- Define `ffconfig = FFConfig()` and `ffmodel = FFModel(ffconfig)` -- `ffmodel` is the Python object for the FlexFlow model
- Define the PyTorch mT5 model:
    ```
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")
    ```
- Load the preprocessed training data from the `.npy` files
- Use `ffmodel.create_tensor()` for the `input_ids`, `attention_mask`, and `decoder_input_ids` -- these are the input tensors to the model
- Construct a `PyTorchModel()` object wrapping the PyTorch model `model` to enable conversion to FlexFlow:
    ```
    hf_model = PyTorchModel(
        model, is_hf_model=True, batch_size=ffconfig.batch_size,
        seq_length=seq_length,
    )
    ```
    - We pass `is_hf_model=True` since HuggingFace models require a special `symbolic_trace()` distinct from the native PyTorch one.
    - `seq_length` is a tuple `(encoder_seq_length, decoder_seq_length)`.
- Convert the model to FlexFlow:
    ```
    output_tensors = hf_model.to_ff(ffmodel, input_tensors)
    ```
- Define the optimizer `ffoptimizer`
- Compile the model:
    ```
    ffmodel.compile(
        optimizer=ffoptimizer,
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[
            MetricsType.METRICS_ACCURACY,
            MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY,
        ],
    )
    ```
- Create the dataloaders for the `input_ids`, `attention_mask`, `decoder_input_ids`, and `labels`
- Initialize the model layers:
    ```
    ffmodel.init_layers()
    ```
- Train the model, passing the appropriate dataloaders into `fit()`:
    ```
    ffmodel.fit(
        x=[input_ids_dl, attention_mask_dl, decoder_ids_dl],
        y=labels_dl, batch_size=batch_size, epochs=epochs,
    )
    ```

A base implementation may be found in `mt5_ff.py`.
```
./python/flexflow_python examples/python/pytorch/mt5/mt5_ff.py -ll:py 1 -ll:gpu 1 -ll:fsize 14000 -ll:zsize 4096
```
_Note:_ Running `mt5_ff.py` requires a CPU-version of PyTorch.
