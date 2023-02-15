import os
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
from flexflow.core import *
from flexflow.torch.model import PyTorchModel, InputNode, OutputNode
from transformers import MT5ForConditionalGeneration

PRETRAINED_MODEL_NAME = "google/mt5-small"

BASE_DIR = "examples/python/pytorch/mt5"
DATA_DIR = os.path.join(BASE_DIR, "data")
BATCH_DIR = os.path.join(DATA_DIR, "batch")
INPUT_IDS_PATH = os.path.join(BATCH_DIR, "ids.pt")
ATTENTION_MASK_PATH = os.path.join(BATCH_DIR, "mask.pt")
DECODER_INPUT_IDS_PATH = os.path.join(BATCH_DIR, "y_ids.pt")
LABELS_PATH = os.path.join(BATCH_DIR, "lm_labels.pt")


def load_batch_ff():
    """Loads a single batch for mT5, consisting of the encoder input IDs,
    encoder attention mask, decoder input IDS, and labels, all as numpy
    arrays."""
    input_ids = torch.load(INPUT_IDS_PATH).numpy()
    attention_mask = torch.load(ATTENTION_MASK_PATH).numpy()
    decoder_input_ids = torch.load(DECODER_INPUT_IDS_PATH).numpy()
    labels = torch.load(LABELS_PATH).numpy()
    return (input_ids, attention_mask, decoder_input_ids, labels)


def init_ff_mt5():
    """
    Initializes the FlexFlow representation of the HuggingFace mT5 model.

    Returns:
        (ffmodel, input_dls, label_dl)

        ffmodel (FFModel): Compiled and initialized FlexFlow model representing
            HuggingFace mT5.
        input_dls (List[SingleDataLoader]): List consisting of the encoder
            input IDs, encoder attention mask, and decoder input IDs
            dataloaders.
        label_dl (SingleDataLoader): Label dataloader.
    """
    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)
    mt5_torch = MT5ForConditionalGeneration.from_pretrained(
        PRETRAINED_MODEL_NAME,
    )
    input_ids, attention_mask, decoder_input_ids, labels = load_batch_ff()
    input_tensors = [
        ffmodel.create_tensor(input_ids.shape, DataType.DT_INT64),
        ffmodel.create_tensor(attention_mask.shape, DataType.DT_INT64),
        ffmodel.create_tensor(decoder_input_ids.shape, DataType.DT_INT64),
    ]
    mt5_model = PyTorchModel(
        mt5_torch,
        is_hf_model=True,
        input_names=["input_ids", "attention_mask", "decoder_input_ids"],
        batch_size=ffconfig.batch_size,
        seq_length=(input_ids.shape[1], decoder_input_ids.shape[1]),
    )
    output_tensors = mt5_model.torch_to_ff(ffmodel, input_tensors)
    ffoptimizer = SGDOptimizer(ffmodel, lr=0.01)
    ffmodel.compile(
        optimizer=ffoptimizer,
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[
            MetricsType.METRICS_ACCURACY,
            MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY,
        ],
    )
    input_ids_dl = ffmodel.create_data_loader(input_tensors[0], input_ids)
    attention_mask_dl = ffmodel.create_data_loader(
        input_tensors[1], attention_mask,
    )
    decoder_input_ids_dl = ffmodel.create_data_loader(
        input_tensors[2], decoder_input_ids,
    )
    # NOTE: We cast down the label tensor data to 32-bit to accomomodate the
    # label tensor's bitwidth requirement
    label_dl = ffmodel.create_data_loader(
        ffmodel.label_tensor, labels.astype("int32"),
    )
    input_dls = [input_ids_dl, attention_mask_dl, decoder_input_ids_dl]
    ffmodel.init_layers()
    return (ffmodel, input_dls, label_dl)


def extract_mt5_subgraph(
    initial_op_name: Optional[str] = None,
    final_op_name: Optional[str] = None,
):
    """
    Extracts the mT5 subgraph starting from ``initial_op_name`` and ending
    with ``final_op_name`` (inclusive) in the topological order. If either
    argument is ``None``, then that side of the limit defaults to the first
    and last operator, respectively.

    NOTE: HuggingFace's symbolic trace only supports tracing a selection of
    classes. As a result, we must extract subgraphs from the full mT5 graph
    in the Python FlexFlow space.

    Returns:
        subgraph (List[Node]): List of the nodes comprising the subgraph.
    """
    mt5_torch = MT5ForConditionalGeneration.from_pretrained(
        PRETRAINED_MODEL_NAME,
    )
    input_ids, _, decoder_input_ids, _ = load_batch_ff()
    BATCH_SIZE = 8
    mt5_model = PyTorchModel(
        mt5_torch,
        is_hf_model=True,
        input_names=["input_ids", "attention_mask", "decoder_input_ids"],
        batch_size=BATCH_SIZE,
        seq_length=(input_ids.shape[1], decoder_input_ids.shape[1]),
    )
    graph = mt5_model._trace_model()
    subgraph = []
    in_subgraph: bool = initial_op_name is None
    for node in graph:
        if initial_op_name is not None and node.name == initial_op_name:
            in_subgraph = True
        if in_subgraph:
            subgraph.append(node)
        if final_op_name is not None and node.name == final_op_name:
            break
    return subgraph


def extract_mt5_encoder():
    """Extracts the mT5 subgraph corresponding to the encoder only."""
    return extract_mt5_subgraph(final_op_name="encoder_dropout_1")


def init_ff_mt5_encoder(encoder_labels_filepath: str):
    """
    Initializes the FlexFlow representation of the HuggingFace mT5 model's
    encoder.

    Returns:
        (ffmodel, input_dls, label_dl)

        ffmodel (FFModel): Compiled and initialized FlexFlow model representing
            HuggingFace mT5's encoder.
        input_dls (List[SingleDataLoader]): List consisting of the encoder
            input IDs, encoder attention mask, and decoder input IDs
            dataloaders.
        label_dl (SingleDataLoader): Label dataloader.
    """
    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)
    input_ids, attention_mask, decoder_input_ids, _ = load_batch_ff()
    labels = torch.load(encoder_labels_filepath).detach().numpy()
    input_tensors = [
        ffmodel.create_tensor(input_ids.shape, DataType.DT_INT64),
        ffmodel.create_tensor(attention_mask.shape, DataType.DT_INT64),
        ffmodel.create_tensor(decoder_input_ids.shape, DataType.DT_INT64),
    ]
    # Add the encoder operators to `ffmodel`
    mt5_encoder_graph = extract_mt5_encoder()
    input_index = 0
    output_tensors = []
    node_to_output = OrderedDict()
    for node in mt5_encoder_graph:
        if isinstance(node, InputNode):
            node_output = node.to_ff(input_tensors, input_index)
            input_index += 1
        elif isinstance(node, OutputNode):
            node.to_ff(ffmodel, node_to_output, output_tensors)
            node_output = None
        else:
            node_output = node.to_ff(ffmodel, node_to_output)
        if node_output is not None:
            node_to_output[node.name] = node_output
    # Compile and initialize the model
    ffoptimizer = SGDOptimizer(ffmodel, lr=0.01)
    ffmodel.compile(
        optimizer=ffoptimizer,
        loss_type=LossType.LOSS_MEAN_SQUARED_ERROR_AVG_REDUCE,
        metrics=[MetricsType.METRICS_MEAN_SQUARED_ERROR],
    )
    input_ids_dl = ffmodel.create_data_loader(input_tensors[0], input_ids)
    attention_mask_dl = ffmodel.create_data_loader(
        input_tensors[1], attention_mask,
    )
    decoder_input_ids_dl = ffmodel.create_data_loader(
        input_tensors[2], decoder_input_ids,
    )
    label_dl = ffmodel.create_data_loader(ffmodel.label_tensor, labels)
    input_dls = [input_ids_dl, attention_mask_dl, decoder_input_ids_dl]
    ffmodel.init_layers()
    return (ffmodel, input_dls, label_dl)
