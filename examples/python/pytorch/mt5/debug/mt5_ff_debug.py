import os

import numpy as np
import torch
from flexflow.core import *
from flexflow.torch.model import PyTorchModel
from flexflow.type import ParameterSyncType
from transformers import MT5ForConditionalGeneration

PRETRAINED_MODEL_NAME = "google/mt5-small"

BASE_DIR = "examples/python/pytorch/mt5"
DATA_DIR = os.path.join(BASE_DIR, "data")
BATCH_DIR = os.path.join(DATA_DIR, "batch")
INPUT_IDS_PATH = os.path.join(BATCH_DIR, "ids.pt")
ATTENTION_MASK_PATH = os.path.join(BATCH_DIR, "mask.pt")
DECODER_INPUT_IDS_PATH = os.path.join(BATCH_DIR, "y_ids.pt")
LABELS_PATH = os.path.join(BATCH_DIR, "lm_labels.pt")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_batch_ff():
    """Loads a single batch for mT5."""
    # Load the data
    input_ids = torch.load(INPUT_IDS_PATH).numpy()
    attention_mask = torch.load(ATTENTION_MASK_PATH).numpy()
    decoder_input_ids = torch.load(DECODER_INPUT_IDS_PATH).numpy()
    labels = torch.load(LABELS_PATH).numpy()

    return (input_ids, attention_mask, decoder_input_ids, labels)


def train_step_ff():
    set_seed(42)
    print("Loading the model...")
    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)
    model = MT5ForConditionalGeneration.from_pretrained(
        PRETRAINED_MODEL_NAME,
    )

    print("Loading data...")
    batch = load_batch_ff()
    input_ids: np.array = batch[0]
    attention_mask: np.array = batch[1]
    decoder_input_ids: np.array = batch[2]
    labels: np.array = batch[3]

    batch_size = ffconfig.batch_size
    assert batch_size == input_ids.shape[0], "Batch size mismatch: " \
        f"config batch size={batch_size} data batch size={input_ids.shape[0]}"
    encoder_seq_length = input_ids.shape[1]
    decoder_seq_length = decoder_input_ids.shape[1]
    input_ids_shape = (batch_size, encoder_seq_length)
    attention_mask_shape = (batch_size, encoder_seq_length)
    decoder_input_ids_shape = (batch_size, decoder_seq_length)

    input_tensors = [
        ffmodel.create_tensor(input_ids_shape, DataType.DT_INT64),
        ffmodel.create_tensor(attention_mask_shape, DataType.DT_INT64),
        ffmodel.create_tensor(decoder_input_ids_shape, DataType.DT_INT64),
    ]
    print("input_tensors:")
    print(f"input_ids ({input_ids_shape})")
    print(f"attention_mask ({attention_mask_shape})")
    print(f"decoder_input_ids ({decoder_input_ids_shape})")

    print("Tracing the model...")
    seq_length = (encoder_seq_length, decoder_seq_length)
    input_names = ["input_ids", "attention_mask", "decoder_input_ids"]
    hf_model = PyTorchModel(
        model, is_hf_model=True, input_names=input_names,
        batch_size=batch_size, seq_length=seq_length,
    )
    output_tensors, node_to_output = \
        hf_model.torch_to_ff(ffmodel, input_tensors, verbose=True)
    ffoptimizer = SGDOptimizer(ffmodel, lr=0.01)

    print("Compiling the model...")
    ffmodel.compile(
        optimizer=ffoptimizer,
        loss_type=LossType.LOSS_SPARSE_CATEGORICAL_CROSSENTROPY,
        metrics=[
            MetricsType.METRICS_ACCURACY,
            MetricsType.METRICS_SPARSE_CATEGORICAL_CROSSENTROPY,
        ],
    )

    print("Creating data loaders...")
    input_ids_dl = ffmodel.create_data_loader(input_tensors[0], input_ids)
    attention_mask_dl = ffmodel.create_data_loader(
        input_tensors[1], attention_mask,
    )
    decoder_input_ids_dl = ffmodel.create_data_loader(
        input_tensors[2], decoder_input_ids,
    )
    # NOTE: We cast down the label tensor data to 32-bit to accommodate the
    # label tensor's required dtype
    labels_dl = ffmodel.create_data_loader(
        ffmodel.label_tensor, labels.astype("int32")
    )

    print("Initializing model layers...")
    ffmodel.init_layers()

    print("Training...")
    # ffmodel.fit(
    #     x=[input_ids_dl, attention_mask_dl, decoder_input_ids_dl],
    #     y=labels_dl, batch_size=batch_size, epochs=1,
    # )
    x = [input_ids_dl, attention_mask_dl, decoder_input_ids_dl]
    y = labels_dl
    dataloaders = x + [y]
    num_samples = y.num_samples
    ffmodel._tracing_id += 1
    for d in dataloaders:
        d.reset()
    ffmodel.reset_metrics()
    num_iters = num_samples // batch_size
    assert num_iters == 1
    for d in dataloaders:
        d.next_batch(ffmodel)
    ffmodel._ffconfig.begin_trace(ffmodel._tracing_id)
    ffmodel.forward()
    ffmodel.zero_gradients()
    ffmodel.backward()
    ffmodel.update()
    ffmodel._ffconfig.end_trace(ffmodel._tracing_id)
    ffmodel._ffconfig.get_current_time()  # synchronization barrier

    # Print per-layer information
    # for i, node in enumerate(node_to_output):
    #     layer = ffmodel.get_layer_by_name(node)
    #     if layer is not None and hasattr(layer, "get_output_tensor"):
    #         np_array = layer.get_output_tensor().get_tensor(ffmodel, ParameterSyncType.PS)
    #         # print(f"{node}\t{np.linalg.norm(np_array):.3f}")
    #         # print(f"{node}\t{np_array}\n")
    #         print(f"{i}: {node}")
    #         if i > 5:
    #             break


if __name__ == "__main__":
    train_step_ff()
