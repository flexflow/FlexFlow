import itertools
import os
import sys

import numpy as np
from flexflow.core import *
from flexflow.torch.model import PyTorchModel
from transformers import MT5ForConditionalGeneration, T5Tokenizer

sys.path.append("./examples/python/pytorch/mt5")
from mt5_torch import DataPreparer, get_dataloaders, set_seed

BASE_DIR = "examples/python/pytorch/mt5"
DATA_DIR = os.path.join(BASE_DIR, "data")
NUMPY_DIR = os.path.join(DATA_DIR, "numpy")


def data_to_numpy() -> None:
    """
    Generates the files:
        - `train_source_ids.npy`
        - `train_source_mask.npy`
        - `train_target_ids.npy`
        - `eval_source_ids.npy`
        - `eval_source_mask.npy`
        - `eval_target_ids.npy`
    This function should only need to be called once (to generate these files).
    """
    model_params = {
        "SEED": 42,
        "MODEL": "google/mt5-small",
        "TRAIN_BATCH_SIZE": None,  # use the full dataset as one batch
        "EVAL_BATCH_SIZE": None,   # use the full dataset as one batch
        "TRAIN_EPOCHS": 1,         # unused
        "MAX_SOURCE_TEXT_LENGTH": 48,
        "MAX_TARGET_TEXT_LENGTH": 48,
    }
    set_seed(model_params)
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    print("Getting dataloaders...")
    train_loader, eval_loader = get_dataloaders(tokenizer, model_params)
    assert len(train_loader) == 1
    assert len(eval_loader) == 1
    print("Saving to numpy...")
    train_set_dict = next(iter(train_loader))
    eval_set_dict = next(iter(eval_loader))
    for k, v in train_set_dict.items():
        np.save(os.path.join(NUMPY_DIR, f"train_{k}.npy"), v.numpy())
    for k, v in eval_set_dict.items():
        np.save(os.path.join(NUMPY_DIR, f"eval_{k}.npy"), v.numpy())


def preprocess_train() -> None:
    """
    Generates the files:
        - `train_y_ids.npy`
        - `train_lm_labels.npy`
    This function should only need to be called once (to generate these files).
    """
    y = np.load(os.path.join(NUMPY_DIR, "train_target_ids.npy"))
    y_shape = y.shape
    assert len(y.shape) == 2, \
        "`y` should have shape (num examples, sequence length)"
    y_ids = np.empty((y_shape[0], y_shape[1] - 1), dtype=np.long)
    lm_labels = np.empty((y_shape[0], y_shape[1] - 1), dtype=np.long)
    y_ids[:, :] = y[:, :-1]
    lm_labels[:, :] = y[:, 1:]

    TOKENIZER_PAD_TOKEN_ID = 0
    NEW_PAD_TOKEN_ID = -100
    # Shift embedding values from {1, ..., n} to {0, ..., n-1}
    y_ids[y[:, :-1] != TOKENIZER_PAD_TOKEN_ID] -= 1
    lm_labels[y[:, 1:] != TOKENIZER_PAD_TOKEN_ID] -= 1
    # Relabel the pad token ID (i.e. `tokenizer.pad_token_id`) from 0 to -100
    y_ids[y[:, :-1] == TOKENIZER_PAD_TOKEN_ID] = NEW_PAD_TOKEN_ID
    lm_labels[y[:, 1:] == TOKENIZER_PAD_TOKEN_ID] = NEW_PAD_TOKEN_ID
    np.save(os.path.join(NUMPY_DIR, "train_y_ids.npy"), y_ids)
    np.save(os.path.join(NUMPY_DIR, "train_lm_labels.npy"), lm_labels)


def top_level_task():
    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)
    model = MT5ForConditionalGeneration.from_pretrained("google/mt5-small")

    # Load train data as numpy arrays
    print("Loading data...")
    ids = np.load(os.path.join(NUMPY_DIR, "train_source_ids.npy"))
    mask = np.load(os.path.join(NUMPY_DIR, "train_source_mask.npy"))
    y_ids = np.load(os.path.join(NUMPY_DIR, "train_y_ids.npy"))
    lm_labels = np.load(os.path.join(NUMPY_DIR, "train_lm_labels.npy"))

    batch_size = ffconfig.batch_size
    input_ids_shape = (batch_size, ids.shape[1])
    attention_mask_shape = (batch_size, mask.shape[1])
    decoder_input_ids_shape = (batch_size, y_ids.shape[1])
    input_tensors = [
        ffmodel.create_tensor(input_ids_shape, DataType.DT_INT64),          # input_ids
        ffmodel.create_tensor(attention_mask_shape, DataType.DT_INT64),     # attention_mask
        ffmodel.create_tensor(decoder_input_ids_shape, DataType.DT_INT64),  # decoder_input_ids
    ]
    encoder_seq_length = ids.shape[1]
    decoder_seq_length = y_ids.shape[1]
    seq_length = (encoder_seq_length, decoder_seq_length)
    input_names = ["input_ids", "attention_mask", "decoder_input_ids"]

    print("Tracing the model...")
    hf_model = PyTorchModel(
        model, is_hf_model=True, input_names=input_names,
        batch_size=batch_size, seq_length=seq_length,
    )
    output_tensors = hf_model.torch_to_ff(ffmodel, input_tensors, verbose=True)
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
    input_ids_dl = ffmodel.create_data_loader(input_tensors[0], ids)
    attention_mask_dl = ffmodel.create_data_loader(input_tensors[1], mask)
    decoder_input_ids_dl = ffmodel.create_data_loader(input_tensors[2], y_ids)
    # NOTE: We cast down the label tensor data to 32-bit to accommodate the
    # label tensor's required dtype
    labels_dl = ffmodel.create_data_loader(
        ffmodel.label_tensor, lm_labels.astype("int32")
    )

    print("Initializing model layers...")
    ffmodel.init_layers()

    print("Training...")
    epochs = ffconfig.epochs
    ffmodel.fit(
        x=[input_ids_dl, attention_mask_dl, decoder_input_ids_dl],
        y=labels_dl, batch_size=batch_size, epochs=epochs,
    )


if __name__ == "__main__":
    # Generate the .tsv files if needed
    if not os.path.exists(os.path.join(DATA_DIR, "train.tsv")) or \
            not os.path.exists(os.path.join(DATA_DIR, "eval.tsv")):
        DataPreparer.data_to_tsv()
    # Convert the .tsv files to .npy if needed
    if not os.path.exists(NUMPY_DIR):
        os.mkdir(NUMPY_DIR)
    prefixes = ["train_", "eval_"]
    suffixes = ["source_ids.npy", "source_mask.npy", "target_ids.npy"]
    npy_filenames = [
        pre + suf for pre, suf in itertools.product(prefixes, suffixes)
    ]
    if any(
        not os.path.exists(os.path.join(NUMPY_DIR, filename))
        for filename in npy_filenames
    ):
        data_to_numpy()
    # Preprocess the training data if needed
    if not os.path.exists(os.path.join(NUMPY_DIR, "train_y_ids.npy")) or \
            not os.path.exists(os.path.join(NUMPY_DIR, "train_lm_labels.npy")):
        preprocess_train()
    top_level_task()
