import os
import sys

import numpy as np
import torch
from transformers import T5Tokenizer

sys.path.append("./examples/python/pytorch/mt5")
from mt5_torch import get_dataloaders

BASE_DIR = "examples/python/pytorch/mt5"
DATA_DIR = os.path.join(BASE_DIR, "data")
NUMPY_DIR = os.path.join(DATA_DIR, "numpy")


def torch_debug():
    model_params = {
        "SEED": 42,
        "MODEL": "google/mt5-small",
        "TRAIN_BATCH_SIZE": 32,
        "EVAL_BATCH_SIZE": 32,
        "TRAIN_EPOCHS": 2,
        "MAX_SOURCE_TEXT_LENGTH": 48,
        "MAX_TARGET_TEXT_LENGTH": 48,
        "LEARNING_RATE": 1e-4,
    }
    tokenizer = T5Tokenizer.from_pretrained(model_params["MODEL"])
    train_loader, _ = get_dataloaders(tokenizer, model_params)
    max_label = None
    min_label = None
    for i, data in enumerate(train_loader):
        y = data["target_ids"].to(dtype=torch.long)
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        _max_label = torch.max(lm_labels)
        _min_label = torch.min(lm_labels)
        max_label = torch.max(max_label, _max_label) if max_label is not None else _max_label
        min_label = torch.min(min_label, _min_label) if min_label is not None else _min_label
    print(f"max label: {max_label:,}")
    print(f"min label: {min_label:,}")


def ff_debug():
    lm_labels = np.load(os.path.join(NUMPY_DIR, "train_lm_labels.npy"))
    lm_label_shape = None
    max_label = float("-inf")
    min_label = float("inf")
    lm_labels = lm_labels.astype(np.int32)
    for i, label in enumerate(lm_labels):
        if lm_label_shape is None:
            lm_label_shape = label.shape
        else:
            assert label.shape == lm_label_shape
        label = np.clip(label, 0, None)
        _max_label = max(label)
        _min_label = min(label)
        max_label = max(max_label, _max_label)
        min_label = min(min_label, _min_label)
    print(f"max label: {max_label:,}")
    print(f"min label: {min_label:,}")


if __name__ == "__main__":
    ff_debug()
    # torch_debug()
