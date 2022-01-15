import os
import time
from typing import Any, Callable

import numpy as np
import torch
from transformers import MT5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

assert torch.cuda.is_available(), "This script requires GPU"
DEVICE = torch.device(0)
PRETRAINED_MODEL_NAME = "google/mt5-small"

BASE_DIR = "examples/python/pytorch/mt5"
DATA_DIR = os.path.join(BASE_DIR, "data")
BATCH_DIR = os.path.join(DATA_DIR, "batch")
INPUT_IDS_PATH = os.path.join(BATCH_DIR, "ids.pt")
ATTENTION_MASK_PATH = os.path.join(BATCH_DIR, "mask.pt")
DECODER_INPUT_IDS_PATH = os.path.join(BATCH_DIR, "y_ids.pt")
LABELS_PATH = os.path.join(BATCH_DIR, "lm_labels.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "debug/output")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def print_l2_norm_hook_ctor(module_name: str):
    def print_l2_norm_hook(module: torch.nn.Module, input: Any, output: Any):
        print(f"{module_name} ", end="")
        if isinstance(output, torch.Tensor):
            print(
                f"[{type(module)}] L2 norm="
                f"{torch.norm(output, p='fro'):.3f}"
            )
        elif hasattr(output, "last_hidden_state"):
            print(
                f"[{type(module)}] L2 norm (last_hidden_state)="
                f"{torch.norm(output.last_hidden_state, p='fro'):.3f}"
            )
        else:
            print(f"[{type(module)}] Unknown output")
    return print_l2_norm_hook


def print_tensor_hook_ctor(module_name: str):
    def print_tensor_hook(module: torch.nn.Module, input: Any, output: Any):
        print(f"{module_name} ", end="")
        if isinstance(output, torch.Tensor):
            print(output)
        elif hasattr(output, "last_hidden_state"):
            print(output.last_hidden_state)
        else:
            print(f"[{type(module)}] Unknown output")
    return print_tensor_hook


def save_tensor_hook_ctor(module_name: str):
    def save_tensor_hook(module: torch.nn.Module, input: Any, output: Any):
        save_path = os.path.join(
            OUTPUT_DIR, f"{module_name.replace('.', '_')}.pt",
        )
        print(f"Saving to {save_path}...")
        if isinstance(output, torch.Tensor):
            torch.save(output, save_path)
        elif hasattr(output, "last_hidden_state"):
            torch.save(output.last_hidden_state, save_path)
        else:
            print(f"[{type(module)}] Unknown output")
    return save_tensor_hook


def null_hook(module: torch.nn.Module, input: Any, output: Any):
    return


def has_forward(module: torch.nn.Module) -> bool:
    """Returns if ``module`` has a forward pass method to instrument."""
    return getattr(module, "forward", None) is not None


def is_leaf_module(module: torch.nn.Module) -> bool:
    return len(list(module.children())) == 0


def load_batch_torch(verbose=False):
    """
    Loads a single batch from disk to be used in :func:`train_step`.

    Returns:
        (input_ids, attention_mask, decoder_input_ids, labels)

        input_ids (torch.LongTensor):
        attention_mask (torch.LongTensor):
        decoder_input_ids (torch.LongTensor):
        labels (torch.LongTensor):
    """
    # Load the data
    input_ids = torch.load(INPUT_IDS_PATH).to(DEVICE)
    attention_mask = torch.load(ATTENTION_MASK_PATH).to(DEVICE)
    decoder_input_ids = torch.load(DECODER_INPUT_IDS_PATH).to(DEVICE)
    labels = torch.load(LABELS_PATH).to(DEVICE)

    # Validate shapes
    assert len(input_ids.shape) == 2
    batch_size, encoder_seq_length = input_ids.shape
    decoder_seq_length = decoder_input_ids.shape[1]
    assert attention_mask.shape[0] == batch_size
    assert decoder_input_ids.shape[0] == batch_size
    assert labels.shape[0] == batch_size
    assert attention_mask.shape[1] == encoder_seq_length
    assert labels.shape[1] == decoder_seq_length

    # Optionally print information
    if verbose:
        print(
            f"Loading a single batch of {batch_size} samples...\n"
            f"input_ids \t\t({torch.typename(input_ids)}):\t"
            f"{input_ids.shape}\n"
            f"attention_mask \t\t({torch.typename(attention_mask)}):\t"
            f"{attention_mask.shape}\n"
            f"decoder_input_ids \t({torch.typename(decoder_input_ids)}):\t"
            f"{decoder_input_ids.shape}\n"
            f"labels \t\t\t({torch.typename(labels)}):\t{labels.shape}\n"
            f"Encoder seq length: {encoder_seq_length}\n"
            f"Decoder seq length: {decoder_seq_length}"
        )
    return (input_ids, attention_mask, decoder_input_ids, labels)


def instrument_model(
    model: torch.nn.Module,
    forward_hook_ctor: Callable[
        [str],
        Callable[[torch.nn.Module, Any, Any], None],
    ],
) -> torch.nn.Module:
    """
    Registers a forward hook constructed based on ``forward_hook_ctor`` in all
    leaf modules in ``model``.
    """
    for name, module in model.named_modules():
        if has_forward(module) and is_leaf_module(module):
            module.register_forward_hook(forward_hook_ctor(name))
    return model


def train_step_torch(
    print_timing: bool = False,
    print_model: bool = False,
):
    """
    Computes one training step, including a forward pass and a backward pass.
    """
    set_seed(42)
    start_time = time.time()
    model = MT5ForConditionalGeneration.from_pretrained(
        PRETRAINED_MODEL_NAME,
    )
    model = model.to(DEVICE)
    model.train()
    model = instrument_model(model, print_tensor_hook_ctor)
    model_load_time = time.time() - start_time
    if print_model:
        # Print each layer in the model in DAG order
        for module in model.modules():
            if is_leaf_module(module):
                print(module)
    if print_timing:
        print(f"Model load:\t{model_load_time:,.3f} s")

    start_time = time.time()
    batch = load_batch_torch(False)
    input_ids: torch.LongTensor = batch[0]
    attention_mask: torch.LongTensor = batch[1]
    decoder_input_ids: torch.LongTensor = batch[2]
    labels: torch.LongTensor = batch[3]
    batch_load_time = time.time() - start_time
    if print_timing:
        print(f"Batch load:\t{batch_load_time:,.3f} s")

    # Forward pass
    start_time = time.time()
    print(f"input_ids ({input_ids.shape}): {input_ids}")
    outputs: Seq2SeqLMOutput = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
    )
    forward_pass_time = time.time() - start_time
    if print_timing:
        print(f"Forward pass:\t{forward_pass_time:,.3f} s")

    # Backward pass
    start_time = time.time()
    loss = outputs.loss
    loss.backward()
    backward_pass_time = time.time() - start_time
    if print_timing:
        print(f"Backward pass:\t{backward_pass_time:,.3f} s")


if __name__ == "__main__":
    train_step_torch(
        print_timing=False,
        print_model=False,
    )
