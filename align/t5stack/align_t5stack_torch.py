import copy
import os
import time
from typing import Any, Callable
import sys

import numpy as np
import torch
from transformers import T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from transformers.modeling_outputs import Seq2SeqLMOutput

sys.path.append("./align/")
from align_utils import BATCH_SIZE, gen_tensor

try:
    assert torch.cuda.is_available(), "Expects at least one GPU"
    DEVICE = torch.device(0)
except AssertionError as e:
    print("AssertionError", e)
SEQ_LENGTH = 5
OUT_DIR = "align/t5stack/out/"


def construct_encoder_t5stack():
    """Constructs an encoder ``T5Stack`` based on  "google/mt5-small"."""
    config = T5Config(
        vocab_size=250112,
        d_model=512,
        d_kv=64,
        d_ff=1024,
        num_layers=8,
        num_decoder_layers=8,
        num_heads=6,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_eps=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        use_cache=True,
    )
    shared = torch.nn.Embedding(config.vocab_size, config.d_model)
    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    return T5Stack(encoder_config, shared)


def init_encoder_t5_stack_params(t5stack: T5Stack):
    """Initializes the parameters of the given ``T5Stack`` instance."""
    ...


def run():
    t5stack = construct_encoder_t5stack()
    num_embeddings = t5stack.config.vocab_size
    embedding_dim = t5stack.d_model

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=num_embeddings,
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, embedding_dim),
        dtype="float32",
    ).to(DEVICE)

    





if __name__ == "__main__":
    run()
