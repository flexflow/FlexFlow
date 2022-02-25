import os
import sys

import torch
from flexflow.core import *
from flexflow.core.flexflow_cffi import Embedding, Op, Parameter
from flexflow.type import AggrMode

sys.path.append("examples/python/pytorch/mt5/debug")
from mt5_ff_utils import init_ff_mt5_encoder

# TODO: ...
# NOTE: We use the PyTorch mT5 encoder output as the labels
OUTPUT_DIR = os.path.join(BASE_DIR, "debug", "output")
ENCODER_LABELS_PATH = os.path.join(OUTPUT_DIR, "hidden_states.pt")


def top_level_task():
    ffmodel, node_to_output, input_dls, labels_dl = init_ff_mt5_encoder()