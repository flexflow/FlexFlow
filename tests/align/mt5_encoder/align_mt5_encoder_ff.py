import os
import sys

from flexflow.core import *

sys.path.append("./align/")
from align_ff_utils import run_fwd_bwd
from mt5_ff_utils import init_ff_mt5_encoder

# NOTE: We use the PyTorch mT5 encoder output as the labels
ENCODER_LABELS_PATH = os.path.join(
    "align", "mt5_encoder", "out", "hidden_states.pt",
)


def run():
    assert os.path.exists(ENCODER_LABELS_PATH), \
        "Make sure to generate the encoder labels file (e.g. by modifying " \
        "the transformers library source code)"
    ffmodel, input_dls, label_dl = init_ff_mt5_encoder(
        ENCODER_LABELS_PATH,
    )
    run_fwd_bwd(ffmodel, ffmodel._ffconfig, input_dls, label_dl)


if __name__ == "__main__":
    run()
