import os
import sys

import torch
from flexflow.core import *
from flexflow.torch.model import GetItemNode

sys.path.append("./align/")
from align_ff_utils import compile_ffmodel, init_ffmodel, run_fwd_bwd, save_tensor_ff
from align_utils import gen_tensor, BATCH_SIZE

SEQ_LENGTH = 5
OUT_DIR = os.path.join("align", "getitem", "out")


def run():
    """Checks the ``getitem()`` code path for tensor slicing."""
    attention_mask = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=2,
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="float32",
    )  # unused

    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)
    attention_mask_tensor = ffmodel.create_tensor(
        attention_mask.shape,
        DataType.DT_INT64,
    )
    extended_attention_mask = GetItemNode.slice_tensor(
        ffmodel,
        attention_mask_tensor,
        (slice(None, None, None), None, None, slice(None, None, None)),
        "slice",
    )

    compile_ffmodel(ffmodel)
    dls = init_ffmodel(
        ffmodel, ((attention_mask_tensor, attention_mask),), label,
    )
    assert len(dls) == 2
    inp_dl, label_dl = dls
    run_fwd_bwd(ffmodel, ffconfig, (inp_dl,), label_dl, run_bwd=False)

    save_tensor_ff(extended_attention_mask, ffmodel, os.path.join(OUT_DIR, "ff_out.pt"))


if __name__ == "__main__":
    run()
