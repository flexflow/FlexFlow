import os
import sys

import torch
from flexflow.core import *
from flexflow.core.flexflow_cffi import Embedding, Op, Parameter
from flexflow.type import AggrMode

sys.path.append("./align/")
from align_ff_utils import (compile_ffmodel, init_ffmodel, run_fwd_bwd,
                            save_param_ff, save_param_grad_ff, save_tensor_ff,
                            save_tensor_grad_ff)
from align_utils import BATCH_SIZE, gen_tensor

SEQ_LENGTH = 5
OUT_DIR = os.path.join("align", "embedding", "out")


def run():
    NUM_EMBEDDINGS = 250112
    EMBEDDING_DIM = 512
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=NUM_EMBEDDINGS,
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_DIM),
        dtype="float32",
    )

    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)
    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_INT64)
    output_tensor = ffmodel.embedding(
        input=input_tensor,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        aggr=AggrMode.AGGR_MODE_NONE,
        kernel_initializer=NormInitializer(seed=42, mean=0, stddev=1),
        name="embedding",
    )
    compile_ffmodel(ffmodel)
    dls = init_ffmodel(ffmodel, ((input_tensor, inp),), label)
    assert len(dls) == 2
    inp_dl, label_dl = dls
    run_fwd_bwd(ffmodel, ffconfig, (inp_dl,), label_dl)

    embedding_layer: Op = ffmodel.get_layers()[0]
    assert isinstance(embedding_layer, Embedding)
    embedding_weight: Parameter = embedding_layer.get_weight_tensor()
    save_tensor_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out.pt"))
    save_tensor_grad_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out_grad.pt"))
    save_param_ff(embedding_weight, ffmodel, os.path.join(OUT_DIR, "ff_weight.pt"))
    save_param_grad_ff(
        embedding_weight, ffmodel, os.path.join(OUT_DIR, "ff_weight_grad.pt")
    )


if __name__ == "__main__":
    run()
