import os
import sys
import torch
import json
from flexflow.core import *
from flexflow.core.flexflow_cffi import Linear, Op, Parameter
from flexflow.type import AggrMode
from flexflow.torch.model import GetItemNode, FunctionNode
sys.path.append("./align/")

from align_utils import parse_create_tensor_args, gen_tensor, create_general_test_tensor_torch, BATCH_SIZE, INPUT_SIZE, SEQ_LENGTH
from align_ff_utils import (compile_ffmodel, init_ffmodel, run_fwd_bwd,
                            save_param_ff, save_param_grad_ff, save_tensor_ff,
                            save_tensor_grad_ff)


# set of operaters that have weight/bias
param_weight_op = {'conv2d': Conv2D, 'embedding': Embedding,
                   'layernorm': LayerNorm, 'view_embedding': Embedding, 'linear': Linear}
param_bias_op = {'conv2d': Conv2D, 'layernorm': LayerNorm, 'linear': Linear}


def top_level_task():
    args = parse_create_tensor_args()
    configs_dict = None
    if args.config_file is not None:
        with open(args.config_file) as f:
            configs_dict = json.load(f)
    init_flexflow_runtime(configs_dict)

    operator_name = args.operator
    OUT_DIR = os.path.join("tests", "align", "out", operator_name)

    ffconfig = FFConfig()
    ffmodel = FFModel(ffconfig)

    if operator_name == 'add':
        input_tensors, label, output_tensor = create_tensors_for_add_ff(
            ffmodel)
    elif operator_name == 'concat':
        input_tensors, label, output_tensor = create_tensors_for_concat_ff(
            ffmodel)
    elif operator_name == 'conv2d':
        input_tensors, label, output_tensor = create_tensors_for_conv2d_ff(
            ffmodel)
    elif operator_name == 'cos':
        input_tensors, label, output_tensor = create_tensors_for_cos_ff(
            ffmodel)
    elif operator_name == 'embedding':
        input_tensors, label, output_tensor = create_tensors_for_embedding_ff(
            ffmodel)
    elif operator_name == 'exp':
        input_tensors, label, output_tensor = create_tensors_for_exp_ff(
            ffmodel)
    elif operator_name == 'flat':
        input_tensors, label, output_tensor = create_tensors_for_flat_ff(
            ffmodel)
    elif operator_name == 'getitem':
        input_tensors, label, output_tensor = create_tensors_for_getitem_ff(
            ffmodel)
    elif operator_name == 'identity':
        input_tensors, label, output_tensor = create_tensors_for_identity_ff(
            ffmodel)
    elif operator_name == 'layernorm':
        input_tensors, label, output_tensor = create_tensors_for_layernorm_ff(
            ffmodel)
    elif operator_name == 'linear':
        input_tensors, label, output_tensor = create_tensors_for_linear_ff(
            ffmodel)
    elif operator_name == 'multiply':
        input_tensors, label, output_tensor = create_tensors_for_multiply_ff(
            ffmodel)
    elif operator_name == 'pool2d':
        input_tensors, label, output_tensor = create_tensors_for_pool2d_ff(
            ffmodel)
    elif operator_name == 'reducesum':
        input_tensors, label, output_tensor = create_tensors_for_reducesum_ff(
            ffmodel)
    elif operator_name == 'relu':
        input_tensors, label, output_tensor = create_tensors_for_relu_ff(
            ffmodel)
    elif operator_name == 'reshape':
        input_tensors, label, output_tensor = create_tensors_for_reshape_ff(
            ffmodel)
    elif operator_name == 'scalar_add':
        input_tensors, label, output_tensor = create_tensors_for_scalar_add_ff(
            ffmodel)
    elif operator_name == 'scalar_multiply':
        input_tensors, label, output_tensor = create_tensors_for_scalar_multiply_ff(
            ffmodel)
    elif operator_name == 'scalar_sub':
        input_tensors, label, output_tensor = create_tensors_for_scalar_sub_ff(
            ffmodel)
    elif operator_name == 'scalar_truediv':
        input_tensors, label, output_tensor = create_tensors_for_scalar_truediv_ff(
            ffmodel)
    elif operator_name == 'sigmoid':
        input_tensors, label, output_tensor = create_tensors_for_sigmoid_ff(
            ffmodel)
    elif operator_name == 'sin':
        input_tensors, label, output_tensor = create_tensors_for_sin_ff(
            ffmodel)
    elif operator_name == 'subtract':
        input_tensors, label, output_tensor = create_tensors_for_subtract_ff(
            ffmodel)
    elif operator_name == 'tanh':
        input_tensors, label, output_tensor = create_tensors_for_tanh_ff(
            ffmodel)
    elif operator_name == 'transpose':
        input_tensors, label, output_tensor = create_tensors_for_transpose_ff(
            ffmodel)
    elif operator_name == 'view_embedding':
        input_tensors, label, output_tensor = create_tensors_for_view_embedding_ff(
            ffmodel)
    elif operator_name == 'max':
            input_tensors, label, output_tensor = create_tensors_for_max_ff(
            ffmodel)
    elif operator_name == 'min':
        input_tensors, label, output_tensor = create_tensors_for_min_ff(
            ffmodel)
    elif operator_name == 'gather':
        input_tensors, label, output_tensor = create_tensors_for_gather_ff(
            ffmodel)
    else:
        raise ValueError(
            'Not include such Operator in Aligment Test', operator_name)

    compile_ffmodel(ffmodel)
    dataloaders = init_ffmodel(ffmodel, input_tensors, label)
    assert len(dataloaders) == len(input_tensors) + 1

    input_dataloaders = dataloaders[0: len(dataloaders) - 1]
    label_dataloaders = dataloaders[len(dataloaders) - 1]
    # forward/backward pass
    run_fwd_bwd(ffmodel, ffconfig, input_dataloaders, label_dataloaders)
    # save data
    save_tensor_ff(output_tensor, ffmodel, os.path.join(OUT_DIR, "ff_out.pt"))
    save_tensor_grad_ff(output_tensor, ffmodel,
                        os.path.join(OUT_DIR, "ff_out_grad.pt"))

    # save weight and bias tensor for some operators
    layer: Op = ffmodel.get_layers()[0]
    if operator_name == 'view_embedding':
        layer = ffmodel.get_layers()[1]

    if operator_name in param_weight_op:
        assert isinstance(layer, param_weight_op[operator_name])
        weight: Parameter = layer.get_weight_tensor()
        save_param_ff(weight, ffmodel, os.path.join(OUT_DIR, "ff_weight.pt"))
        save_param_grad_ff(weight, ffmodel, os.path.join(
            OUT_DIR, "ff_weight_grad.pt"))
    if operator_name in param_bias_op:
        assert isinstance(layer, param_bias_op[operator_name])
        bias: Parameter = layer.get_bias_tensor()
        save_param_ff(bias, ffmodel, os.path.join(OUT_DIR, "ff_bias.pt"))
        save_param_grad_ff(bias, ffmodel, os.path.join(
            OUT_DIR, "ff_bias_grad.pt"))


def create_tensors_for_add_ff(ffmodel):
    inp1 = create_general_test_tensor_torch()
    inp2 = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()
    input_tensor_1 = ffmodel.create_tensor(inp1.shape, DataType.DT_FLOAT)
    input_tensor_2 = ffmodel.create_tensor(inp2.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.add(
        x=input_tensor_1,
        y=input_tensor_2,
        name="add"
    )
    return ((input_tensor_1, inp1), (input_tensor_2, inp2)), label, output_tensor


def create_tensors_for_concat_ff(ffmodel):
    inp1 = create_general_test_tensor_torch()
    inp2 = create_general_test_tensor_torch()
    inp3 = create_general_test_tensor_torch()
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH * 3, INPUT_SIZE),
        dtype="float32"
    )

    input_tensor_1 = ffmodel.create_tensor(inp1.shape, DataType.DT_FLOAT)
    input_tensor_2 = ffmodel.create_tensor(inp2.shape, DataType.DT_FLOAT)
    input_tensor_3 = ffmodel.create_tensor(inp3.shape, DataType.DT_FLOAT)

    output_tensor = ffmodel.concat(
        tensors=[input_tensor_1, input_tensor_2, input_tensor_3],
        axis=1,
        name="concat"
    )
    return ((input_tensor_1, inp1), (input_tensor_2, inp2), (input_tensor_3, inp3)), label, output_tensor


def create_tensors_for_conv2d_ff(ffmodel):
    KERNEL_SIZE = 3
    INPUT_SIZE = 512
    IN_CHANNELS = 3
    OUTPUT_SIZE = 510
    OUT_CHANNELS = 5

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE),
        dtype="float32"
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, OUT_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE),
        dtype="float32"
    )
    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.conv2d(
        input=input_tensor,
        out_channels=OUT_CHANNELS,
        kernel_h=KERNEL_SIZE,
        kernel_w=KERNEL_SIZE,
        stride_h=1,
        stride_w=1,
        padding_h=0,
        padding_w=0,
        name="conv2d"
    )
    return ((input_tensor, inp), ), label, output_tensor


def create_tensors_for_cos_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.cos(
        x=input_tensor,
        name="cos"
    )
    return ((input_tensor, inp), ), label, output_tensor


def create_tensors_for_divide_ff(ffmodel):
    inp1 = create_general_test_tensor_torch()
    inp2 = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()
    input_tensor_1 = ffmodel.create_tensor(inp1.shape, DataType.DT_FLOAT)
    input_tensor_2 = ffmodel.create_tensor(inp2.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.divide(
        x=input_tensor_1,
        y=input_tensor_2,
        name="divide"
    )
    return ((input_tensor_1, inp1), (input_tensor_2, inp2)), label, output_tensor


def create_tensors_for_embedding_ff(ffmodel):
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

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_INT64)
    output_tensor = ffmodel.embedding(
        input=input_tensor,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        aggr=AggrMode.AGGR_MODE_NONE,
        kernel_initializer=NormInitializer(seed=42, mean=0, stddev=1),
        name="embedding",
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_exp_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.exp(
        x=input_tensor,
        name="exp"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_flat_ff(ffmodel):
    INPUT_SIZE_2 = 512

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE, INPUT_SIZE_2),
        dtype="float32"
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, INPUT_SIZE * INPUT_SIZE_2 * SEQ_LENGTH),
        dtype="float32"
    )

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.flat(
        input=input_tensor,
        name="flat"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_getitem_ff(ffmodel):
    attention_mask = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="float32",
        low=0,
        high=2,
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="float32",
    )  # unused

    attention_mask_tensor = ffmodel.create_tensor(
        attention_mask.shape,
        DataType.DT_FLOAT,
    )
    extended_attention_mask = GetItemNode.slice_tensor(
        ffmodel,
        attention_mask_tensor,
        (slice(None, None, None), None, None, slice(None, None, None)),
        "slice",
    )
    return ((attention_mask_tensor, attention_mask),), label, extended_attention_mask


def create_tensors_for_identity_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.identity(
        input=input_tensor,
        name="identity"
    )
    return ((input_tensor, inp),), label, output_tensor

def create_tensors_for_layernorm_ff(ffmodel):
    HIDDEN_SIZE = 512
    EPS = 1e-6
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    )

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.layer_norm(
        input=input_tensor,
        axes=[len(input_tensor.dims) - 1],  # normalize over the last dimension
        elementwise_affine=True,
        eps=EPS,
        name="layernorm",
    )
    return ((input_tensor, inp),), label, output_tensor

def create_tensors_for_linear_ff(ffmodel):
  OUTPUT_SIZE = 128
  inp: torch.Tensor = gen_tensor(
      (BATCH_SIZE, INPUT_SIZE),
      dtype="float32"
  )
  label: torch.Tensor = gen_tensor(
      (BATCH_SIZE, OUTPUT_SIZE),
      dtype="float32"
  )

  input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
  output_tensor = ffmodel.dense(
      input=input_tensor,
      out_dim=128,
      name="linear"
  )
  return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_multiply_ff(ffmodel):
    inp1 = create_general_test_tensor_torch()
    inp2 = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor_1 = ffmodel.create_tensor(inp1.shape, DataType.DT_FLOAT)
    input_tensor_2 = ffmodel.create_tensor(inp2.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.multiply(
        x=input_tensor_1,
        y=input_tensor_2,
        name="multiply"
    )
    return ((input_tensor_1, inp1), (input_tensor_2, inp2)), label, output_tensor


def create_tensors_for_pool2d_ff(ffmodel):
    KERNEL_SIZE = 3
    IN_CHANNELS = 3
    OUTPUT_SIZE = 510
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE),
        dtype="float32"
    )
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, IN_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE),
        dtype="float32"
    )

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)

    output_tensor = ffmodel.pool2d(
        input=input_tensor,
        kernel_h=KERNEL_SIZE,
        kernel_w=KERNEL_SIZE,
        stride_h=1,
        stride_w=1,
        padding_h=0,
        padding_w=0,
        name="pool2d"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_reducesum_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, INPUT_SIZE),
        dtype="float32"
    )

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.reduce_sum(
        input=input_tensor,
        axes=(1,),
        keepdims=False,
        name="reducesum"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_relu_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.relu(
        input=input_tensor,
        name="relu"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_reshape_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, INPUT_SIZE, SEQ_LENGTH),
        dtype="float32"
    )

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.reshape(
        input=input_tensor,
        shape=(BATCH_SIZE, INPUT_SIZE, SEQ_LENGTH),
        name="reshape"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_scalar_add_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.scalar_add(
        input=input_tensor,
        scalar=1,
        inplace=False,
        name="scalar_add"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_scalar_multiply_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.scalar_multiply(
        input=input_tensor,
        scalar=2,
        inplace=False,
        name="scalar_multiply"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_scalar_sub_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.scalar_sub(
        input=input_tensor,
        scalar=1,
        inplace=False,
        name="scalar_sub"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_scalar_truediv_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()
    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.scalar_true_divide(
        input=input_tensor,
        scalar=2,
        inplace=False,
        name="scalar_truediv"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_sigmoid_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.sigmoid(
        input=input_tensor,
        name="sigmoid"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_sin_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()
    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.sin(
        x=input_tensor,
        name="sin"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_subtract_ff(ffmodel):
    inp1 = create_general_test_tensor_torch()
    inp2 = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor_1 = ffmodel.create_tensor(inp1.shape, DataType.DT_FLOAT)
    input_tensor_2 = ffmodel.create_tensor(inp2.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.subtract(
        x=input_tensor_1,
        y=input_tensor_2,
        name="subtract"
    )
    return ((input_tensor_1, inp1), (input_tensor_2, inp2)), label, output_tensor


def create_tensors_for_tanh_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.tanh(
        input=input_tensor,
        name="tanh"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_transpose_ff(ffmodel):
    inp = create_general_test_tensor_torch()
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, INPUT_SIZE, SEQ_LENGTH),
        dtype="float32"
    )

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.transpose(
        input=input_tensor,
        perm=(0, 2, 1),
        name="sin"
    )
    return ((input_tensor, inp),), label, output_tensor


def create_tensors_for_view_embedding_ff(ffmodel):
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

    input_tensor = ffmodel.create_tensor(inp.shape, DataType.DT_INT64)
    # Treat `view()` as a special case of `reshape()`
    view_tensor = ffmodel.reshape(
        input=input_tensor,
        shape=FunctionNode.get_view_shape(input_tensor, (-1, inp.shape[-1])),
        name="view",
    )
    output_tensor = ffmodel.embedding(
        input=view_tensor,
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        aggr=AggrMode.AGGR_MODE_NONE,
        kernel_initializer=NormInitializer(seed=42, mean=0, stddev=1),
        name="embedding",
    )
    return ((input_tensor, inp),), label, output_tensor

def create_tensors_for_max_ff(ffmodel):
    inp1 = create_general_test_tensor_torch()
    inp2 = create_general_test_tensor_torch().add(1)
    
    label = create_general_test_tensor_torch()

    input_tensor_1 = ffmodel.create_tensor(inp1.shape, DataType.DT_FLOAT)
    input_tensor_2 = ffmodel.create_tensor(inp2.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.max(
        x=input_tensor_1,
        y=input_tensor_2,
        name="max"
    )
    
    return ((input_tensor_1, inp1),(input_tensor_2, inp2)), label, output_tensor

def create_tensors_for_min_ff(ffmodel):
    inp1 = create_general_test_tensor_torch()
    inp2 = create_general_test_tensor_torch().add(1)
    
    label = create_general_test_tensor_torch()

    input_tensor_1 = ffmodel.create_tensor(inp1.shape, DataType.DT_FLOAT)
    input_tensor_2 = ffmodel.create_tensor(inp2.shape, DataType.DT_FLOAT)
    output_tensor = ffmodel.min(
        x=input_tensor_1,
        y=input_tensor_2,
        name="max"
    )
    return ((input_tensor_1, inp1),(input_tensor_2, inp2)), label, output_tensor

def create_tensors_for_gather_ff(ffmodel):
    inp1 = create_general_test_tensor_torch()
    index = torch.zeros(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE, dtype=torch.int64)
    
    label = create_general_test_tensor_torch()

    input_tensor = ffmodel.create_tensor(inp1.shape, DataType.DT_FLOAT)
    index_tensor = ffmodel.create_tensor(index.shape, DataType.DT_INT64)
    output_tensor = ffmodel.gather(
        input=input_tensor,
        index=index_tensor,
        dim=0,
        name="gather"
    )
    return ((input_tensor, inp1),(index_tensor, index)), label, output_tensor




if __name__ == "__main__":
    top_level_task()
