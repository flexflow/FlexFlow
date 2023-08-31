import os
import sys

import torch
sys.path.append("./align/")
from align_utils import gen_tensor, parse_create_tensor_args, create_general_test_tensor_torch, BATCH_SIZE, INPUT_SIZE, SEQ_LENGTH

assert torch.cuda.is_available(), "Expects at least one GPU"
DEVICE = torch.device(0)
param_weight_op = {'conv2d', 'embedding', 'view_embedding', 'linear'}
param_bias_op = {'conv2d', 'linear'}


def create_single_operator_torch():
    args = parse_create_tensor_args()
    operator_name = args.operator
    OUT_DIR = os.path.join("tests", "align", "out", operator_name)

    if operator_name == 'add':
        label, output = create_tensors_for_add_torch()
    elif operator_name == 'concat':
        label, output = create_tensors_for_concat_torch()
    elif operator_name == 'conv2d':
        label, output, weight, bias = create_tensors_for_conv2d_torch(
            param_dir=OUT_DIR)
    elif operator_name == 'cos':
        label, output = create_tensors_for_cos_torch()
    elif operator_name == 'embedding':
        label, output, weight = create_tensors_for_embedding_torch(param_dir=OUT_DIR)
    elif operator_name == 'exp':
        label, output = create_tensors_for_exp_torch()
    elif operator_name == 'flat':
        label, output = create_tensors_for_flat_torch()
    elif operator_name == 'getitem':
        return create_tensors_for_getitem_torch(param_dir=OUT_DIR)
    elif operator_name == 'identity':
        label, output = create_tensors_for_identity_torch()
    elif operator_name == 'layernorm':
        label, output = create_tensors_for_layernorm_torch(param_dir=OUT_DIR);
    elif operator_name == 'linear':
        label, output, weight, bias = create_tensors_for_linear_torch(param_dir=OUT_DIR);
    elif operator_name == 'multiply':
        label, output = create_tensors_for_multiply_torch()
    elif operator_name == 'pool2d':
        label, output = create_tensors_for_pool2d_torch()
    elif operator_name == 'reducesum':
        label, output = create_tensors_for_reducesum_torch()
    elif operator_name == 'relu':
        label, output = create_tensors_for_relu_torch()
    elif operator_name == 'reshape':
        label, output = create_tensors_for_reshape_torch()
    elif operator_name == 'scalar_add':
        label, output = create_tensors_for_scalar_add_torch()
    elif operator_name == 'scalar_multiply':
        label, output = create_tensors_for_scalar_multiply_torch()
    elif operator_name == 'scalar_sub':
        label, output = create_tensors_for_scalar_sub_torch()
    elif operator_name == 'scalar_truediv':
        label, output = create_tensors_for_scalar_truediv_torch()
    elif operator_name == 'sigmoid':
        label, output = create_tensors_for_sigmoid_torch()
    elif operator_name == 'sin':
        label, output = create_tensors_for_sin_torch()
    elif operator_name == 'subtract':
        label, output = create_tensors_for_subtract_torch()
    elif operator_name == 'tanh':
        label, output = create_tensors_for_tanh_torch()
    elif operator_name == 'transpose':
        label, output = create_tensors_for_transpose_torch()
    elif operator_name == 'view_embedding':
        label, output, weight = create_tensors_for_scalar_view_embedding_torch(param_dir=OUT_DIR)
    elif operator_name == 'max':
        label, output = create_tensors_for_max_torch()
    elif operator_name == 'min':
        label, output = create_tensors_for_min_torch()
    elif operator_name == 'gather':
        label, output = create_tensors_for_gather_torch()
    else:
        raise ValueError('Not Include such Operator in Aligment Test ', operator_name)

    output.retain_grad()
    loss_fn = torch.nn.MSELoss(reduction="mean")
    loss = loss_fn(output, label)
    loss.backward()

    # save tensors to file
    torch.save(output.cpu(), os.path.join(OUT_DIR, "torch_out.pt"))
    torch.save(output.grad.cpu(), os.path.join(OUT_DIR, "torch_out_grad.pt"))

    if (operator_name in param_weight_op):
        torch.save(weight.grad.cpu(), os.path.join(
            OUT_DIR, "torch_weight_grad.pt"))
    if (operator_name in param_bias_op):
        torch.save(bias.grad.cpu(), os.path.join(
            OUT_DIR, "torch_bias_grad.pt"))

# run tests for all operators


def create_tensors_for_add_torch():
    inp1 = create_general_test_tensor_torch().to(DEVICE)
    inp2 = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.add(
        input=inp1,
        other=inp2
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_concat_torch():
    inp1 = create_general_test_tensor_torch().to(DEVICE)
    inp2 = create_general_test_tensor_torch().to(DEVICE)
    inp3 = create_general_test_tensor_torch().to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH * 3, INPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)
    output = torch.cat(
        tensors=(inp1, inp2, inp3),
        dim=1
    ).to(DEVICE)
    output.requires_grad = True

    return label, output


def create_tensors_for_conv2d_torch(param_dir):
    KERNEL_SIZE = 3
    IN_CHANNELS = 3
    OUTPUT_SIZE = 510
    OUT_CHANNELS = 5
    conv2d = torch.nn.Conv2d(
        in_channels=IN_CHANNELS,
        out_channels=OUT_CHANNELS,
        kernel_size=KERNEL_SIZE
    ).to(DEVICE)

    linear_weight = torch.load(os.path.join(param_dir, "ff_weight.pt"))
    linear_bias = torch.load(os.path.join(param_dir, "ff_bias.pt"))
    assert conv2d.weight.shape == linear_weight.shape, (
        "Shape mismatch: " f"FF={linear_weight.shape} torch={conv2d.weight.shape}"
    )
    assert conv2d.bias.shape == linear_bias.shape, (
        "Shape mismatch: " f"FF={linear_bias.shape} torch={conv2d.bias.shape}"
    )

    conv2d.weight = torch.nn.Parameter(linear_weight.to(DEVICE))
    conv2d.bias = torch.nn.Parameter(linear_bias.to(DEVICE))

    # generate input/label tensors
    # imitating 3-channel image input
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, 3, INPUT_SIZE, INPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, 5, OUTPUT_SIZE, OUTPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)

    output = conv2d(inp)
    conv2d.zero_grad()

    return label, output, conv2d.weight, conv2d.bias


def create_tensors_for_cos_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.cos(
        input=inp,
    ).to(DEVICE)
    output.requires_grad = True

    return label, output



def create_tensors_for_embedding_torch(param_dir):
    NUM_EMBEDDINGS = 250112
    EMBEDDING_DIM = 512
    embedding = torch.nn.Embedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        device=DEVICE,
    )
    embedding_weight = torch.load(os.path.join(param_dir, "ff_weight.pt"))
    assert embedding_weight.shape == embedding.weight.shape
    embedding.weight = torch.nn.Parameter(embedding_weight.to(DEVICE))

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=NUM_EMBEDDINGS,
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_DIM),
        dtype="float32",
    ).to(DEVICE)

    output = embedding(inp)
    embedding.zero_grad()
    return label, output, embedding.weight


def create_tensors_for_exp_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.exp(
        input=inp,
    ).to(DEVICE)
    output.requires_grad = True
    
    return label, output


def create_tensors_for_flat_torch():
    INPUT_SIZE_2 = 512
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE, INPUT_SIZE_2),
        dtype="float32"
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH * INPUT_SIZE * INPUT_SIZE_2),
        dtype="float32"
    ).to(DEVICE)

    """todo start/end dim"""
    output = torch.flatten(input=inp, start_dim=1, end_dim=3).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_getitem_torch(param_dir):
    attention_mask = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="float32",
        low=0,
        high=2,
    ).to(DEVICE)
    extended_attention_mask = attention_mask[:, None, None, :]
    torch.save(extended_attention_mask.cpu(), os.path.join(param_dir, "torch_out.pt"))


def create_tensors_for_identity_torch():
    identity = torch.nn.Identity()

    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)

    output = identity(input=inp).to(DEVICE)
    identity.zero_grad()
    output.requires_grad = True
    return label, output


def create_tensors_for_layernorm_torch(param_dir):
    HIDDEN_SIZE = 512
    EPS = 1e-6
    layernorm = torch.nn.LayerNorm(
        normalized_shape=HIDDEN_SIZE,
        eps=EPS,
        elementwise_affine=True,
    ).to(DEVICE)
    layernorm_weight = torch.load(os.path.join(param_dir, "ff_weight.pt"))
    layernorm_bias = torch.load(os.path.join(param_dir, "ff_bias.pt"))
    assert layernorm.weight.shape == layernorm_weight.shape, (
        "Shape mismatch: " f"FF={layernorm_weight.shape} torch={layernorm.weight.shape}"
    )
    assert layernorm.bias.shape == layernorm_bias.shape, (
        "Shape mismatch: " f"FF={layernorm_bias.shape} torch={layernorm.bias.shape}"
    )
    layernorm.weight = torch.nn.Parameter(layernorm_weight.to(DEVICE))
    layernorm.bias = torch.nn.Parameter(layernorm_bias.to(DEVICE))

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, HIDDEN_SIZE),
        dtype="float32",
    ).to(DEVICE)

    output = layernorm(inp)
    layernorm.zero_grad()
    return label, output


def create_tensors_for_linear_torch(param_dir):
    OUTPUT_SIZE = 128
    linear = torch.nn.Linear(
        in_features=512,
        out_features=128
    ).to(DEVICE)

    # get weight/bias from ff files, check same shape
    linear_weight = torch.load(os.path.join(param_dir, "ff_weight.pt"))
    linear_bias = torch.load(os.path.join(param_dir, "ff_bias.pt"))
    assert linear.weight.shape == linear_weight.shape, (
        "Shape mismatch: " f"FF={linear_weight.shape} torch={linear.weight.shape}"
    )
    assert linear.bias.shape == linear_bias.shape, (
        "Shape mismatch: " f"FF={linear_bias.shape} torch={linear.bias.shape}"
    )

    # set weight/bias
    linear.weight = torch.nn.Parameter(linear_weight.to(DEVICE))
    linear.bias = torch.nn.Parameter(linear_bias.to(DEVICE))

    # generate input/label tensors w/ gen_tensor
    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, INPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)
    
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, OUTPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)

    # get output running input through layer
    output = linear(inp)
    linear.zero_grad()

    return label, output, linear.weight, linear.bias 


def create_tensors_for_multiply_torch():
    inp1 = create_general_test_tensor_torch().to(DEVICE)
    inp2 = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.mul(
        input=inp1,
        other=inp2
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_pool2d_torch():
    KERNEL_SIZE = 3
    IN_CHANNELS = 3
    OUTPUT_SIZE = 510

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, IN_CHANNELS, OUTPUT_SIZE, OUTPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)
    pool2d = torch.nn.MaxPool2d(
        kernel_size=KERNEL_SIZE, stride=1, padding=0).to(DEVICE)

    output = pool2d(inp)
    output.requires_grad = True
    pool2d.zero_grad()
    return label, output


def create_tensors_for_reducesum_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, INPUT_SIZE),
        dtype="float32"
    ).to(DEVICE)

    output = torch.sum(
        input=inp,
        dim=1,
        keepdim=False
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_relu_torch():
    relu = torch.nn.ReLU(inplace=True)

    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)

    output = relu(input=inp).to(DEVICE)
    relu.zero_grad()
    output.requires_grad = True
    return label, output


def create_tensors_for_reshape_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, INPUT_SIZE, SEQ_LENGTH),
        dtype="float32"
    ).to(DEVICE)
    output = torch.reshape(
        input=inp,
        shape=(BATCH_SIZE, INPUT_SIZE, SEQ_LENGTH)
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_scalar_add_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.add(
        input=inp,
        other=1,
        alpha=1
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_scalar_multiply_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.mul(
        input=inp,
        other=2
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_scalar_sub_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.sub(
        input=inp,
        other=1
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_scalar_truediv_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.div(
        input=inp,
        other=2
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_sigmoid_torch():
    sigmoid = torch.nn.Sigmoid()

    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)

    output = sigmoid(input=inp).to(DEVICE)
    sigmoid.zero_grad()
    output.requires_grad = True
    return label, output


def create_tensors_for_sin_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.sin(
        input=inp,
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_subtract_torch():
    inp1 = create_general_test_tensor_torch().to(DEVICE)
    inp2 = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.sub(
        input=inp1,
        other=inp2
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_tanh_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.tanh(
        input=inp,
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_transpose_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, INPUT_SIZE, SEQ_LENGTH),
        dtype="float32"
    ).to(DEVICE)
    output = torch.transpose(
        input=inp,
        dim0=1,
        dim1=2,
    ).to(DEVICE)
    output.requires_grad = True
    return label, output


def create_tensors_for_scalar_view_embedding_torch(param_dir):
    NUM_EMBEDDINGS = 250112
    EMBEDDING_DIM = 512
    embedding = torch.nn.Embedding(
        num_embeddings=NUM_EMBEDDINGS,
        embedding_dim=EMBEDDING_DIM,
        device=DEVICE,
    )
    embedding_weight = torch.load(os.path.join(param_dir, "ff_weight.pt"))
    assert embedding_weight.shape == embedding.weight.shape, \
        "Shape mismatch: " \
        f"FF={embedding_weight.shape} torch={embedding.weight.shape}"
    embedding.weight = torch.nn.Parameter(embedding_weight.to(DEVICE))

    inp: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH),
        dtype="int64",
        low=0,
        high=NUM_EMBEDDINGS,
    ).to(DEVICE)
    label: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, EMBEDDING_DIM),
        dtype="float32",
    ).to(DEVICE)

    output = embedding(inp.view(-1, inp.shape[-1]))
    embedding.zero_grad()
    return label, output, embedding.weight

def create_tensors_for_max_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    oth = create_general_test_tensor_torch().add(1).to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.maximum(
        input=inp,
        other=oth
    ).to(DEVICE)
    output.requires_grad = True
    return label, output
   
    

def create_tensors_for_min_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    oth = create_general_test_tensor_torch().add(1).to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.minimum(
        input=inp,
        other=oth
    ).to(DEVICE)
    output.requires_grad = True
    return label, output

def create_tensors_for_gather_torch():
    inp = create_general_test_tensor_torch().to(DEVICE)
    index = torch.zeros(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE, dtype=torch.int64).to(DEVICE)
    label = create_general_test_tensor_torch().to(DEVICE)
    output = torch.gather(
        input=inp,
        index=index,
        dim=0
    ).to(DEVICE)
    output.requires_grad = True
    return label, output

if __name__ == "__main__":
    create_single_operator_torch()
