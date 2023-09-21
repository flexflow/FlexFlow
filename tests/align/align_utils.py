from typing import Any, Dict, Tuple

import numpy as np
import torch
import os

from typing import Iterable, NamedTuple
from argparse import ArgumentParser

BATCH_SIZE = 16
INPUT_SIZE = 512
SEQ_LENGTH = 5

def make_deterministic(seed: int = 42) -> None:
    """Makes ensuing runs determinstic by setting seeds and using deterministic
    backends."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def str_dtype_to_torch_dtype(dtype: str) -> torch.dtype:
    """Converts a string representation of a dtype to the corresponding
    PyTorch dtype."""
    if dtype == "int32":
        return torch.int32
    elif dtype == "int64":
        return torch.int64
    elif dtype == "float32":
        return torch.float32
    elif dtype == "float64":
        return torch.float64
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def gen_tensor(
    shape: Tuple[int, ...],
    dtype: str = "float32",
    **kwargs: Dict[str, Any],
) -> torch.Tensor:
    """
    Generates a random tensor on host with the given shape. If ``dtype`` names
    an integer type, then the tensor elements are chosen uniform randomly from
    the range [-5, 5] by default or a user-specified range (taken from
    ``kwargs``). If ``dtype`` names a floating-point type, then the tensor
    elements are chosen from the standard normal distribution.

    Arguments:
        shape (Tuple[int, ...]): Shape of the tensor to generate.
        dtype (Union[str, torch.dtype, np.dtype]): Data type of the tensor to
            generate. (Default: "float32")
        kwargs (Dict[str, Any]): Keyword arguments to forward to the
            constructor---namely, if ``dtype`` names an integer type, then
            ``kwargs`` can specify ``low`` and ``high`` to define the range
            [``low``, ``high``) from which the random tensor elements are
            generated.
    """
    make_deterministic()
    is_integer_dtype = str(dtype).find("int") >= 0
    if is_integer_dtype:
        low = kwargs.get("low", -5)
        high = kwargs.get("high", 6)
        assert high > low, f"Invalid range: [{low}, {high})"
        np_array = np.random.randint(
            low=low,
            high=high,
            size=shape,
            dtype=dtype,
        )
    else:
        np_array = np.random.randn(*shape)
    return torch.from_numpy(np_array).to(str_dtype_to_torch_dtype(dtype))


class TensorAlignmentData(NamedTuple):
    """
    This contains the data for aligning FlexFlow and PyTorch on a tensor
    quantity. It includes a pair of filepaths (``ff_filepath`` and
    ``torch_filepath``) to PyTorch tensors (saved as ``.pt`` files)
    representing the FlexFlow and PyTorch versions of the tensor quantity
    given by ``tensor_name``.
    """
    tensor_name: str
    ff_filepath: str
    torch_filepath: str


def align_tensors(tensor_alignment_data_iter: Iterable[TensorAlignmentData]):
    """
    Checks the alignment between tensors specified by
    ``tensor_alignment_data_iter``. Each element in the iterable specifies a
    single tensor quantity to align between FlexFlow and PyTorch.
    """
    for tensor_alignment_data in tensor_alignment_data_iter:
        ff_filepath = tensor_alignment_data.ff_filepath
        torch_filepath = tensor_alignment_data.torch_filepath
        assert os.path.exists(ff_filepath), \
            f"Missing FlexFlow tensor at {ff_filepath}"
        assert os.path.exists(torch_filepath), \
            f"Missing PyTorch tensor at {torch_filepath}"
        ff_tensor = torch.load(ff_filepath).cpu()
        torch_tensor = torch.load(torch_filepath).cpu()
        print(f"Checking {tensor_alignment_data.tensor_name} alignment...")
        torch.testing.assert_close(ff_tensor, torch_tensor, rtol=1e-2, atol=1e-4)


def parse_create_tensor_args():
    """
    get operator name from command line for creating tensors
    """
    parser = ArgumentParser(description='Pytorch Aligment Test Suite')
    parser.add_argument("-o", "--operator", dest="operator",
                        required=False, metavar="", help="operator needs to be test")
    parser.add_argument(
        "-config-file",
        help="The path to a JSON file with the configs. If omitted, a sample model and configs will be used instead.",
        type=str,
        default=None,
    )
    args, unknown = parser.parse_known_args()
    return args

def create_general_test_tensor_torch() -> torch.Tensor:
    """
    generate general input size of alignment tests
    """
    tensor: torch.Tensor = gen_tensor(
        (BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE),
        dtype="float32"
    )
    return tensor