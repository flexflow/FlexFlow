import os, re, torch
import numpy as np
from typing import List
from enum import Enum
from dataclasses import dataclass

abs_dirname = os.path.dirname(os.path.abspath(__file__))
cache_folder = os.path.expanduser(os.getenv("FF_CACHE_PATH", "~/.cache/flexflow"))
hf_path = os.path.join(cache_folder, "debug/huggingface")
ff_path = os.path.join(cache_folder, "debug/flexflow")


def print_unique_files_list(dirname):
    files_list = os.listdir(dirname)
    for f in sorted(files_list):
        match = re.search(r"layers.\d+", f)
        if match:
            if "layers." in match[0]:
                layer_num = int(match[0].split(".")[1])
                if layer_num > 0:
                    files_list.remove(f)
            elif "layers_" in match[0]:
                layer_num = int(match[0].split("_")[1])
                if layer_num > 0 and layer_num != 100:
                    files_list.remove(f)
    return sorted(files_list)


def compare_tensors(hf_tensor_filepath: str, ff_tensor_filepath: str, tolerance=1e-2):
    """Check whether a HuggingFace tensor and a FlexFlow tensor are equal

    Args:
        hf_tensor_filepath (str): The file path of the HuggingFace tensor
        ff_tensor_filepath (str): The file path of the FlexFlow tensor
        tolerance (float, optional): Floating-point error tolerance for the checks. Defaults to 1e-2.

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
    """
    if not os.path.exists(hf_tensor_filepath):
        raise FileNotFoundError(f"HF tensor file: {hf_tensor_filepath} not found")
    if not os.path.exists(ff_tensor_filepath):
        raise FileNotFoundError(f"FF tensor file {ff_tensor_filepath} not found")
    hf_tensor = torch.load(hf_tensor_filepath)
    if type(hf_tensor) == tuple or type(hf_tensor) == list:
        assert len(hf_tensor) == 1
        hf_tensor = hf_tensor[0]
    hf_tensor = torch.nan_to_num(hf_tensor)
    hf_tensor = hf_tensor.flatten().detach().cpu().numpy()
    ff_tensor = np.loadtxt(ff_tensor_filepath, delimiter=",")

    len_hf_tensor = hf_tensor.shape[0]
    ff_tensor = ff_tensor[:len_hf_tensor]

    mismatches = []
    if not np.allclose(ff_tensor, hf_tensor, atol=tolerance):
        print(f"mismatch between {hf_tensor_filepath} and {ff_tensor_filepath}")
        print(f"HF: {hf_tensor}\nFF:{ff_tensor}")
        print(np.isclose(ff_tensor, hf_tensor, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor, hf_tensor, atol=tolerance))[0]
        print(mismatches)
        # print(np.nonzero(hf_tensor)[0])
        # print(np.where(np.isclose(ff_tensor, hf_tensor, atol=tolerance) ==0)[0])
        # print(ff_tensor[36], hf_tensor[36])
    # assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert len(mismatches) <= 0.05 * len_hf_tensor
    print("Ok!")


def compare_tensors_difference(
    hf_tensor_filepath: str,
    ff_tensor1_filepath: str,
    ff_tensor2_filepath: str,
    tolerance: float = 1e-2,
):
    """Check whether a HuggingFace tensor is equal to the difference between two FlexFlow tensors

    Args:
        hf_tensor_filepath (str): The file path of the HuggingFace tensor
        ff_tensor1_filepath (str): The file path of the first FlexFlow tensor
        ff_tensor2_filepath (str): The file path of the second FlexFlow tensor
        tolerance (float, optional): The floating-point error tolerance for the equality check. Defaults to 1e-2.
    """
    assert os.path.exists(hf_tensor_filepath)
    assert os.path.exists(ff_tensor1_filepath)
    assert os.path.exists(ff_tensor2_filepath)
    hf_tensor = torch.load(hf_tensor_filepath)
    if type(hf_tensor) == tuple or type(hf_tensor) == list:
        assert len(hf_tensor) == 1
        hf_tensor = hf_tensor[0]
    hf_tensor = torch.nan_to_num(hf_tensor)
    hf_tensor = hf_tensor.flatten().detach().cpu().numpy()
    ff_tensor1 = np.loadtxt(ff_tensor1_filepath, delimiter=",")
    ff_tensor2 = np.loadtxt(ff_tensor2_filepath, delimiter=",")

    len_hf_tensor = hf_tensor.shape[0]
    ff_tensor1 = ff_tensor1[:len_hf_tensor]
    ff_tensor2 = ff_tensor2[:len_hf_tensor]
    ff_tensor = ff_tensor1 - ff_tensor2

    mismatches = []
    if not np.allclose(ff_tensor, hf_tensor, atol=tolerance):
        print(
            f"mismatch between {hf_tensor_filepath} and {ff_tensor1_filepath} - {ff_tensor2_filepath}"
        )
        print(f"HF: {hf_tensor}\nFF:{ff_tensor}")
        print(np.isclose(ff_tensor, hf_tensor, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor, hf_tensor, atol=tolerance))[0]
        print(mismatches)
        # print(np.nonzero(hf_tensor)[0])
        # print(np.where(np.isclose(ff_tensor, hf_tensor, atol=tolerance) ==0)[0])
        # print(ff_tensor[36], hf_tensor[36])
    # assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert len(mismatches) <= 0.05 * len_hf_tensor
    print("Ok!")


def compare_hf_tensors(tensor1_fp: str, tensor2_fp: str):
    """Checks whether two HuggingFace tensors are equal

    Args:
        tensor1_fp (str): The file path of the first tensor
        tensor2_fp (str): The file path of the second tensor
    """
    if not os.path.exists(tensor1_fp):
        raise FileNotFoundError(f"HF tensor file: {tensor1_fp} not found")
    if not os.path.exists(tensor2_fp):
        raise FileNotFoundError(f"HF tensor file {tensor2_fp} not found")
    hf_tensor1 = torch.load(tensor1_fp)
    hf_tensor2 = torch.load(tensor2_fp)
    if type(hf_tensor1) == tuple or type(hf_tensor1) == list:
        assert len(hf_tensor1) == 1
        hf_tensor1 = hf_tensor1[0]
    if type(hf_tensor2) == tuple or type(hf_tensor2) == list:
        assert len(hf_tensor2) == 1
        hf_tensor2 = hf_tensor2[0]
    assert torch.squeeze(hf_tensor1).shape == torch.squeeze(hf_tensor2).shape
    hf_tensor1 = torch.nan_to_num(hf_tensor1)
    hf_tensor2 = torch.nan_to_num(hf_tensor2)
    if not (
        np.allclose(
            hf_tensor1.detach().cpu().numpy(), hf_tensor2.detach().cpu().numpy()
        )
    ):
        print(f"mismatch between {tensor1_fp} and {tensor2_fp}")
        print(hf_tensor1)
        print(hf_tensor2)
        print(
            np.isclose(
                hf_tensor1.detach().cpu().numpy(), hf_tensor2.detach().cpu().numpy()
            )
        )
        mismatches = np.where(
            ~np.isclose(
                hf_tensor1.detach().cpu().numpy(), hf_tensor2.detach().cpu().numpy()
            )
        )[0]
        print(mismatches)
        assert False
    print("Ok!")


def check_hf_sum_tensors(tensor_sum_fp: str, tensor1_fp: str, tensor2_fp: str):
    """Checks whether a HuggingFace tensor is equal to the sum of two other HuggingFace tensors

    Args:
        tensor_sum_fp (str): The file path of the sum tensor
        tensor1_fp (str): The file path of the first tensor
        tensor2_fp (str): The file path of the second tensor
    """
    if not os.path.exists(tensor_sum_fp):
        raise FileNotFoundError(f"HF tensor file: {tensor_sum_fp} not found")
    if not os.path.exists(tensor1_fp):
        raise FileNotFoundError(f"HF tensor file {tensor1_fp} not found")
    if not os.path.exists(tensor2_fp):
        raise FileNotFoundError(f"HF tensor file {tensor2_fp} not found")
    hf_tensor_sum = torch.load(tensor_sum_fp)
    hf_tensor1 = torch.load(tensor1_fp)
    hf_tensor2 = torch.load(tensor2_fp)
    if type(hf_tensor_sum) == tuple or type(hf_tensor_sum) == list:
        assert len(hf_tensor_sum) == 1
        hf_tensor_sum = hf_tensor_sum[0]
    if type(hf_tensor1) == tuple or type(hf_tensor1) == list:
        assert len(hf_tensor1) == 1
        hf_tensor1 = hf_tensor1[0]
    if type(hf_tensor2) == tuple or type(hf_tensor2) == list:
        assert len(hf_tensor2) == 1
        hf_tensor2 = hf_tensor2[0]
    assert torch.squeeze(hf_tensor_sum).shape == torch.squeeze(hf_tensor1).shape
    assert torch.squeeze(hf_tensor1).shape == torch.squeeze(hf_tensor2).shape
    hf_tensor1 = torch.nan_to_num(hf_tensor1)
    hf_tensor2 = torch.nan_to_num(hf_tensor2)
    hf_tensor_sum = torch.nan_to_num(hf_tensor_sum)
    sum_check_tensor = hf_tensor1 + hf_tensor2
    if not (
        np.allclose(
            sum_check_tensor.detach().cpu().numpy(),
            hf_tensor_sum.detach().cpu().numpy(),
        )
    ):
        print(f"mismatch between {sum_check_tensor} and {tensor1_fp} + {tensor2_fp}")
        print(tensor_sum_fp)
        print(sum_check_tensor)
        print(hf_tensor1)
        print(hf_tensor2)
        print(
            np.isclose(
                sum_check_tensor.detach().cpu().numpy(),
                hf_tensor_sum.detach().cpu().numpy(),
            )
        )
        mismatches = np.where(
            ~np.isclose(
                sum_check_tensor.detach().cpu().numpy(),
                hf_tensor_sum.detach().cpu().numpy(),
            )
        )[0]
        print(mismatches)
        assert False
    print("Ok!")


def check_hf_zero_tensor(hf_tensor_fp: str):
    """Check whether a HuggingFace tensor is a zero tensor

    Args:
        hf_tensor_fp (str): The file path of the HuggingFace tensor
    """
    if not os.path.exists(hf_tensor_fp):
        raise FileNotFoundError(f"HF tensor file: {hf_tensor_fp} not found")
    hf_tensor1 = torch.load(hf_tensor_fp)
    if type(hf_tensor1) == tuple or type(hf_tensor1) == list:
        assert len(hf_tensor1) == 1
        hf_tensor1 = hf_tensor1[0]
    assert torch.count_nonzero(torch.nan_to_num(hf_tensor1)).sum() == 0


def print_tensors(hf_tensor_filepath: str, ff_tensor_filepath: str, txt: str = ""):
    """Print the contents of a HuggingFace tensor and a FlexFlow tensor

    Args:
        hf_tensor_filepath (str): The file path of the HuggingFace tensor
        ff_tensor_filepath (str): The file path of the FlexFlow tensor
        txt (str, optional): Additional text to prepend to the tensors. Defaults to "".
    """
    assert os.path.exists(hf_tensor_filepath) and os.path.exists(ff_tensor_filepath)
    hf_tensor = torch.load(hf_tensor_filepath)
    if type(hf_tensor) == tuple or type(hf_tensor) == list:
        assert len(hf_tensor) == 1
        hf_tensor = hf_tensor[0]
    hf_tensor = torch.nan_to_num(hf_tensor)
    hf_tensor = hf_tensor.flatten().detach().cpu().numpy()
    ff_tensor = np.loadtxt(ff_tensor_filepath, delimiter=",")

    len_hf_tensor = hf_tensor.shape[0]
    ff_tensor = ff_tensor[:len_hf_tensor]

    print(f"{txt} - HF tensor:")
    print(hf_tensor)
    print(f"{txt} - FF tensor: ")
    print(ff_tensor)


def compare_flexflow_tensors(
    ff_tensor1_fp: str, ff_tensor2_fp: str, tolerance: float = 1e-5, max_len: int = -1
):
    """Check whether two FlexFlow tensors are equal

    Args:
        ff_tensor1_fp (str): The file path of the first FlexFlow tensor
        ff_tensor2_fp (str): The file path of the second FlexFlow tensor
        tolerance (float, optional): Floating-point error tolernace for the check. Defaults to 1e-5.
        max_len (int, optional): Maximum number of elements to check (if > 0). Defaults to -1.

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
    """
    if not os.path.exists(ff_tensor1_fp):
        raise FileNotFoundError(f"FF tensor file: {ff_tensor1_fp} not found")
    if not os.path.exists(ff_tensor2_fp):
        raise FileNotFoundError(f"FF tensor file {ff_tensor2_fp} not found")
    assert os.path.exists(ff_tensor1_fp) and os.path.exists(ff_tensor2_fp)
    ff_tensor1 = np.loadtxt(ff_tensor1_fp, delimiter=",")
    ff_tensor2 = np.loadtxt(ff_tensor2_fp, delimiter=",")

    if ff_tensor1.shape != ff_tensor2.shape:
        print(ff_tensor1.shape, ff_tensor2.shape)
    assert ff_tensor1.shape == ff_tensor2.shape

    if max_len > -1:
        ff_tensor1 = ff_tensor1[:max_len]
        ff_tensor2 = ff_tensor2[:max_len]

    mismatches = []
    if not np.allclose(ff_tensor1, ff_tensor2, atol=tolerance):
        print(f"mismatch between {ff_tensor1_fp} and {ff_tensor2_fp}")
        print(f"Tensor1: {ff_tensor1}\nTensor2:{ff_tensor2}")
        print(np.isclose(ff_tensor1, ff_tensor2, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor1, ff_tensor2, atol=tolerance))[0]
        print(mismatches)
    # assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert len(mismatches) <= 0.05 * len(ff_tensor1)
    print("Ok!")


def compare_flexflow_tensors_shortest(
    ff_tensor1_fp: str, ff_tensor2_fp: str, tolerance: float = 1e-5
):
    """Compare two FlexFlow tensors up to the maximum length of the shortest tensor

    Args:
        ff_tensor1_fp (str): The file path of the first FlexFlow tensor
        ff_tensor2_fp (str): The file path of the second FlexFlow tensor
        tolerance (float, optional): Floating point error tolerance for the check. Defaults to 1e-5.

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
    """
    if not os.path.exists(ff_tensor1_fp):
        raise FileNotFoundError(f"FF tensor file: {ff_tensor1_fp} not found")
    if not os.path.exists(ff_tensor2_fp):
        raise FileNotFoundError(f"FF tensor file {ff_tensor2_fp} not found")
    ff_tensor1 = np.loadtxt(ff_tensor1_fp, delimiter=",")
    ff_tensor2 = np.loadtxt(ff_tensor2_fp, delimiter=",")
    minlen = min(ff_tensor1.shape[0], ff_tensor2.shape[0])
    ff_tensor1 = ff_tensor1[:minlen]
    ff_tensor2 = ff_tensor2[:minlen]
    mismatches = []
    if not np.allclose(ff_tensor1, ff_tensor2, atol=tolerance):
        print(f"mismatch between {ff_tensor1_fp} and {ff_tensor2_fp}")
        print(f"Tensor1: {ff_tensor1}\nTensor2:{ff_tensor2}")
        print(np.isclose(ff_tensor1, ff_tensor2, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor1, ff_tensor2, atol=tolerance))[0]
        print(mismatches)
    # assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert len(mismatches) <= 0.05 * len(ff_tensor1)
    print("Ok!")


def check_flexflow_tensors_sum(
    ff_tensor_sum_fp: str, ff_tensor1_fp: str, ff_tensor2_fp: str, tolerance=1e-5
):
    """Check whether a FlexFlow tensor is equal to the sum of two other FlexFlow tensors

    Args:
        ff_tensor_sum_fp (str): The file path of the FlexFlow sum tensor
        ff_tensor1_fp (str): The file path of the first FlexFlow tensor
        ff_tensor2_fp (str): The file path of the second FlexFlow tensor
        tolerance (_type_, optional): Floating-point error tolerance for the check. Defaults to 1e-5.

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
    """
    if not os.path.exists(ff_tensor1_fp):
        raise FileNotFoundError(f"FF tensor file: {ff_tensor1_fp} not found")
    if not os.path.exists(ff_tensor2_fp):
        raise FileNotFoundError(f"FF tensor file {ff_tensor2_fp} not found")
    ff_tensor1 = np.loadtxt(ff_tensor1_fp, delimiter=",")
    ff_tensor2 = np.loadtxt(ff_tensor2_fp, delimiter=",")
    ff_tensor_sum = np.loadtxt(ff_tensor_sum_fp, delimiter=",")

    ff_sum = ff_tensor1 + ff_tensor2
    assert ff_tensor1.shape == ff_tensor2.shape

    mismatches = []
    if not np.allclose(ff_tensor_sum, ff_sum, atol=tolerance):
        print(
            f"mismatch between {ff_tensor_sum_fp} and sum of {ff_tensor1_fp} + {ff_tensor2_fp}"
        )
        print(f"Tensor1: {ff_tensor1}\nTensor2:{ff_tensor2}")
        print(f"Sum Tensor: {ff_tensor_sum}\nActual sum:{ff_sum}")
        print(np.isclose(ff_tensor_sum, ff_sum, atol=tolerance))
        mismatches = np.where(~np.isclose(ff_tensor_sum, ff_sum, atol=tolerance))[0]
        print(mismatches)
    # assert(np.allclose(ff_tensor, hf_tensor, atol=tolerance))
    assert len(mismatches) <= 0.05 * len(ff_tensor1)
    print("Ok!")


def load_ff_tensor(filename: str, shape: List[int]):
    """Load a FlexFlow tensor from a file as a numpy array

    Args:
        filename (str): The file path of the FF tensor
        shape (List[int]): The shape of the FF tensor

    Returns:
        _type_: The FF tensor as a numpy array
    """
    if ff_path not in filename:
        filename = os.path.join(ff_path, filename)
    ff_tensor = np.loadtxt(filename, delimiter=",").reshape(shape, order="F")
    return ff_tensor


def load_hf_tensor(filename: str):
    """Load a HuggingFace tensor from a file as a numpy array

    Args:
        filename (str): The file path of the HF tensor

    Returns:
        _type_: The HF tensor as a numpy array
    """
    if hf_path not in filename:
        filename = os.path.join(hf_path, filename)
    hf_tensor = torch.load(filename)
    hf_tensor = hf_tensor.detach().cpu().numpy()
    return hf_tensor


def compare_loaded_tensors(hf_tensor, ff_tensor, tolerance=1e-2):
    """Check whether a Huggingface and a FlexFlow tensors, both loaded to memory in the form of a numpy array, are equal

    Args:
        hf_tensor (_type_): The HuggingFace tensor (in numpy array form)
        ff_tensor (_type_): The FlexFlow tensor (in numpy array form)
        tolerance (_type_, optional): The floating point error tolerance for the check. Defaults to 1e-2.
    """
    assert hf_tensor.shape == ff_tensor.shape
    mismatches = []
    if not np.allclose(hf_tensor, ff_tensor, atol=tolerance):
        print(f"mismatch between hf_tensor and ff_tensor")
        print(f"HF: {hf_tensor}\nFF:{ff_tensor}")
        print(np.isclose(hf_tensor, ff_tensor, atol=tolerance))
        mismatches = np.where(~np.isclose(hf_tensor, ff_tensor, atol=tolerance))[0]
        print(mismatches)
    len_hf_tensor = hf_tensor.flatten().shape[0]
    assert len(mismatches) <= 0.05 * len_hf_tensor
    print("Ok!")


def are_np_arrays_identical(*np_arrays):
    if len(np_arrays) < 2:
        return True

    first = np_arrays[0]

    # Check shapes and dtypes
    if not all(
        t.shape == first.shape and t.dtype == first.dtype for t in np_arrays[1:]
    ):
        return False

    # Stack all tensors along a new axis
    stacked = np.stack(np_arrays)

    # Check if all elements along the new axis are equal
    return np.all(stacked == stacked[0])


class TPType(Enum):
    REPLICATE = 0
    PARTITION = 1
    TO_REDUCE = 2


@dataclass
class TensorComparisonIdxs:
    hf_tensor_type: str
    ff_tensor_type: str
    hf_tensor_idx: int
    ff_tensor_idx: int


def replace_value(lst, old_value, new_value):
    occurrences = lst.count(old_value)
    if occurrences == 0:
        raise ValueError(f"Value {old_value} not found in the list.")
    elif occurrences > 1:
        raise ValueError(f"Multiple instances of {old_value} found in the list.")
    else:
        index = lst.index(old_value)
        lst[index] = new_value
        return lst


def truncate_dimension(tensor, old_dim, new_dim):
    # Check if old_dim appears exactly once in the tensor's shape
    shape = tensor.shape
    dim_occurrences = shape.count(old_dim)

    if dim_occurrences == 0:
        raise ValueError(f"Dimension {old_dim} not found in the tensor shape.")
    elif dim_occurrences > 1:
        raise ValueError(
            f"Multiple instances of dimension {old_dim} found in the tensor shape."
        )

    # Check if new_dim is less than or equal to old_dim
    if new_dim > old_dim:
        raise ValueError(
            f"New dimension ({new_dim}) must be less than or equal to old dimension ({old_dim})."
        )

    # Find the index of the dimension to truncate
    dim_index = shape.index(old_dim)

    # Create a slice object for truncation
    slices = [slice(None)] * len(shape)
    slices[dim_index] = slice(0, new_dim)

    # Truncate the tensor
    truncated_tensor = tensor[tuple(slices)]

    return truncated_tensor
