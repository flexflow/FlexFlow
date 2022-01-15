from typing import Any, Dict, Tuple

import numpy as np
import torch


def make_deterministic(seed: int = 42) -> None:
    """Makes ensuing runs determinstic by setting seeds and using deterministic
    backends."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def str_dtype_to_torch_dtype(dtype: str):
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
    shape: Tuple[int, ...], dtype: str = "float32", **kwargs: Dict[str, Any],
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
            low=low, high=high, size=shape, dtype=dtype,
        )
    else:
        np_array = np.random.randn(*shape)
    return torch.from_numpy(np_array)


def diff_tensors(t1: torch.Tensor, t2: torch.Tensor):
    """Compares two tensors for equality."""
    assert t1.shape == t2.shape, f"Shape mismatch: t1={t1.shape} t2={t2.shape}"
    assert t1.dtype == t2.dtype, f"dtype mismatch: t1={t1.dtype} t2={t2.dtype}"
    diff = t1 - t2
    assert torch.allclose(t1, t2), f"diff={diff}\n" \
        "Number of mismatched elements: " \
        f"{torch.count_nonzero(diff)}/{diff.numel()}"
